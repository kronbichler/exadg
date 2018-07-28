/*
 * HelmholtzOperator.h
 *
 *  Created on: May 11, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_HELMHOLTZ_OPERATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_HELMHOLTZ_OPERATOR_H_

// TODO
#include <deal.II/base/timer.h>

#include "../../incompressible_navier_stokes/spatial_discretization/navier_stokes_operators.h"
#include "../../solvers_and_preconditioners/invert_diagonal.h"
#include "../../solvers_and_preconditioners/util/verify_calculation_of_diagonal.h"

namespace IncNS
{

template<int dim>
struct HelmholtzOperatorData
{
  HelmholtzOperatorData ()
    :
    unsteady_problem(true),
    dof_index(0),
    bc(new BoundaryDescriptorP<dim>()),
    scaling_factor_time_derivative_term(-1.0)
  {}

  bool unsteady_problem;

  unsigned int dof_index;
  
  // TODO
  std::shared_ptr<BoundaryDescriptorP<dim>> bc;
  double scaling_factor_time_derivative_term;
  MassMatrixOperatorData mass_matrix_operator_data;
  ViscousOperatorData<dim> viscous_operator_data;
};

template <int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number = double>
class HelmholtzOperator : public MatrixOperatorBaseNew<dim, Number>
{
public:
    // Issue#2:
  typedef Number value_type;
  static const int DIM = dim;

  static const bool is_xwall = (xwall_quad_rule>1) ? true : false;
  static const unsigned int n_actual_q_points_vel_linear = (is_xwall) ? xwall_quad_rule : fe_degree+1;
  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,
                              dim,value_type,is_xwall> FEEval_Velocity_Velocity_linear;

  typedef HelmholtzOperator<dim,fe_degree,fe_degree_xwall,xwall_quad_rule,Number> This;

  HelmholtzOperator()
    :
    block_jacobi_matrices_have_been_initialized(false),
    data(nullptr),
    mass_matrix_operator(nullptr),
    viscous_operator(nullptr),
    wall_time(0.0),
    use_optimized_implementation(false) //TODO: use optimized implementation for performance measurements only
  {}

  //TODO
  double get_wall_time() const
  {
    return wall_time;
  }


  void initialize(MatrixFree<dim,Number> const                                                        &mf_data_in,
                  HelmholtzOperatorData<dim> const                                                    &operator_data_in,
                  MassMatrixOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>  const &mass_matrix_operator_in,
                  ViscousOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> const     &viscous_operator_in)
  {
    // copy parameters into element variables
    this->data = &mf_data_in;
    this->operator_data = operator_data_in;
    this->mass_matrix_operator = &mass_matrix_operator_in;
    this->viscous_operator = &viscous_operator_in;
  }

  /*
   *  This function is called by the multigrid algorithm to initialize the
   *  matrices on all levels. To construct the matrices, and object of
   *  type UnderlyingOperator is used that provides all the information for
   *  the setup, i.e., the information that is needed to call the
   *  member function initialize(...).
   */
  void reinit (const DoFHandler<dim>                            &dof_handler,
               const Mapping<dim>                                &mapping,
               void * operator_data_in,
               const MGConstrainedDoFs &/*mg_constrained_dofs*/,
               const unsigned int level)
  {
    auto &add = *static_cast<HelmholtzOperatorData<dim> *>(operator_data_in);
    // create copy of data and ...
    auto operator_data = add;
      
    // setup own matrix free object
    const QGauss<1> quad(dof_handler.get_fe().degree+1);
    typename MatrixFree<dim,Number>::AdditionalData addit_data;
    addit_data.tasks_parallel_scheme = MatrixFree<dim,Number>::AdditionalData::none;
    if (dof_handler.get_fe().dofs_per_vertex == 0)
      addit_data.build_face_info = true;
    addit_data.level_mg_handler = level;

    ConstraintMatrix constraints;

    // reinit
    own_matrix_free_storage.reinit(mapping, dof_handler, constraints, quad, addit_data);

    // setup own mass matrix operator
    MassMatrixOperatorData& mass_matrix_operator_data = operator_data.mass_matrix_operator_data;
    mass_matrix_operator_data.dof_index = 0;
    own_mass_matrix_operator_storage.initialize(own_matrix_free_storage,mass_matrix_operator_data);

    // setup own viscous operator
    ViscousOperatorData<dim>& viscous_operator_data = operator_data.viscous_operator_data;
    // set dof index to zero since matrix free object only contains one dof-handler
    viscous_operator_data.dof_index = 0;
    own_viscous_operator_storage.initialize(mapping,own_matrix_free_storage, viscous_operator_data);

    // setup Helmholtz operator
    initialize(own_matrix_free_storage, operator_data, own_mass_matrix_operator_storage, own_viscous_operator_storage);

    // Initialize other variables:

    // mass matrix term: set scaling factor time derivative term
    // set_scaling_factor_time_derivative_term(underlying_operator.get_scaling_factor_time_derivative_term());

    // viscous term:

  }

  /*
   *  Scaling factor of time derivative term (mass matrix term)
   */
  void set_scaling_factor_time_derivative_term(double const &factor)
  {
    this->operator_data.scaling_factor_time_derivative_term = factor;
  }

  double get_scaling_factor_time_derivative_term() const
  {
    return this->operator_data.scaling_factor_time_derivative_term;
  }

  /*
   *  Operator data
   */
  HelmholtzOperatorData<dim> const & get_operator_data() const
  {
    return this->operator_data;
  }

  /*
   *  Operator data of basic operators: mass matrix, viscous operator
   */
  MassMatrixOperatorData const & get_mass_matrix_operator_data() const
  {
    return mass_matrix_operator->get_operator_data();
  }

  ViscousOperatorData<dim> const & get_viscous_operator_data() const
  {
    return viscous_operator->get_operator_data();
  }

  /*
   *  This function does nothing in case of the velocity conv diff operator.
   *  IT is only necessary due to the interface of the multigrid preconditioner
   *  and especially the coarse grid solver that calls this function.
   */
  void apply_nullspace_projection(parallel::distributed::Vector<Number> &/*vec*/) const {}

  /*
   *  Other function needed in order to apply geometric multigrid to this operator
   */
  void vmult_interface_down(parallel::distributed::Vector<Number>       &dst,
                            const parallel::distributed::Vector<Number> &src) const
  {
    vmult(dst,src);
  }

  void vmult_add_interface_up(parallel::distributed::Vector<Number>       &dst,
                              const parallel::distributed::Vector<Number> &src) const
  {
    vmult_add(dst,src);
  }

  types::global_dof_index m() const
  {
    return data->get_vector_partitioner(get_dof_index())->size();
  }

  Number el (const unsigned int,  const unsigned int) const
  {
    AssertThrow(false, ExcMessage("Matrix-free does not allow for entry access"));
    return Number();
  }

  MatrixFree<dim,value_type> const & get_data() const
  {
    return *data;
  }

  /*
   *  This function applies the matrix vector multiplication.
   */
  void vmult (parallel::distributed::Vector<Number>       &dst,
              const parallel::distributed::Vector<Number> &src) const
  {
    // TODO
    Timer timer;
    timer.restart();

    if(use_optimized_implementation == true) // optimized version (use only for performance measurements)
    {
      viscous_operator->apply_helmholtz_operator(dst,this->get_scaling_factor_time_derivative_term(),src);
    }
    else // standard implementation with modular implementation (operator by operator)
    {
      // helmholtz operator = mass_matrix_operator + viscous_operator
      if(operator_data.unsteady_problem == true)
      {
        AssertThrow(this->get_scaling_factor_time_derivative_term() > 0.0,
          ExcMessage("Scaling factor of time derivative term has not been initialized for Helmholtz operator!"));

        mass_matrix_operator->apply_scale(dst,this->get_scaling_factor_time_derivative_term(),src);
      }
      else
      {
        dst = 0.0;
      }

      viscous_operator->apply_add(dst,src);
    }

    // TODO
    wall_time += timer.wall_time();
  }

  /*
   *  This function applies the matrix-vector product and adds the result
   *  to the dst-vector.
   */
  void vmult_add(parallel::distributed::Vector<Number>       &dst,
                 const parallel::distributed::Vector<Number> &src) const
  {
    // TODO
    Timer timer;
    timer.restart();

    // helmholtz operator = mass_matrix_operator + viscous_operator
    if(operator_data.unsteady_problem == true)
    {
      AssertThrow(this->get_scaling_factor_time_derivative_term() > 0.0,
        ExcMessage("Scaling factor of time derivative term has not been initialized for Helmholtz operator!"));

      mass_matrix_operator->apply_scale_add(dst,this->get_scaling_factor_time_derivative_term(),src);
    }

    viscous_operator->apply_add(dst,src);

    // TODO
    wall_time += timer.wall_time();
  }

  unsigned int get_dof_index() const
  {
    return operator_data.dof_index;
  }

  /*
   *  This function initializes a global dof-vector.
   */
  void initialize_dof_vector(parallel::distributed::Vector<Number> &vector) const
  {
    data->initialize_dof_vector(vector,get_dof_index());
  }

  /*
   *  Calculation of inverse diagonal (needed for smoothers and preconditioners)
   */
  void calculate_inverse_diagonal(parallel::distributed::Vector<Number> &diagonal) const
  {
    calculate_diagonal(diagonal);

    // verify_calculation_of_diagonal(*this,diagonal);

    invert_diagonal(diagonal);
  }

  /*
   *  Apply block Jacobi preconditioner.
   */
  void apply_block_jacobi (parallel::distributed::Vector<Number>       &dst,
                           parallel::distributed::Vector<Number> const &src) const
  {
    // check_block_jacobi_matrices(src);

    data->cell_loop(&This::cell_loop_apply_inverse_block_jacobi_matrices, this, dst, src);
  }

  /*
   *  This function updates the block Jacobi preconditioner.
   *  Since this function also initializes the block Jacobi preconditioner,
   *  make sure that the block Jacobi matrices are allocated before calculating
   *  the matrices and the LU factorization.
   */
  void update_block_jacobi () const
  {
    if(block_jacobi_matrices_have_been_initialized == false)
    {
      // Note that the velocity has dim components.
      unsigned int dofs_per_cell = data->get_shape_info().dofs_per_component_on_cell*dim;

      matrices.resize(data->n_macro_cells()*VectorizedArray<Number>::n_array_elements,
        LAPACKFullMatrix<Number>(dofs_per_cell, dofs_per_cell));

      block_jacobi_matrices_have_been_initialized = true;
    }

    calculate_block_jacobi_matrices();
    calculate_lu_factorization_block_jacobi(matrices);
  }

private:
  /*
   *  This function calculates the diagonal of the discrete operator representing the
   *  velocity convection-diffusion operator.
   */
  void calculate_diagonal(parallel::distributed::Vector<Number> &diagonal) const
  {
    if(operator_data.unsteady_problem == true)
    {
      AssertThrow(this->get_scaling_factor_time_derivative_term() > 0.0,
        ExcMessage("Scaling factor of time derivative term has not been initialized for Helmholtz operator!"));

      mass_matrix_operator->calculate_diagonal(diagonal);
      diagonal *= this->get_scaling_factor_time_derivative_term();
    }
    else
    {
      diagonal = 0.0;
    }

    viscous_operator->add_diagonal(diagonal);
  }

  /*
   * This function calculates the block Jacobi matrices.
   * This is done sequentially for the different operators.
   */
  void calculate_block_jacobi_matrices() const
  {
    // initialize block Jacobi matrices with zeros
    initialize_block_jacobi_matrices_with_zero(matrices);

    // calculate block Jacobi matrices
    if(operator_data.unsteady_problem == true)
    {
      AssertThrow(this->get_scaling_factor_time_derivative_term() > 0.0,
        ExcMessage("Scaling factor of time derivative term has not been initialized for Helmholtz operator!"));

      mass_matrix_operator->add_block_jacobi_matrices(matrices);

      for(typename std::vector<LAPACKFullMatrix<Number> >::iterator
          it = matrices.begin(); it != matrices.end(); ++it)
      {
        (*it) *= this->get_scaling_factor_time_derivative_term();
      }
    }

    viscous_operator->add_block_jacobi_matrices(matrices);
  }

  /*
   *  This function loops over all cells and applies the inverse block Jacobi matrices elementwise.
   */
  void cell_loop_apply_inverse_block_jacobi_matrices (MatrixFree<dim,Number> const                &data,
                                                      parallel::distributed::Vector<Number>       &dst,
                                                      parallel::distributed::Vector<Number> const &src,
                                                      std::pair<unsigned int,unsigned int> const  &cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval(data,viscous_operator->get_fe_param(),operator_data.dof_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);

      unsigned int dofs_per_cell = fe_eval.dofs_per_cell;

      for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
      {
        // fill source vector
        Vector<Number> src_vector(dofs_per_cell);
        for (unsigned int j=0; j<dofs_per_cell; ++j)
          src_vector(j) = fe_eval.begin_dof_values()[j][v];

        // apply inverse matrix
        matrices[cell*VectorizedArray<Number>::n_array_elements+v].solve(src_vector,false);

        // write solution to dst-vector
        for (unsigned int j=0; j<dofs_per_cell; ++j)
          fe_eval.begin_dof_values()[j][v] = src_vector(j);
      }

      fe_eval.set_dof_values (dst);
    }
  }

  /*
   * Verify computation of block Jacobi matrices.
   */
   void check_block_jacobi_matrices(parallel::distributed::Vector<Number> const &src) const
   {
     calculate_block_jacobi_matrices();

     // test matrix-vector product for block Jacobi problem by comparing
     // matrix-free matrix-vector product and matrix-based matrix-vector product
     // (where the matrices are generated using the matrix-free implementation)
     parallel::distributed::Vector<Number> tmp1(src), tmp2(src), diff(src);
     tmp1 = 0.0;
     tmp2 = 0.0;

     // variant 1 (matrix-free)
     vmult_block_jacobi(tmp1,src);

     // variant 2 (matrix-based)
     vmult_block_jacobi_test(tmp2,src);

     diff = tmp2;
     diff.add(-1.0,tmp1);

     std::cout << "L2 norm variant 1 = " << tmp1.l2_norm() << std::endl
               << "L2 norm variant 2 = " << tmp2.l2_norm() << std::endl
               << "L2 norm v2 - v1 = " << diff.l2_norm() << std::endl << std::endl;
   }

   /*
    * Apply matrix-vector multiplication (matrix-free) for global block Jacobi system.
    * Do that sequentially for the different operators.
    * This function is only needed when solving the global block Jacobi problem
    * iteratively in which case the function vmult_block_jacobi() represents
    * the "vmult()" operation of the linear system of equations.
    */
   void vmult_block_jacobi (parallel::distributed::Vector<Number>       &dst,
                            const parallel::distributed::Vector<Number> &src) const
   {
     if(operator_data.unsteady_problem == true)
     {
       AssertThrow(this->get_scaling_factor_time_derivative_term() > 0.0,
           ExcMessage("Scaling factor of time derivative term has not been initialized for Helmholtz operator!"));

       // mass matrix operator has already "block Jacobi form" in DG
       mass_matrix_operator->apply_scale(dst,this->get_scaling_factor_time_derivative_term(),src);
     }
     else
     {
       dst = 0.0;
     }

     viscous_operator->apply_block_jacobi_add(dst,src);
   }

   /*
    * Apply matrix-vector multiplication (matrix-based) for global block Jacobi system
    * by looping over all cells and applying the matrix-based matrix-vector product cellwise.
    * This function is only needed for testing.
    */
   void vmult_block_jacobi_test (parallel::distributed::Vector<Number>       &dst,
                                 parallel::distributed::Vector<Number> const &src) const
   {
     data->cell_loop(&This::cell_loop_apply_block_jacobi_matrices_test, this, dst, src);
   }

   /*
    *  This function is only needed for testing.
    */
   void cell_loop_apply_block_jacobi_matrices_test (MatrixFree<dim,Number> const                &data,
                                                    parallel::distributed::Vector<Number>       &dst,
                                                    parallel::distributed::Vector<Number> const &src,
                                                    std::pair<unsigned int,unsigned int> const  &cell_range) const
   {
     FEEval_Velocity_Velocity_linear fe_eval(data,viscous_operator->get_fe_param(),operator_data.dof_index);

     for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
     {
       fe_eval.reinit(cell);
       fe_eval.read_dof_values(src);

       unsigned int dofs_per_cell = fe_eval.dofs_per_cell;

       for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
       {
         // fill source vector
         Vector<Number> src_vector(dofs_per_cell);
         Vector<Number> dst_vector(dofs_per_cell);
         for (unsigned int j=0; j<dofs_per_cell; ++j)
           src_vector(j) = fe_eval.begin_dof_values()[j][v];

         // apply matrix-vector product
         matrices[cell*VectorizedArray<Number>::n_array_elements+v].vmult(dst_vector,src_vector,false);

         // write solution to dst-vector
         for (unsigned int j=0; j<dofs_per_cell; ++j)
           fe_eval.begin_dof_values()[j][v] = dst_vector(j);
       }

       fe_eval.set_dof_values (dst);
     }
   }
   
  virtual MatrixOperatorBaseNew<dim, Number>* get_new(unsigned int deg) const {
    switch (deg) {
#if DEGREE_1
    case 1: return new HelmholtzOperator<dim, 1, fe_degree_xwall, xwall_quad_rule, Number>();
#endif
#if DEGREE_2
    case 2: return new HelmholtzOperator<dim, 2, fe_degree_xwall, xwall_quad_rule, Number>();
#endif
#if DEGREE_3
    case 3: return new HelmholtzOperator<dim, 3, fe_degree_xwall, xwall_quad_rule, Number>();
#endif
#if DEGREE_4
    case 4: return new HelmholtzOperator<dim, 4, fe_degree_xwall, xwall_quad_rule, Number>();
#endif
#if DEGREE_5
    case 5: return new HelmholtzOperator<dim, 5, fe_degree_xwall, xwall_quad_rule, Number>();
#endif
#if DEGREE_6
    case 6: return new HelmholtzOperator<dim, 6, fe_degree_xwall, xwall_quad_rule, Number>();
#endif
#if DEGREE_7
    case 7: return new HelmholtzOperator<dim, 7, fe_degree_xwall, xwall_quad_rule, Number>();
#endif
#if DEGREE_8
    case 8: return new HelmholtzOperator<dim, 8, fe_degree_xwall, xwall_quad_rule, Number>();
#endif
#if DEGREE_9
    case 9: return new HelmholtzOperator<dim, 9, fe_degree_xwall, xwall_quad_rule, Number>();
#endif
    default:
      AssertThrow(false,
                  ExcMessage("HelmholtzOperator not implemented for this degree!"));
      return nullptr; 
          // dummy return (statement not reached)
    }
  }

  mutable std::vector<LAPACKFullMatrix<Number> > matrices;
  mutable bool block_jacobi_matrices_have_been_initialized;

  MatrixFree<dim,Number> const * data;
  MassMatrixOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>  const *mass_matrix_operator;
  ViscousOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>  const *viscous_operator;
  HelmholtzOperatorData<dim> operator_data;

  /*
   * The following variables are necessary when applying the multigrid
   * preconditioner to the Helmholtz operator. In that case, the
   * Helmholtz has to be generated for each level of the multigrid algorithm.
   * Accordingly, in a first step one has to setup own objects of
   * MatrixFree, MassMatrixOperator, ViscousOperator,
   *   e.g., own_matrix_free_storage.reinit(...);
   * and later initialize the HelmholtzOperator with these
   * ojects by setting the above pointers to the own_objects_storage,
   *   e.g., data = &own_matrix_free_storage;
   */
  MatrixFree<dim,Number> own_matrix_free_storage;
  MassMatrixOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> own_mass_matrix_operator_storage;
  ViscousOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> own_viscous_operator_storage;

  // TODO
  mutable double wall_time;

  // TODO
  bool use_optimized_implementation;
};


}

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_HELMHOLTZ_OPERATOR_H_ */
