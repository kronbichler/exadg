/*
 * DivergenceAndContinuityPenaltyOperators.h
 *
 *  Created on: 2017 M03 1
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DIVERGENCE_AND_CONTINUITY_PENALTY_OPERATORS_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DIVERGENCE_AND_CONTINUITY_PENALTY_OPERATORS_H_

namespace IncNS
{
/*
 *  Operator data
 */
struct DivergencePenaltyOperatorData
{
  DivergencePenaltyOperatorData()
    : type_penalty_parameter(TypePenaltyParameter::ConvectiveTerm),
      viscosity(0.0),
      penalty_parameter(1.0)
  {
  }

  // type of penalty parameter (viscous and/or convective terms)
  TypePenaltyParameter type_penalty_parameter;

  // kinematic viscosity
  double viscosity;

  // scaling factor
  double penalty_parameter;
};

/*
 *  Divergence penalty operator: ( div(v_h) , tau_div * div(u_h) )_Omega^e where
 *   v_h : test function
 *   u_h : solution
 *   tau_div: divergence penalty factor
 *
 *            use convective term:  tau_div_conv = K * ||U||_mean * h_eff
 *
 *                                  where h_eff = h / (k_u+1) and
 *                                  h = V_e^{1/3} with the element volume V_e
 *
 *            use viscous term:     tau_div_viscous = K * nu
 *
 *            use both terms:       tau_div = tau_div_conv + tau_div_viscous
 */
template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename value_type>
class DivergencePenaltyOperator : public BaseOperator<dim>
{
public:
  static const bool         is_xwall = (xwall_quad_rule > 1) ? true : false;
  static const unsigned int n_actual_q_points_vel_linear =
    (is_xwall) ? xwall_quad_rule : fe_degree + 1;

  typedef LinearAlgebra::distributed::Vector<value_type> VectorType;

  typedef FEEvaluationWrapper<dim,
                              fe_degree,
                              fe_degree_xwall,
                              n_actual_q_points_vel_linear,
                              dim,
                              value_type,
                              is_xwall>
    FEEval_Velocity_Velocity_linear;

  typedef DivergencePenaltyOperator<dim,
                                    fe_degree,
                                    fe_degree_p,
                                    fe_degree_xwall,
                                    xwall_quad_rule,
                                    value_type>
    This;

  DivergencePenaltyOperator(MatrixFree<dim, value_type> const & data_in,
                            unsigned int const                  dof_index_in,
                            unsigned int const                  quad_index_in,
                            DivergencePenaltyOperatorData const operator_data_in)
    : data(data_in),
      dof_index(dof_index_in),
      quad_index(quad_index_in),
      array_penalty_parameter(0),
      operator_data(operator_data_in)
  {
    array_penalty_parameter.resize(data.n_macro_cells() + data.n_macro_ghost_cells());
  }

  void
  calculate_array_penalty_parameter(VectorType const & velocity)
  {
    velocity.update_ghost_values();

    FEEval_Velocity_Velocity_linear fe_eval(data, this->fe_param, dof_index);

    AlignedVector<VectorizedArray<value_type>> JxW_values(fe_eval.n_q_points);

    for(unsigned int cell = 0; cell < data.n_macro_cells() + data.n_macro_ghost_cells(); ++cell)
    {
      VectorizedArray<value_type> tau_convective = make_vectorized_array<value_type>(0.0);
      VectorizedArray<value_type> tau_viscous =
        make_vectorized_array<value_type>(operator_data.viscosity);

      if(operator_data.type_penalty_parameter == TypePenaltyParameter::ConvectiveTerm ||
         operator_data.type_penalty_parameter == TypePenaltyParameter::ViscousAndConvectiveTerms)
      {
        fe_eval.reinit(cell);
        fe_eval.read_dof_values(velocity);
        fe_eval.evaluate(true, false);
        VectorizedArray<value_type> volume      = make_vectorized_array<value_type>(0.0);
        VectorizedArray<value_type> norm_U_mean = make_vectorized_array<value_type>(0.0);
        JxW_values.resize(fe_eval.n_q_points);
        fe_eval.fill_JxW_values(JxW_values);
        for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
        {
          volume += JxW_values[q];
          norm_U_mean += JxW_values[q] * fe_eval.get_value(q).norm();
        }
        norm_U_mean /= volume;

        tau_convective =
          norm_U_mean * std::exp(std::log(volume) / (double)dim) / (double)(fe_degree + 1);
      }

      if(operator_data.type_penalty_parameter == TypePenaltyParameter::ConvectiveTerm)
      {
        array_penalty_parameter[cell] = operator_data.penalty_parameter * tau_convective;
      }
      else if(operator_data.type_penalty_parameter == TypePenaltyParameter::ViscousTerm)
      {
        array_penalty_parameter[cell] = operator_data.penalty_parameter * tau_viscous;
      }
      else if(operator_data.type_penalty_parameter ==
              TypePenaltyParameter::ViscousAndConvectiveTerms)
      {
        array_penalty_parameter[cell] =
          operator_data.penalty_parameter * (tau_convective + tau_viscous);
      }
    }
  }

  MatrixFree<dim, value_type> const &
  get_data() const
  {
    return data;
  }

  AlignedVector<VectorizedArray<value_type>> const &
  get_array_penalty_parameter() const
  {
    return array_penalty_parameter;
  }

  unsigned int
  get_dof_index() const
  {
    return dof_index;
  }

  unsigned int
  get_quad_index() const
  {
    return quad_index;
  }

  FEParameters<dim> const *
  get_fe_param() const
  {
    return this->fe_param;
  }

  void
  vmult(VectorType & dst, VectorType const & src) const
  {
    apply(dst, src);
  }

  void
  apply(VectorType & dst, VectorType const & src) const
  {
    this->get_data().cell_loop(&This::cell_loop, this, dst, src, true /*zero_dst_vector = true*/);
  }

  void
  apply_add(VectorType & dst, VectorType const & src) const
  {
    this->get_data().cell_loop(&This::cell_loop, this, dst, src, false /*zero_dst_vector = false*/);
  }

  void
  calculate_diagonal(VectorType & diagonal) const
  {
    VectorType src;
    this->get_data().cell_loop(
      &This::cell_loop_diagonal, this, diagonal, src, true /*zero_dst_vector = true*/);
  }

  void
  add_diagonal(VectorType & diagonal) const
  {
    VectorType src;
    this->get_data().cell_loop(
      &This::cell_loop_diagonal, this, diagonal, src, false /*zero_dst_vector = false*/);
  }

  void
  add_block_diagonal_matrices(std::vector<LAPACKFullMatrix<value_type>> & matrices) const
  {
    VectorType src;
    this->get_data().cell_loop(&This::cell_loop_calculate_block_jacobi_matrices,
                               this,
                               matrices,
                               src,
                               false /*zero_dst_vector = false*/);
  }

private:
  template<typename FEEvaluation>
  inline void
  do_cell_integral(FEEvaluation & fe_eval, unsigned int const cell) const
  {
    fe_eval.evaluate(false, true, false);

    VectorizedArray<value_type> tau = this->get_array_penalty_parameter()[cell];

    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      VectorizedArray<value_type> divergence = fe_eval.get_divergence(q);

      Tensor<2, dim, VectorizedArray<value_type>> unit_times_divU;
      for(unsigned int d = 0; d < dim; ++d)
      {
        unit_times_divU[d][d] = divergence;
      }
      fe_eval.submit_gradient(tau * unit_times_divU, q);
    }

    fe_eval.integrate(false, true);
  }

  void
  cell_loop(MatrixFree<dim, value_type> const &           data,
            VectorType &                                  dst,
            VectorType const &                            src,
            std::pair<unsigned int, unsigned int> const & cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval(data, this->get_fe_param(), this->get_dof_index());

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);

      do_cell_integral(fe_eval, cell);

      fe_eval.distribute_local_to_global(dst);
    }
  }

  void
  cell_loop_diagonal(MatrixFree<dim, value_type> const & data,
                     VectorType &                        dst,
                     VectorType const &,
                     std::pair<unsigned int, unsigned int> const & cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval(data, this->get_fe_param(), this->get_dof_index());

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);

      unsigned int dofs_per_cell = fe_eval.dofs_per_cell;

      VectorizedArray<value_type> local_diagonal_vector[fe_eval.tensor_dofs_per_cell];
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval.write_cellwise_dof_value(i, make_vectorized_array<value_type>(0.));
        fe_eval.write_cellwise_dof_value(j, make_vectorized_array<value_type>(1.));

        do_cell_integral(fe_eval, cell);

        local_diagonal_vector[j] = fe_eval.read_cellwise_dof_value(j);
      }
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        fe_eval.write_cellwise_dof_value(j, local_diagonal_vector[j]);

      fe_eval.distribute_local_to_global(dst);
    }
  }

  void
  cell_loop_calculate_block_jacobi_matrices(
    MatrixFree<dim, value_type> const &         data,
    std::vector<LAPACKFullMatrix<value_type>> & matrices,
    VectorType const &,
    std::pair<unsigned int, unsigned int> const & cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval(data, this->get_fe_param(), this->get_dof_index());

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);

      unsigned int dofs_per_cell = fe_eval.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        do_cell_integral(fe_eval, cell);

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          for(unsigned int v = 0; v < VectorizedArray<value_type>::n_array_elements; ++v)
            matrices[cell * VectorizedArray<value_type>::n_array_elements + v](i, j) +=
              fe_eval.begin_dof_values()[i][v];
      }
    }
  }

  MatrixFree<dim, value_type> const &        data;
  unsigned int const                         dof_index;
  unsigned int const                         quad_index;
  AlignedVector<VectorizedArray<value_type>> array_penalty_parameter;
  DivergencePenaltyOperatorData              operator_data;
};


/*
 *  Operator data.
 */
template<int dim>
struct ContinuityPenaltyOperatorData
{
  ContinuityPenaltyOperatorData()
    : type_penalty_parameter(TypePenaltyParameter::ConvectiveTerm),
      viscosity(0.0),
      penalty_parameter(1.0),
      which_components(ContinuityPenaltyComponents::Normal),
      use_boundary_data(false)
  {
  }

  // type of penalty parameter (viscous and/or convective terms)
  TypePenaltyParameter type_penalty_parameter;

  // kinematic viscosity
  double viscosity;

  // scaling factor
  double penalty_parameter;

  // the continuity penalty term can be applied to all velocity components and to the normal
  // component only
  ContinuityPenaltyComponents which_components;

  // the continuity penalty term can be applied on boundary faces or on interior faces only.
  bool use_boundary_data;

  std::shared_ptr<BoundaryDescriptorU<dim>> bc;
};


/*
 *  Continuity penalty operator: ( v_h , tau_conti * jump(u_h) )_dOmega^e where
 *   v_h : test function
 *   u_h : solution
 *   jump(u_h) = u_h^{-} - u_h^{+}
 *     where "-" denotes interior information and "+" exterior information
 *
 *   tau_conti: continuity penalty factor
 *
 *            use convective term:  tau_conti_conv = K * ||U||_mean
 *
 *            use viscous term:     tau_conti_viscous = K * nu / h
 *
 *                                  where h_eff = h / (k_u+1) and
 *                                  h = V_e^{1/3} with the element volume V_e
 *
 *            use both terms:       tau_conti = tau_conti_conv + tau_conti_viscous
 */
template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename value_type>
class ContinuityPenaltyOperator : public BaseOperator<dim>
{
public:
  enum class BoundaryType
  {
    undefined,
    dirichlet,
    neumann
  };

  static const bool         is_xwall = (xwall_quad_rule > 1) ? true : false;
  static const unsigned int n_actual_q_points_vel_linear =
    (is_xwall) ? xwall_quad_rule : fe_degree + 1;

  typedef LinearAlgebra::distributed::Vector<value_type> VectorType;

  typedef FEEvaluationWrapper<dim,
                              fe_degree,
                              fe_degree_xwall,
                              n_actual_q_points_vel_linear,
                              dim,
                              value_type,
                              is_xwall>
    FEEval_Velocity_Velocity_linear;

  typedef FEFaceEvaluationWrapper<dim,
                                  fe_degree,
                                  fe_degree_xwall,
                                  n_actual_q_points_vel_linear,
                                  dim,
                                  value_type,
                                  is_xwall>
    FEFaceEval_Velocity_Velocity_linear;

  typedef ContinuityPenaltyOperator<dim,
                                    fe_degree,
                                    fe_degree_p,
                                    fe_degree_xwall,
                                    xwall_quad_rule,
                                    value_type>
    This;

  ContinuityPenaltyOperator(MatrixFree<dim, value_type> const &      data_in,
                            unsigned int const                       dof_index_in,
                            unsigned int const                       quad_index_in,
                            ContinuityPenaltyOperatorData<dim> const operator_data_in)
    : data(data_in),
      dof_index(dof_index_in),
      quad_index(quad_index_in),
      array_penalty_parameter(0),
      operator_data(operator_data_in),
      eval_time(0.0)
  {
    array_penalty_parameter.resize(data.n_macro_cells() + data.n_macro_ghost_cells());
  }

  void
  calculate_array_penalty_parameter(VectorType const & velocity)
  {
    velocity.update_ghost_values();

    FEEval_Velocity_Velocity_linear fe_eval(data, this->fe_param, dof_index);

    AlignedVector<VectorizedArray<value_type>> JxW_values(fe_eval.n_q_points);

    for(unsigned int cell = 0; cell < data.n_macro_cells() + data.n_macro_ghost_cells(); ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(velocity);
      fe_eval.evaluate(true, false);
      VectorizedArray<value_type> volume      = make_vectorized_array<value_type>(0.0);
      VectorizedArray<value_type> norm_U_mean = make_vectorized_array<value_type>(0.0);
      JxW_values.resize(fe_eval.n_q_points);
      fe_eval.fill_JxW_values(JxW_values);
      for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      {
        volume += JxW_values[q];
        norm_U_mean += JxW_values[q] * fe_eval.get_value(q).norm();
      }
      norm_U_mean /= volume;

      VectorizedArray<value_type> tau_convective = norm_U_mean;
      VectorizedArray<value_type> h =
        std::exp(std::log(volume) / (double)dim) / (double)(fe_degree + 1);
      VectorizedArray<value_type> tau_viscous =
        make_vectorized_array<value_type>(operator_data.viscosity) / h;

      if(operator_data.type_penalty_parameter == TypePenaltyParameter::ConvectiveTerm)
      {
        array_penalty_parameter[cell] = operator_data.penalty_parameter * tau_convective;
      }
      else if(operator_data.type_penalty_parameter == TypePenaltyParameter::ViscousTerm)
      {
        array_penalty_parameter[cell] = operator_data.penalty_parameter * tau_viscous;
      }
      else if(operator_data.type_penalty_parameter ==
              TypePenaltyParameter::ViscousAndConvectiveTerms)
      {
        array_penalty_parameter[cell] =
          operator_data.penalty_parameter * (tau_convective + tau_viscous);
      }
    }
  }

  MatrixFree<dim, value_type> const &
  get_data() const
  {
    return data;
  }

  AlignedVector<VectorizedArray<value_type>> const &
  get_array_penalty_parameter() const
  {
    return array_penalty_parameter;
  }

  unsigned int
  get_dof_index() const
  {
    return dof_index;
  }

  unsigned int
  get_quad_index() const
  {
    return quad_index;
  }

  FEParameters<dim> const *
  get_fe_param() const
  {
    return this->fe_param;
  }

  void
  vmult(VectorType & dst, VectorType const & src) const
  {
    apply(dst, src);
  }

  void
  apply(VectorType & dst, VectorType const & src) const
  {
    this->get_data().loop(&This::cell_loop,
                          &This::face_loop,
                          &This::boundary_face_loop_hom_operator,
                          this,
                          dst,
                          src,
                          true /*zero dst vector = true*/,
                          MatrixFree<dim, value_type>::only_values,
                          MatrixFree<dim, value_type>::only_values);
  }

  void
  apply_add(VectorType & dst, VectorType const & src) const
  {
    this->get_data().loop(&This::cell_loop,
                          &This::face_loop,
                          &This::boundary_face_loop_hom_operator,
                          this,
                          dst,
                          src,
                          false /*zero dst vector = false*/,
                          MatrixFree<dim, value_type>::only_values,
                          MatrixFree<dim, value_type>::only_values);
  }

  void
  rhs(VectorType & dst, double const evaluation_time) const
  {
    this->eval_time = evaluation_time;

    VectorType src;
    this->get_data().loop(&This::cell_loop_inhom_operator,
                          &This::face_loop_inhom_operator,
                          &This::boundary_face_loop_inhom_operator,
                          this,
                          dst,
                          src,
                          true /*zero dst vector = true*/,
                          MatrixFree<dim, value_type>::only_values,
                          MatrixFree<dim, value_type>::only_values);
  }

  void
  rhs_add(VectorType & dst, double const evaluation_time) const
  {
    this->eval_time = evaluation_time;

    VectorType src;
    this->get_data().loop(&This::cell_loop_inhom_operator,
                          &This::face_loop_inhom_operator,
                          &This::boundary_face_loop_inhom_operator,
                          this,
                          dst,
                          src,
                          false /*zero dst vector = false*/,
                          MatrixFree<dim, value_type>::only_values,
                          MatrixFree<dim, value_type>::only_values);
  }

  void
  evaluate(VectorType & dst, VectorType const & src, double const evaluation_time) const
  {
    this->eval_time = evaluation_time;

    this->get_data().loop(&This::cell_loop,
                          &This::face_loop,
                          &This::boundary_face_loop_full_operator,
                          this,
                          dst,
                          src,
                          true /*zero dst vector = true*/,
                          MatrixFree<dim, value_type>::only_values,
                          MatrixFree<dim, value_type>::only_values);
  }

  void
  evaluate_add(VectorType & dst, VectorType const & src, double const evaluation_time) const
  {
    this->eval_time = evaluation_time;

    this->get_data().loop(&This::cell_loop,
                          &This::face_loop,
                          &This::boundary_face_loop_full_operator,
                          this,
                          dst,
                          src,
                          false /*zero dst vector = false*/,
                          MatrixFree<dim, value_type>::only_values,
                          MatrixFree<dim, value_type>::only_values);
  }

  void
  calculate_diagonal(VectorType & diagonal) const
  {
    VectorType src_dummy(diagonal);
    this->get_data().loop(&This::cell_loop_diagonal,
                          &This::face_loop_diagonal,
                          &This::boundary_face_loop_diagonal,
                          this,
                          diagonal,
                          src_dummy,
                          true /*zero dst vector = true*/,
                          MatrixFree<dim, value_type>::only_values,
                          MatrixFree<dim, value_type>::only_values);
  }

  void
  add_diagonal(VectorType & diagonal) const
  {
    VectorType src_dummy(diagonal);
    this->get_data().loop(&This::cell_loop_diagonal,
                          &This::face_loop_diagonal,
                          &This::boundary_face_loop_diagonal,
                          this,
                          diagonal,
                          src_dummy,
                          false /*zero dst vector = false*/,
                          MatrixFree<dim, value_type>::only_values,
                          MatrixFree<dim, value_type>::only_values);
  }

  void
  add_block_diagonal_matrices(std::vector<LAPACKFullMatrix<value_type>> & matrices) const
  {
    VectorType src;

    this->get_data().loop(&This::cell_loop_calculate_block_jacobi_matrices,
                          &This::face_loop_calculate_block_jacobi_matrices,
                          &This::boundary_face_loop_calculate_block_jacobi_matrices,
                          this,
                          matrices,
                          src);
  }

private:
  void
  cell_loop(MatrixFree<dim, value_type> const &,
            VectorType &,
            VectorType const &,
            std::pair<unsigned int, unsigned int> const &) const
  {
    // do nothing, i.e. no volume integrals
  }

  void
  face_loop(MatrixFree<dim, value_type> const &           data,
            VectorType &                                  dst,
            VectorType const &                            src,
            std::pair<unsigned int, unsigned int> const & face_range) const
  {
    FEFaceEval_Velocity_Velocity_linear fe_eval(data,
                                                this->get_fe_param(),
                                                true,
                                                this->get_dof_index());
    FEFaceEval_Velocity_Velocity_linear fe_eval_neighbor(data,
                                                         this->get_fe_param(),
                                                         false,
                                                         this->get_dof_index());

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval.reinit(face);
      fe_eval_neighbor.reinit(face);

      // TODO (Martin): will be included in matrix-free implementation for vectorial quantities
      // (probably end of 2017)
      //      fe_eval.gather_evaluate(src, true, false);
      //      fe_eval_neighbor.gather_evaluate(src, true, false);

      fe_eval.read_dof_values(src);
      fe_eval_neighbor.read_dof_values(src);

      fe_eval.evaluate(true, false);
      fe_eval_neighbor.evaluate(true, false);

      VectorizedArray<value_type> tau =
        0.5 * (fe_eval.read_cell_data(this->get_array_penalty_parameter()) +
               fe_eval_neighbor.read_cell_data(this->get_array_penalty_parameter()));

      for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      {
        Tensor<1, dim, VectorizedArray<value_type>> uM         = fe_eval.get_value(q);
        Tensor<1, dim, VectorizedArray<value_type>> uP         = fe_eval_neighbor.get_value(q);
        Tensor<1, dim, VectorizedArray<value_type>> jump_value = uM - uP;

        if(operator_data.which_components == ContinuityPenaltyComponents::All)
        {
          // penalize all velocity components
          fe_eval.submit_value(tau * jump_value, q);
          fe_eval_neighbor.submit_value(-tau * jump_value, q);
        }
        else if(operator_data.which_components == ContinuityPenaltyComponents::Normal)
        {
          // penalize normal components only
          Tensor<1, dim, VectorizedArray<value_type>> normal = fe_eval.get_normal_vector(q);

          fe_eval.submit_value(tau * (jump_value * normal) * normal, q);
          fe_eval_neighbor.submit_value(-tau * (jump_value * normal) * normal, q);
        }
        else
        {
          AssertThrow(operator_data.which_components == ContinuityPenaltyComponents::All ||
                        operator_data.which_components == ContinuityPenaltyComponents::Normal,
                      ExcMessage("not implemented."));
        }
      }
      fe_eval.integrate(true, false);
      fe_eval_neighbor.integrate(true, false);

      fe_eval.distribute_local_to_global(dst);
      fe_eval_neighbor.distribute_local_to_global(dst);

      //      fe_eval.integrate_scatter(true,false,dst);
      //      fe_eval_neighbor.integrate_scatter(true,false,dst);
    }
  }

  void
  boundary_face_loop_hom_operator(MatrixFree<dim, value_type> const &           data,
                                  VectorType &                                  dst,
                                  VectorType const &                            src,
                                  std::pair<unsigned int, unsigned int> const & face_range) const
  {
    if(operator_data.use_boundary_data == true)
    {
      FEFaceEval_Velocity_Velocity_linear fe_eval(data,
                                                  this->get_fe_param(),
                                                  true,
                                                  this->get_dof_index());

      for(unsigned int face = face_range.first; face < face_range.second; face++)
      {
        types::boundary_id boundary_id   = data.get_boundary_id(face);
        BoundaryType       boundary_type = BoundaryType::undefined;

        if(operator_data.bc->dirichlet_bc.find(boundary_id) != operator_data.bc->dirichlet_bc.end())
          boundary_type = BoundaryType::dirichlet;
        else if(operator_data.bc->neumann_bc.find(boundary_id) !=
                operator_data.bc->neumann_bc.end())
          boundary_type = BoundaryType::neumann;

        AssertThrow(boundary_type != BoundaryType::undefined,
                    ExcMessage("Boundary type of face is invalid or not implemented."));

        fe_eval.reinit(face);
        fe_eval.read_dof_values(src);
        fe_eval.evaluate(true, false);

        VectorizedArray<value_type> tau =
          fe_eval.read_cell_data(this->get_array_penalty_parameter());

        for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
        {
          Tensor<1, dim, VectorizedArray<value_type>> uM = fe_eval.get_value(q);
          Tensor<1, dim, VectorizedArray<value_type>> uP;

          if(boundary_type == BoundaryType::dirichlet)
          {
            uP = -uM;
          }
          else if(boundary_type == BoundaryType::neumann)
          {
            uP = uM;
          }
          else
          {
            AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
          }

          Tensor<1, dim, VectorizedArray<value_type>> jump_value = uM - uP;

          if(operator_data.which_components == ContinuityPenaltyComponents::All)
          {
            // penalize all velocity components
            fe_eval.submit_value(tau * jump_value, q);
          }
          else if(operator_data.which_components == ContinuityPenaltyComponents::Normal)
          {
            // penalize normal components only
            Tensor<1, dim, VectorizedArray<value_type>> normal = fe_eval.get_normal_vector(q);
            fe_eval.submit_value(tau * (jump_value * normal) * normal, q);
          }
          else
          {
            AssertThrow(operator_data.which_components == ContinuityPenaltyComponents::All ||
                          operator_data.which_components == ContinuityPenaltyComponents::Normal,
                        ExcMessage("not implemented."));
          }
        }
        fe_eval.integrate(true, false);

        fe_eval.distribute_local_to_global(dst);
      }
    }
  }

  void
  boundary_face_loop_full_operator(MatrixFree<dim, value_type> const &           data,
                                   VectorType &                                  dst,
                                   VectorType const &                            src,
                                   std::pair<unsigned int, unsigned int> const & face_range) const
  {
    if(operator_data.use_boundary_data == true)
    {
      FEFaceEval_Velocity_Velocity_linear fe_eval(data,
                                                  this->get_fe_param(),
                                                  true,
                                                  this->get_dof_index());

      for(unsigned int face = face_range.first; face < face_range.second; face++)
      {
        types::boundary_id boundary_id   = data.get_boundary_id(face);
        BoundaryType       boundary_type = BoundaryType::undefined;

        if(operator_data.bc->dirichlet_bc.find(boundary_id) != operator_data.bc->dirichlet_bc.end())
          boundary_type = BoundaryType::dirichlet;
        else if(operator_data.bc->neumann_bc.find(boundary_id) !=
                operator_data.bc->neumann_bc.end())
          boundary_type = BoundaryType::neumann;

        AssertThrow(boundary_type != BoundaryType::undefined,
                    ExcMessage("Boundary type of face is invalid or not implemented."));

        fe_eval.reinit(face);
        fe_eval.read_dof_values(src);
        fe_eval.evaluate(true, false);

        VectorizedArray<value_type> tau =
          fe_eval.read_cell_data(this->get_array_penalty_parameter());

        for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
        {
          Tensor<1, dim, VectorizedArray<value_type>> uM = fe_eval.get_value(q);
          Tensor<1, dim, VectorizedArray<value_type>> uP;

          if(boundary_type == BoundaryType::dirichlet)
          {
            Tensor<1, dim, VectorizedArray<value_type>> g;

            typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::iterator it;
            it = operator_data.bc->dirichlet_bc.find(boundary_id);
            Point<dim, VectorizedArray<value_type>> q_points = fe_eval.quadrature_point(q);
            evaluate_vectorial_function(g, it->second, q_points, eval_time);

            uP = -uM + make_vectorized_array<value_type>(2.0) * g;
          }
          else if(boundary_type == BoundaryType::neumann)
          {
            uP = uM;
          }
          else
          {
            AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
          }

          Tensor<1, dim, VectorizedArray<value_type>> jump_value = uM - uP;

          if(operator_data.which_components == ContinuityPenaltyComponents::All)
          {
            // penalize all velocity components
            fe_eval.submit_value(tau * jump_value, q);
          }
          else if(operator_data.which_components == ContinuityPenaltyComponents::Normal)
          {
            // penalize normal components only
            Tensor<1, dim, VectorizedArray<value_type>> normal = fe_eval.get_normal_vector(q);
            fe_eval.submit_value(tau * (jump_value * normal) * normal, q);
          }
          else
          {
            AssertThrow(operator_data.which_components == ContinuityPenaltyComponents::All ||
                          operator_data.which_components == ContinuityPenaltyComponents::Normal,
                        ExcMessage("not implemented."));
          }
        }
        fe_eval.integrate(true, false);

        fe_eval.distribute_local_to_global(dst);
      }
    }
  }


  void
  cell_loop_inhom_operator(MatrixFree<dim, value_type> const &,
                           VectorType &,
                           VectorType const &,
                           std::pair<unsigned int, unsigned int> const &) const
  {
    // do nothing, i.e. no volume integrals
  }

  void
  face_loop_inhom_operator(MatrixFree<dim, value_type> const &,
                           VectorType &,
                           VectorType const &,
                           std::pair<unsigned int, unsigned int> const &) const
  {
    // do nothing, i.e. no volume integrals
  }

  void
  boundary_face_loop_inhom_operator(MatrixFree<dim, value_type> const & data,
                                    VectorType &                        dst,
                                    VectorType const &,
                                    std::pair<unsigned int, unsigned int> const & face_range) const
  {
    if(operator_data.use_boundary_data == true)
    {
      FEFaceEval_Velocity_Velocity_linear fe_eval(data,
                                                  this->get_fe_param(),
                                                  true,
                                                  this->get_dof_index());

      for(unsigned int face = face_range.first; face < face_range.second; face++)
      {
        types::boundary_id boundary_id   = data.get_boundary_id(face);
        BoundaryType       boundary_type = BoundaryType::undefined;

        if(operator_data.bc->dirichlet_bc.find(boundary_id) != operator_data.bc->dirichlet_bc.end())
          boundary_type = BoundaryType::dirichlet;
        else if(operator_data.bc->neumann_bc.find(boundary_id) !=
                operator_data.bc->neumann_bc.end())
          boundary_type = BoundaryType::neumann;

        AssertThrow(boundary_type != BoundaryType::undefined,
                    ExcMessage("Boundary type of face is invalid or not implemented."));

        fe_eval.reinit(face);

        VectorizedArray<value_type> tau =
          fe_eval.read_cell_data(this->get_array_penalty_parameter());

        for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
        {
          Tensor<1, dim, VectorizedArray<value_type>> jump_value;

          if(boundary_type == BoundaryType::dirichlet)
          {
            Tensor<1, dim, VectorizedArray<value_type>> g;

            typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::iterator it;
            it = operator_data.bc->dirichlet_bc.find(boundary_id);
            Point<dim, VectorizedArray<value_type>> q_points = fe_eval.quadrature_point(q);
            evaluate_vectorial_function(g, it->second, q_points, eval_time);

            // + sign since this term appears on the rhs of the equations
            jump_value = make_vectorized_array<value_type>(2.0) * g;
          }
          else if(boundary_type == BoundaryType::neumann)
          {
            // do nothing (jump_value = 0)
          }
          else
          {
            AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
          }

          if(operator_data.which_components == ContinuityPenaltyComponents::All)
          {
            // penalize all velocity components
            fe_eval.submit_value(tau * jump_value, q);
          }
          else if(operator_data.which_components == ContinuityPenaltyComponents::Normal)
          {
            // penalize normal components only
            Tensor<1, dim, VectorizedArray<value_type>> normal = fe_eval.get_normal_vector(q);
            fe_eval.submit_value(tau * (jump_value * normal) * normal, q);
          }
          else
          {
            AssertThrow(operator_data.which_components == ContinuityPenaltyComponents::All ||
                          operator_data.which_components == ContinuityPenaltyComponents::Normal,
                        ExcMessage("not implemented."));
          }
        }
        fe_eval.integrate(true, false);

        fe_eval.distribute_local_to_global(dst);
      }
    }
  }

  void
  cell_loop_diagonal(MatrixFree<dim, value_type> const &,
                     VectorType &,
                     VectorType const &,
                     std::pair<unsigned int, unsigned int> const &) const
  {
    // do nothing, i.e. no volume integrals
  }

  void
  face_loop_diagonal(MatrixFree<dim, value_type> const & data,
                     VectorType &                        dst,
                     VectorType const &,
                     std::pair<unsigned int, unsigned int> const & face_range) const
  {
    FEFaceEval_Velocity_Velocity_linear fe_eval(data,
                                                this->get_fe_param(),
                                                true,
                                                this->get_dof_index());
    FEFaceEval_Velocity_Velocity_linear fe_eval_neighbor(data,
                                                         this->get_fe_param(),
                                                         false,
                                                         this->get_dof_index());

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval.reinit(face);
      fe_eval_neighbor.reinit(face);

      // element-
      unsigned int                dofs_per_cell = fe_eval.dofs_per_cell;
      VectorizedArray<value_type> local_diagonal_vector[fe_eval.tensor_dofs_per_cell];
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval.write_cellwise_dof_value(i, make_vectorized_array<value_type>(0.));
        fe_eval.write_cellwise_dof_value(j, make_vectorized_array<value_type>(1.));

        fe_eval.evaluate(true, false);

        VectorizedArray<value_type> tau =
          0.5 * (fe_eval.read_cell_data(this->get_array_penalty_parameter()) +
                 fe_eval_neighbor.read_cell_data(this->get_array_penalty_parameter()));

        for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
        {
          Tensor<1, dim, VectorizedArray<value_type>> uM = fe_eval.get_value(q);
          // set uP to zero
          Tensor<1, dim, VectorizedArray<value_type>> uP;
          Tensor<1, dim, VectorizedArray<value_type>> jump_value = uM - uP;

          if(operator_data.which_components == ContinuityPenaltyComponents::All)
          {
            // penalize all velocity components
            fe_eval.submit_value(tau * jump_value, q);
          }
          else if(operator_data.which_components == ContinuityPenaltyComponents::Normal)
          {
            // penalize normal components only
            Tensor<1, dim, VectorizedArray<value_type>> normal = fe_eval.get_normal_vector(q);
            fe_eval.submit_value(tau * (jump_value * normal) * normal, q);
          }
          else
          {
            AssertThrow(operator_data.which_components == ContinuityPenaltyComponents::All ||
                          operator_data.which_components == ContinuityPenaltyComponents::Normal,
                        ExcMessage("not implemented."));
          }
        }

        fe_eval.integrate(true, false);

        local_diagonal_vector[j] = fe_eval.read_cellwise_dof_value(j);
      }
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        fe_eval.write_cellwise_dof_value(j, local_diagonal_vector[j]);

      fe_eval.distribute_local_to_global(dst);

      // neighbor (element+)
      unsigned int dofs_per_cell_neighbor = fe_eval_neighbor.dofs_per_cell;
      VectorizedArray<value_type>
        local_diagonal_vector_neighbor[fe_eval_neighbor.tensor_dofs_per_cell];
      for(unsigned int j = 0; j < dofs_per_cell_neighbor; ++j)
      {
        // set dof value j of element+ to 1 and all other dof values of element+ to zero
        for(unsigned int i = 0; i < dofs_per_cell_neighbor; ++i)
          fe_eval_neighbor.write_cellwise_dof_value(i, make_vectorized_array<value_type>(0.));
        fe_eval_neighbor.write_cellwise_dof_value(j, make_vectorized_array<value_type>(1.));

        fe_eval_neighbor.evaluate(true, false);

        VectorizedArray<value_type> tau =
          0.5 * (fe_eval.read_cell_data(this->get_array_penalty_parameter()) +
                 fe_eval_neighbor.read_cell_data(this->get_array_penalty_parameter()));

        for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
        {
          // set uM to zero
          Tensor<1, dim, VectorizedArray<value_type>> uM;
          Tensor<1, dim, VectorizedArray<value_type>> uP = fe_eval_neighbor.get_value(q);
          Tensor<1, dim, VectorizedArray<value_type>> jump_value =
            uP - uM; // interior - exterior = uP - uM (neighbor!)

          if(operator_data.which_components == ContinuityPenaltyComponents::All)
          {
            // penalize all velocity components
            fe_eval_neighbor.submit_value(tau * jump_value, q);
          }
          else if(operator_data.which_components == ContinuityPenaltyComponents::Normal)
          {
            // penalize normal components only
            Tensor<1, dim, VectorizedArray<value_type>> normal =
              fe_eval_neighbor.get_normal_vector(q);
            fe_eval_neighbor.submit_value(tau * (jump_value * normal) * normal, q);
          }
          else
          {
            AssertThrow(operator_data.which_components == ContinuityPenaltyComponents::All ||
                          operator_data.which_components == ContinuityPenaltyComponents::Normal,
                        ExcMessage("not implemented."));
          }
        }
        fe_eval_neighbor.integrate(true, false);

        local_diagonal_vector_neighbor[j] = fe_eval_neighbor.read_cellwise_dof_value(j);
      }
      for(unsigned int j = 0; j < dofs_per_cell_neighbor; ++j)
        fe_eval_neighbor.write_cellwise_dof_value(j, local_diagonal_vector_neighbor[j]);

      fe_eval_neighbor.distribute_local_to_global(dst);
    }
  }

  void
  boundary_face_loop_diagonal(MatrixFree<dim, value_type> const & data,
                              VectorType &                        dst,
                              VectorType const &,
                              std::pair<unsigned int, unsigned int> const & face_range) const
  {
    if(operator_data.use_boundary_data == true)
    {
      FEFaceEval_Velocity_Velocity_linear fe_eval(data,
                                                  this->get_fe_param(),
                                                  true,
                                                  this->get_dof_index());

      for(unsigned int face = face_range.first; face < face_range.second; face++)
      {
        types::boundary_id boundary_id   = data.get_boundary_id(face);
        BoundaryType       boundary_type = BoundaryType::undefined;

        if(operator_data.bc->dirichlet_bc.find(boundary_id) != operator_data.bc->dirichlet_bc.end())
        {
          boundary_type = BoundaryType::dirichlet;
        }
        else if(operator_data.bc->neumann_bc.find(boundary_id) !=
                operator_data.bc->neumann_bc.end())
        {
          boundary_type = BoundaryType::neumann;
        }

        AssertThrow(boundary_type != BoundaryType::undefined,
                    ExcMessage("Boundary type of face is invalid or not implemented."));

        fe_eval.reinit(face);

        unsigned int                dofs_per_cell = fe_eval.dofs_per_cell;
        VectorizedArray<value_type> local_diagonal_vector[fe_eval.tensor_dofs_per_cell];
        for(unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          // set dof value j of element- to 1 and all other dof values of element- to zero
          for(unsigned int i = 0; i < dofs_per_cell; ++i)
            fe_eval.write_cellwise_dof_value(i, make_vectorized_array<value_type>(0.));
          fe_eval.write_cellwise_dof_value(j, make_vectorized_array<value_type>(1.));

          fe_eval.evaluate(true, false);

          VectorizedArray<value_type> tau =
            fe_eval.read_cell_data(this->get_array_penalty_parameter());

          for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
          {
            Tensor<1, dim, VectorizedArray<value_type>> uM = fe_eval.get_value(q);
            Tensor<1, dim, VectorizedArray<value_type>> uP;

            if(boundary_type == BoundaryType::dirichlet)
            {
              uP = -uM;
            }
            else if(boundary_type == BoundaryType::neumann)
            {
              uP = uM;
            }
            else
            {
              AssertThrow(false,
                          ExcMessage("Boundary type of face is invalid or not implemented."));
            }

            Tensor<1, dim, VectorizedArray<value_type>> jump_value = uM - uP;

            if(operator_data.which_components == ContinuityPenaltyComponents::All)
            {
              // penalize all velocity components
              fe_eval.submit_value(tau * jump_value, q);
            }
            else if(operator_data.which_components == ContinuityPenaltyComponents::Normal)
            {
              // penalize normal components only
              Tensor<1, dim, VectorizedArray<value_type>> normal = fe_eval.get_normal_vector(q);
              fe_eval.submit_value(tau * (jump_value * normal) * normal, q);
            }
            else
            {
              AssertThrow(operator_data.which_components == ContinuityPenaltyComponents::All ||
                            operator_data.which_components == ContinuityPenaltyComponents::Normal,
                          ExcMessage("not implemented."));
            }
          }

          fe_eval.integrate(true, false);

          local_diagonal_vector[j] = fe_eval.read_cellwise_dof_value(j);
        }
        for(unsigned int j = 0; j < dofs_per_cell; ++j)
          fe_eval.write_cellwise_dof_value(j, local_diagonal_vector[j]);

        fe_eval.distribute_local_to_global(dst);
      }
    }
  }

  void
  cell_loop_calculate_block_jacobi_matrices(MatrixFree<dim, value_type> const &,
                                            std::vector<LAPACKFullMatrix<value_type>> &,
                                            VectorType const &,
                                            std::pair<unsigned int, unsigned int> const &) const
  {
    // do nothing
  }

  void
  face_loop_calculate_block_jacobi_matrices(
    MatrixFree<dim, value_type> const &         data,
    std::vector<LAPACKFullMatrix<value_type>> & matrices,
    VectorType const &,
    std::pair<unsigned int, unsigned int> const & face_range) const
  {
    // TODO
    FEFaceEval_Velocity_Velocity_linear fe_eval(data,
                                                this->get_fe_param(),
                                                true,
                                                this->get_dof_index());
    FEFaceEval_Velocity_Velocity_linear fe_eval_neighbor(data,
                                                         this->get_fe_param(),
                                                         false,
                                                         this->get_dof_index());

    // Perform face intergrals for element e⁻.
    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval.reinit(face);
      fe_eval_neighbor.reinit(face);

      VectorizedArray<value_type> tau =
        0.5 * (fe_eval.read_cell_data(this->get_array_penalty_parameter()) +
               fe_eval_neighbor.read_cell_data(this->get_array_penalty_parameter()));

      unsigned int dofs_per_cell = fe_eval.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        fe_eval.evaluate(true, false);

        for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
        {
          Tensor<1, dim, VectorizedArray<value_type>> uM = fe_eval.get_value(q);
          // set uP to zero
          Tensor<1, dim, VectorizedArray<value_type>> uP;
          Tensor<1, dim, VectorizedArray<value_type>> jump_value = uM - uP;

          if(operator_data.which_components == ContinuityPenaltyComponents::All)
          {
            // penalize all velocity components
            fe_eval.submit_value(tau * jump_value, q);
          }
          else if(operator_data.which_components == ContinuityPenaltyComponents::Normal)
          {
            // penalize normal components only
            Tensor<1, dim, VectorizedArray<value_type>> normal = fe_eval.get_normal_vector(q);
            fe_eval.submit_value(tau * (jump_value * normal) * normal, q);
          }
          else
          {
            AssertThrow(operator_data.which_components == ContinuityPenaltyComponents::All ||
                          operator_data.which_components == ContinuityPenaltyComponents::Normal,
                        ExcMessage("not implemented."));
          }
        }

        fe_eval.integrate(true, false);

        for(unsigned int v = 0; v < VectorizedArray<value_type>::n_array_elements; ++v)
        {
          const unsigned int cell_number = data.get_face_info(face).cells_interior[v];
          if(cell_number != numbers::invalid_unsigned_int)
            for(unsigned int i = 0; i < dofs_per_cell; ++i)
              matrices[cell_number](i, j) += fe_eval.begin_dof_values()[i][v];
        }
      }
    }



    // TODO: This has to be removed as soon as the new infrastructure is used that
    // allows to perform face integrals over all faces of the current element.
    // Perform face intergrals for element e⁺.
    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval.reinit(face);
      fe_eval_neighbor.reinit(face);

      VectorizedArray<value_type> tau =
        0.5 * (fe_eval.read_cell_data(this->get_array_penalty_parameter()) +
               fe_eval_neighbor.read_cell_data(this->get_array_penalty_parameter()));

      // Note that the velocity has dim components.
      unsigned int dofs_per_cell = fe_eval_neighbor.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        // set dof value j of element+ to 1 and all other dof values of element+ to zero
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval_neighbor.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval_neighbor.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        fe_eval_neighbor.evaluate(true, false);

        for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
        {
          // set uM to zero
          Tensor<1, dim, VectorizedArray<value_type>> uM;
          Tensor<1, dim, VectorizedArray<value_type>> uP = fe_eval_neighbor.get_value(q);
          Tensor<1, dim, VectorizedArray<value_type>> jump_value =
            uP - uM; // interior - exterior = uP - uM (neighbor!)

          if(operator_data.which_components == ContinuityPenaltyComponents::All)
          {
            // penalize all velocity components
            fe_eval_neighbor.submit_value(tau * jump_value, q);
          }
          else if(operator_data.which_components == ContinuityPenaltyComponents::Normal)
          {
            // penalize normal components only
            Tensor<1, dim, VectorizedArray<value_type>> normal =
              fe_eval_neighbor.get_normal_vector(q);
            fe_eval_neighbor.submit_value(tau * (jump_value * normal) * normal, q);
          }
          else
          {
            AssertThrow(operator_data.which_components == ContinuityPenaltyComponents::All ||
                          operator_data.which_components == ContinuityPenaltyComponents::Normal,
                        ExcMessage("not implemented."));
          }
        }
        fe_eval_neighbor.integrate(true, false);

        for(unsigned int v = 0; v < VectorizedArray<value_type>::n_array_elements; ++v)
        {
          const unsigned int cell_number = data.get_face_info(face).cells_exterior[v];
          if(cell_number != numbers::invalid_unsigned_int)
            for(unsigned int i = 0; i < dofs_per_cell; ++i)
              matrices[cell_number](i, j) += fe_eval_neighbor.begin_dof_values()[i][v];
        }
      }
    }
  }

  // TODO: This function has to be removed as soon as the new infrastructure is used that
  // allows to perform face integrals over all faces of the current element.
  void
  boundary_face_loop_calculate_block_jacobi_matrices(
    MatrixFree<dim, value_type> const &         data,
    std::vector<LAPACKFullMatrix<value_type>> & matrices,
    VectorType const &,
    std::pair<unsigned int, unsigned int> const & face_range) const
  {
    if(operator_data.use_boundary_data == true)
    {
      FEFaceEval_Velocity_Velocity_linear fe_eval(data,
                                                  this->get_fe_param(),
                                                  true,
                                                  this->get_dof_index());

      for(unsigned int face = face_range.first; face < face_range.second; face++)
      {
        types::boundary_id boundary_id   = data.get_boundary_id(face);
        BoundaryType       boundary_type = BoundaryType::undefined;

        if(operator_data.bc->dirichlet_bc.find(boundary_id) != operator_data.bc->dirichlet_bc.end())
        {
          boundary_type = BoundaryType::dirichlet;
        }
        else if(operator_data.bc->neumann_bc.find(boundary_id) !=
                operator_data.bc->neumann_bc.end())
        {
          boundary_type = BoundaryType::neumann;
        }

        AssertThrow(boundary_type != BoundaryType::undefined,
                    ExcMessage("Boundary type of face is invalid or not implemented."));

        fe_eval.reinit(face);

        unsigned int dofs_per_cell = fe_eval.dofs_per_cell;

        VectorizedArray<value_type> tau =
          fe_eval.read_cell_data(this->get_array_penalty_parameter());

        for(unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          // set dof value j of element- to 1 and all other dof values of element- to zero
          for(unsigned int i = 0; i < dofs_per_cell; ++i)
            fe_eval.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
          fe_eval.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

          fe_eval.evaluate(true, false);

          for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
          {
            Tensor<1, dim, VectorizedArray<value_type>> uM = fe_eval.get_value(q);
            Tensor<1, dim, VectorizedArray<value_type>> uP;

            if(boundary_type == BoundaryType::dirichlet)
            {
              uP = -uM;
            }
            else if(boundary_type == BoundaryType::neumann)
            {
              uP = uM;
            }
            else
            {
              AssertThrow(false,
                          ExcMessage("Boundary type of face is invalid or not implemented."));
            }

            Tensor<1, dim, VectorizedArray<value_type>> jump_value = uM - uP;

            if(operator_data.which_components == ContinuityPenaltyComponents::All)
            {
              // penalize all velocity components
              fe_eval.submit_value(tau * jump_value, q);
            }
            else if(operator_data.which_components == ContinuityPenaltyComponents::Normal)
            {
              // penalize normal components only
              Tensor<1, dim, VectorizedArray<value_type>> normal = fe_eval.get_normal_vector(q);
              fe_eval.submit_value(tau * (jump_value * normal) * normal, q);
            }
            else
            {
              AssertThrow(operator_data.which_components == ContinuityPenaltyComponents::All ||
                            operator_data.which_components == ContinuityPenaltyComponents::Normal,
                          ExcMessage("not implemented."));
            }
          }

          fe_eval.integrate(true, false);

          for(unsigned int v = 0; v < VectorizedArray<value_type>::n_array_elements; ++v)
          {
            const unsigned int cell_number = data.get_face_info(face).cells_interior[v];
            if(cell_number != numbers::invalid_unsigned_int)
            {
              for(unsigned int i = 0; i < dofs_per_cell; ++i)
                matrices[cell_number](i, j) += fe_eval.begin_dof_values()[i][v];
            }
          }
        }
      }
    }
  }

  MatrixFree<dim, value_type> const & data;

  unsigned int const dof_index;
  unsigned int const quad_index;

  AlignedVector<VectorizedArray<value_type>> array_penalty_parameter;

  ContinuityPenaltyOperatorData<dim> operator_data;

  double mutable eval_time;
};


} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DIVERGENCE_AND_CONTINUITY_PENALTY_OPERATORS_H_ \
        */
