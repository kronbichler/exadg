/*
 * multigrid_preconditioner.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_COMPATIBLE_LAPLACE_MULTIGRID_PRECONDITIONER_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_COMPATIBLE_LAPLACE_MULTIGRID_PRECONDITIONER_H_

#include <vector>

#include "../../operators/multigrid_operator.h"
#include "../../solvers_and_preconditioners/multigrid/multigrid_preconditioner_base.h"
#include "compatible_laplace_operator.h"

namespace IncNS
{
/*
 *  Multigrid preconditioner for compatible Laplace operator.
 */
template<int dim, typename Number, typename MultigridNumber>
class CompatibleLaplaceMultigridPreconditioner
  : public MultigridPreconditionerBase<dim, Number, MultigridNumber>
{
public:
  typedef CompatibleLaplaceOperator<dim, MultigridNumber> PDEOperator;

  typedef MultigridOperatorBase<dim, MultigridNumber>          MGOperatorBase;
  typedef MultigridOperator<dim, MultigridNumber, PDEOperator> MGOperator;

  typedef MultigridPreconditionerBase<dim, Number, MultigridNumber> Base;

  typedef typename Base::Map               Map;
  typedef typename Base::PeriodicFacePairs PeriodicFacePairs;
  typedef typename Base::VectorType        VectorType;
  typedef typename Base::VectorTypeMG      VectorTypeMG;

  typedef typename MatrixFree<dim, MultigridNumber>::AdditionalData MatrixFreeData;

  CompatibleLaplaceMultigridPreconditioner(MPI_Comm const & mpi_comm)
    : Base(mpi_comm), mesh_is_moving(false)
  {
  }

  void
  initialize(MultigridData const &                      mg_data,
             const parallel::TriangulationBase<dim> *   tria,
             const FiniteElement<dim> &                 fe,
             Mapping<dim> const &                       mapping,
             CompatibleLaplaceOperatorData<dim> const & data_in,
             bool const                                 mesh_is_moving,
             Map const *                                dirichlet_bc        = nullptr,
             PeriodicFacePairs *                        periodic_face_pairs = nullptr)
  {
    data                    = data_in;
    data.dof_index_velocity = 1;
    data.dof_index_pressure = 0;

    data.gradient_operator_data.dof_index_velocity = data.dof_index_velocity;
    data.gradient_operator_data.dof_index_pressure = data.dof_index_pressure;
    data.gradient_operator_data.quad_index         = 0;

    data.divergence_operator_data.dof_index_velocity = data.dof_index_velocity;
    data.divergence_operator_data.dof_index_pressure = data.dof_index_pressure;
    data.divergence_operator_data.quad_index         = 0;

    this->mesh_is_moving = mesh_is_moving;

    Base::initialize(
      mg_data, tria, fe, mapping, data.operator_is_singular, dirichlet_bc, periodic_face_pairs);
  }

  void
  update() override
  {
    // update of this multigrid preconditioner is only needed
    // if the mesh is moving
    if(mesh_is_moving)
    {
      this->update_matrix_free();

      update_operators_after_mesh_movement();

      this->update_smoothers();

      // singular operators do not occur for this operator
      this->update_coarse_solver(data.operator_is_singular);
    }
  }

private:
  void
  initialize_matrix_free() override
  {
    if(mesh_is_moving)
    {
      matrix_free_data_update.resize(0, this->n_levels - 1);
    }

    dof_handler_vec.resize(0, this->n_levels - 1);
    constraint_matrix_vec.resize(0, this->n_levels - 1);
    quadrature_vec.resize(0, this->n_levels - 1);

    Base::initialize_matrix_free();
  }

  std::shared_ptr<MatrixFree<dim, MultigridNumber>>
  do_initialize_matrix_free(unsigned int const level) override
  {
    std::shared_ptr<MatrixFree<dim, MultigridNumber>> matrix_free;
    matrix_free.reset(new MatrixFree<dim, MultigridNumber>);

    auto & dof_handler_p = *this->dof_handlers[level];
    auto & dof_handler_u = *this->dof_handlers_velocity[level];

    // dof_handler
    // TODO: instead of 2 use something more general like DofHandlerSelector::n_variants
    dof_handler_vec[level].resize(2);
    dof_handler_vec[level][data.dof_index_velocity] = &dof_handler_u;
    dof_handler_vec[level][data.dof_index_pressure] = &dof_handler_p;

    // constraint matrix
    // TODO: instead of 2 use something more general like DofHandlerSelector::n_variants
    constraint_matrix_vec[level].resize(2);
    constraint_matrix_vec[level][data.dof_index_velocity] = &*this->constraints_velocity[level];
    constraint_matrix_vec[level][data.dof_index_pressure] = &*this->constraints[level];

    // quadratures
    quadrature_vec[level].resize(2);
    // quadrature formula with (fe_degree_velocity+1) quadrature points: this is the quadrature
    // formula that is used for the gradient operator and the divergence operator (and the inverse
    // velocity mass matrix operator)
    quadrature_vec[level][0] =
      QGauss<1>(this->level_info[level].degree() + 1 + (data.degree_u - data.degree_p));
    // quadrature formula with (fe_degree_p+1) quadrature points: this is the quadrature
    // that is needed for p-transfer
    quadrature_vec[level][1] = QGauss<1>(this->level_info[level].degree() + 1);

    MatrixFreeData addit_data;
    addit_data.mapping_update_flags =
      (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
       update_values);

    if(this->level_info[level].is_dg())
    {
      addit_data.mapping_update_flags_inner_faces =
        (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
         update_values);

      addit_data.mapping_update_flags_boundary_faces =
        (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
         update_values);
    }

    addit_data.mg_level = this->level_info[level].h_level();

    // if(data.use_cell_based_loops)
    //{
    //  auto tria = dynamic_cast<parallel::distributed::Triangulation<dim> const *>(
    //    &dof_handler_p.get_triangulation());
    //  Categorization::do_cell_based_loops(*tria, additional_data,
    //  this->level_info[level].level);
    //}

    if(mesh_is_moving)
    {
      matrix_free_data_update[level] = addit_data;
      matrix_free_data_update[level].initialize_indices =
        false; // connectivity of elements stays the same
      matrix_free_data_update[level].initialize_mapping = true;
    }

    matrix_free->reinit(*this->mapping,
                        dof_handler_vec[level],
                        constraint_matrix_vec[level],
                        quadrature_vec[level],
                        addit_data);

    return matrix_free;
  }

  std::shared_ptr<MGOperatorBase>
  initialize_operator(unsigned int const level)
  {
    // initialize pde_operator in a first step
    std::shared_ptr<PDEOperator> pde_operator(new PDEOperator());
    pde_operator->reinit_multigrid(*this->matrix_free_objects[level],
                                   *this->constraints[level],
                                   data);

    // initialize MGOperator which is a wrapper around the PDEOperator
    std::shared_ptr<MGOperator> mg_operator(new MGOperator(pde_operator));

    return mg_operator;
  }

  void
  initialize_dof_handler_and_constraints(bool const                 operator_is_singular,
                                         PeriodicFacePairs *        periodic_face_pairs,
                                         FiniteElement<dim> const & fe,
                                         parallel::TriangulationBase<dim> const * tria,
                                         Map const *                              dirichlet_bc)
  {
    Base::initialize_dof_handler_and_constraints(
      operator_is_singular, periodic_face_pairs, fe, tria, dirichlet_bc);

    // do setup required for derived class

    std::vector<MGLevelInfo>            level_info_velocity;
    std::vector<MGDoFHandlerIdentifier> p_levels_velocity;

    // setup global velocity levels
    for(auto & level : this->level_info)
      level_info_velocity.push_back(
        {level.h_level(), level.degree() + data.degree_u - data.degree_p, level.is_dg()});

    // setup p velocity levels
    for(auto level : level_info_velocity)
      p_levels_velocity.push_back(level.dof_handler_id());

    sort(p_levels_velocity.begin(), p_levels_velocity.end());
    p_levels_velocity.erase(unique(p_levels_velocity.begin(), p_levels_velocity.end()),
                            p_levels_velocity.end());
    std::reverse(std::begin(p_levels_velocity), std::end(p_levels_velocity));

    // setup dofhandler and constraint matrices
    FE_DGQ<dim>   temp(data.degree_u);
    FESystem<dim> fe_velocity(temp, dim);

    Map dirichlet_bc_velocity;
    this->do_initialize_dof_handler_and_constraints(false,
                                                    *periodic_face_pairs,
                                                    fe_velocity,
                                                    tria,
                                                    dirichlet_bc_velocity,
                                                    level_info_velocity,
                                                    p_levels_velocity,
                                                    this->dof_handlers_velocity,
                                                    this->constrained_dofs_velocity,
                                                    this->constraints_velocity);
  }

  void
  do_update_matrix_free(unsigned int const level) override
  {
    this->matrix_free_objects[level]->reinit(*this->mapping,
                                             dof_handler_vec[level],
                                             constraint_matrix_vec[level],
                                             quadrature_vec[level],
                                             matrix_free_data_update[level]);
  }

  /*
   * This function performs the updates that are necessary after the mesh has been moved
   * and after matrix_free has been updated.
   */
  void
  update_operators_after_mesh_movement()
  {
    for(unsigned int level = this->coarse_level; level <= this->fine_level; ++level)
    {
      get_operator(level)->update_after_mesh_movement();
    }
  }

  std::shared_ptr<PDEOperator>
  get_operator(unsigned int level)
  {
    std::shared_ptr<MGOperator> mg_operator =
      std::dynamic_pointer_cast<MGOperator>(this->operators[level]);

    return mg_operator->get_pde_operator();
  }

  MGLevelObject<std::shared_ptr<const DoFHandler<dim>>>     dof_handlers_velocity;
  MGLevelObject<std::shared_ptr<MGConstrainedDoFs>>         constrained_dofs_velocity;
  MGLevelObject<std::shared_ptr<AffineConstraints<double>>> constraints_velocity;

  CompatibleLaplaceOperatorData<dim> data;

  MGLevelObject<MatrixFreeData> matrix_free_data_update;

  MGLevelObject<std::vector<const DoFHandler<dim> *>>           dof_handler_vec;
  MGLevelObject<std::vector<AffineConstraints<double> const *>> constraint_matrix_vec;
  MGLevelObject<std::vector<Quadrature<1>>>                     quadrature_vec;

  bool mesh_is_moving;
};

} // namespace IncNS


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_COMPATIBLE_LAPLACE_MULTIGRID_PRECONDITIONER_H_ \
        */
