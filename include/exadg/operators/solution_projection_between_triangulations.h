/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2025 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef EXADG_OPERATORS_SOLUTION_PROJECTION_BETWEEN_TRIANGULATIONS_H_
#define EXADG_OPERATORS_SOLUTION_PROJECTION_BETWEEN_TRIANGULATIONS_H_

// C/C++
#include <algorithm>

// deal.II
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/base/timer.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/numerics/vector_tools.h>

// ExaDG
#include <exadg/grid/grid_data.h>
#include <exadg/matrix_free/integrators.h>
#include <exadg/matrix_free/matrix_free_data.h>
#include <exadg/operators/inverse_mass_operator.h>
#include <exadg/operators/inverse_mass_parameters.h>
#include <exadg/operators/mapping_flags.h>
#include <exadg/operators/quadrature.h>
#include <exadg/postprocessor/write_output.h>

namespace ExaDG
{
namespace GridToGridProjection
{
// Parameters controlling the grid-to-grid projection.
template<int dim>
struct GridToGridProjectionData
{
  GridToGridProjectionData()
    : rpe_data(),
      solver_data(SolverData(1e3, 1e-20, 1e-6, LinearSolver::CG)),
      preconditioner(PreconditionerMass::PointJacobi),
      amg_data(AMGData()),
      additional_quadrature_points(1),
      grids_and_maps_identical(false)
  {
  }

  void
  print(dealii::ConditionalOStream const & pcout) const
  {
    print_parameter(pcout, "RPE tolerance", rpe_data.tolerance);
    print_parameter(pcout, "RPE enforce unique mapping", rpe_data.enforce_unique_mapping);
    print_parameter(pcout, "RPE rtree level", rpe_data.rtree_level);

    // These parameters play only a role if an iterative scheme is used for projection.
    // That is for `InverseMassType != InverseMassType::MatrixfreeOperator` determined at runtime.
    solver_data.print(pcout);
    print_parameter(pcout, "Preconditioner", preconditioner);
    amg_data.print(pcout);
  }

  typename dealii::Utilities::MPI::RemotePointEvaluation<dim>::AdditionalData rpe_data;
  SolverData                                                                  solver_data;
  PreconditionerMass                                                          preconditioner;
  AMGData                                                                     amg_data;

  // Number of additional integration points used for sampling the source grid.
  // The default `additional_quadrature_points = 1` considers `fe_degree + 1` quadrature points in
  // 1D using the `fe_degree` of the target grid's finite element.
  unsigned int additional_quadrature_points;

  // Having identical grids and mappings, one can avoid RPE for mapped configurations. This flag has
  // to be provided by the user, to avoid an expensive check for independent copies of identical
  // grids and mappings.
  bool grids_and_maps_identical;
};

/**
 * Utility function to collect integration points in the exact sequence they are encountered
 * in.
 */
template<int dim, int n_components, typename Number>
std::vector<dealii::Point<dim>>
collect_integration_points(
  dealii::MatrixFree<dim, Number, dealii::VectorizedArray<Number>> const & matrix_free,
  unsigned int const                                                       dof_index,
  unsigned int const                                                       quad_index)
{
  CellIntegrator<dim, n_components, Number> fe_eval(matrix_free, dof_index, quad_index);

  // Conservative estimate for the number of points.
  std::vector<dealii::Point<dim>> integration_points;
  integration_points.reserve(
    matrix_free.get_dof_handler(dof_index).get_triangulation().n_active_cells() *
    fe_eval.n_q_points);

  for(unsigned int cell_batch_idx = 0; cell_batch_idx < matrix_free.n_cell_batches();
      ++cell_batch_idx)
  {
    fe_eval.reinit(cell_batch_idx);
    for(const unsigned int q : fe_eval.quadrature_point_indices())
    {
      dealii::Point<dim, dealii::VectorizedArray<Number>> const cell_batch_points =
        fe_eval.quadrature_point(q);
      for(unsigned int i = 0; i < dealii::VectorizedArray<Number>::size(); ++i)
      {
        dealii::Point<dim> p;
        for(unsigned int d = 0; d < dim; ++d)
        {
          p[d] = cell_batch_points[d][i];
        }
        integration_points.push_back(p);
      }
    }
  }

  return integration_points;
}

/**
 * Utility function to compute the right-hand side of a projection (mass matrix solve), assuming the
 * `dealii::DoFHandler`s corresponding to *identical* grids with *identical* mappings, which are
 * distributed *identically*. The function space, however, might be different. We interpolate the
 * source and assemble the projection right-hand side with the target finite element.
 */
template<int dim, int n_components, typename Number, typename VectorType>
VectorType
assemble_projection_rhs(VectorType &                              system_rhs,
                        VectorType const &                        source_vector,
                        dealii::DoFHandler<dim> const &           source_dof_handler,
                        dealii::Mapping<dim> const &              source_and_target_mapping,
                        dealii::DoFHandler<dim> const &           target_dof_handler,
                        dealii::AffineConstraints<Number> const & target_constraints,
                        dealii::Quadrature<dim> const &           quadrature_dim)
{
  // Check if the triangulations *might* be identical, assuming there is no adaptive refinement.
  dealii::Triangulation<dim> const & source_triangulation = source_dof_handler.get_triangulation();
  dealii::Triangulation<dim> const & target_triangulation = target_dof_handler.get_triangulation();
  bool const                         spatial_resolution_identical =
    source_triangulation.n_global_levels() == target_triangulation.n_global_levels() and
    source_triangulation.n_global_active_cells() == target_triangulation.n_global_active_cells();
  AssertThrow(spatial_resolution_identical == true,
              dealii::ExcMessage("The triangulations used cannot be identical."));

  dealii::FiniteElement<dim> const & source_fe = source_dof_handler.get_fe();
  dealii::FiniteElement<dim> const & target_fe = target_dof_handler.get_fe();

  const dealii::Quadrature<1> & quadrature = quadrature_dim.get_tensor_basis()[0];

  dealii::FEEvaluation<dim, -1, 0, n_components, Number, dealii::VectorizedArray<Number, 1>>
    fe_values_source(source_and_target_mapping, source_fe, quadrature, dealii::update_values);
  dealii::FEEvaluation<dim, -1, 0, n_components, Number, dealii::VectorizedArray<Number, 1>>
    fe_values_target(source_and_target_mapping,
                     target_fe,
                     quadrature,
                     dealii::update_values | dealii::update_JxW_values);

  dealii::Vector<Number>                       cell_rhs(target_fe.dofs_per_cell);
  std::vector<dealii::types::global_dof_index> dof_indices(target_fe.dofs_per_cell);

  for(const auto & cell_target : target_dof_handler.active_cell_iterators())
  {
    const auto cell_source = cell_target->as_dof_handler_iterator(source_dof_handler);
    AssertThrow(cell_source->id() == cell_target->id(),
                dealii::ExcMessage("Meshes were supposed to be iterated through the "
                                   "same way, but got " +
                                   cell_source->id().to_string() + " vs " +
                                   cell_target->id().to_string()));

    if(cell_target->is_locally_owned())
    {
      fe_values_source.reinit(cell_source);
      fe_values_target.reinit(cell_target);

      fe_values_source.read_dof_values(source_vector);
      fe_values_source.evaluate(dealii::EvaluationFlags::values);
      for(const unsigned int q : fe_values_source.quadrature_point_indices())
        fe_values_target.submit_value(fe_values_source.get_value(q), q);

      fe_values_target.integrate(dealii::EvaluationFlags::values);
      for(unsigned int i = 0; i < cell_rhs.size(); ++i)
        cell_rhs(fe_values_target.get_internal_dof_numbering()[i]) =
          fe_values_target.begin_dof_values()[i][0];
      cell_target->get_dof_indices(dof_indices);
      target_constraints.distribute_local_to_global(cell_rhs, dof_indices, system_rhs);
    }
  }
  system_rhs.compress(dealii::VectorOperation::add);

  return system_rhs;
}

/**
 * Utility function to compute the right-hand side of a projection (mass matrix solve)
 * with values given in integration points in the exact sequence they are encountered in.
 */
template<int dim, int n_components, typename Number, typename VectorType>
VectorType
assemble_projection_rhs(
  dealii::MatrixFree<dim, Number, dealii::VectorizedArray<Number>> const & matrix_free,
  std::vector<
    typename dealii::FEPointEvaluation<n_components, dim, dim, Number>::value_type> const &
                     values_source_in_q_points_target,
  unsigned int const dof_index,
  unsigned int const quad_index)
{
  VectorType system_rhs;
  matrix_free.initialize_dof_vector(system_rhs, dof_index);

  CellIntegrator<dim, n_components, Number> fe_eval(matrix_free, dof_index, quad_index);

  unsigned int idx_q_point = 0;

  for(unsigned int cell_batch_idx = 0; cell_batch_idx < matrix_free.n_cell_batches();
      ++cell_batch_idx)
  {
    fe_eval.reinit(cell_batch_idx);
    for(unsigned int const q : fe_eval.quadrature_point_indices())
    {
      dealii::Tensor<1, n_components, dealii::VectorizedArray<Number>> tmp;

      for(unsigned int i = 0; i < dealii::VectorizedArray<Number>::size(); ++i)
      {
        typename dealii::FEPointEvaluation<n_components, dim, dim, Number>::value_type const
          values = values_source_in_q_points_target[idx_q_point];

        // Increment index into `values_source_in_q_points_target`, meaning that the sequence of
        // function values in integration points need to match the particular sequence here.
        ++idx_q_point;

        if constexpr(n_components == 1)
        {
          tmp[0][i] = values;
        }
        else
        {
          for(unsigned int c = 0; c < n_components; ++c)
          {
            tmp[c][i] = values[c];
          }
        }
      }

      fe_eval.submit_value(tmp, q);
    }
    fe_eval.integrate(dealii::EvaluationFlags::values);
    fe_eval.distribute_local_to_global(system_rhs);
  }
  system_rhs.compress(dealii::VectorOperation::add);

  return system_rhs;
}

/**
 * Utilitiy function to project vectors from a source to a target triangulation via
 * matrix-free mass operator evaluation and preconditioned CG solver.
 */
template<int dim, typename Number, int n_components, typename VectorType>
void
project_vectors(
  std::shared_ptr<dealii::Mapping<dim> const> const &                      source_mapping,
  dealii::DoFHandler<dim> const &                                          source_dof_handler,
  std::vector<VectorType *> const &                                        source_vectors,
  dealii::MatrixFree<dim, Number, dealii::VectorizedArray<Number>> const & target_matrix_free,
  std::vector<VectorType *> const &                                        target_vectors,
  unsigned int const                                                       dof_index,
  unsigned int const                                                       quad_index,
  GridToGridProjectionData<dim> const &                                    data)
{
  // Setup inverse mass operator.
  InverseMassOperatorData<Number> inverse_mass_operator_data;
  inverse_mass_operator_data.dof_index                 = dof_index;
  inverse_mass_operator_data.quad_index                = quad_index;
  inverse_mass_operator_data.parameters.preconditioner = data.preconditioner;
  inverse_mass_operator_data.parameters.solver_data    = data.solver_data;
  inverse_mass_operator_data.parameters.amg_data       = data.amg_data;

  const bool cartesian_or_affine_mapping =
    std::all_of(target_matrix_free.get_mapping_info().cell_type.begin(),
                target_matrix_free.get_mapping_info().cell_type.end(),
                [](auto g) {
                  return g <= dealii::internal::MatrixFreeFunctions::GeometryType::affine;
                });
  inverse_mass_operator_data.parameters.implementation_type =
    InverseMassOperatorData<Number>::get_optimal_inverse_mass_type(
      target_matrix_free.get_dof_handler(dof_index).get_fe(), cartesian_or_affine_mapping);

  InverseMassOperator<dim, n_components, Number> inverse_mass_operator;
  inverse_mass_operator.initialize(target_matrix_free, inverse_mass_operator_data);

  MPI_Comm const             mpi_comm = source_dof_handler.get_mpi_communicator();
  dealii::ConditionalOStream pcout(std::cout,
                                   (dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0));
  pcout << std::scientific << std::setprecision(4);
  std::chrono::time_point<std::chrono::high_resolution_clock> time_start, time_end;

  // If the grids and maps are *not* identical, interpolate the source grid in the target grid's
  // integration points using `dealii::RemotePointEvaluation`.
  std::shared_ptr<dealii::Utilities::MPI::RemotePointEvaluation<dim>> rpe_source =
    std::make_shared<dealii::Utilities::MPI::RemotePointEvaluation<dim>>(data.rpe_data);
  if(not data.grids_and_maps_identical)
  {
    // The sequence of integration points follows from the sequence of points as encountered during
    // cell batch loop.
    std::vector<dealii::Point<dim>> integration_points_target =
      collect_integration_points<dim, n_components, Number>(target_matrix_free,
                                                            dof_index,
                                                            quad_index);

    time_start = std::chrono::high_resolution_clock::now();
    rpe_source->reinit(integration_points_target,
                       source_dof_handler.get_triangulation(),
                       *source_mapping);
    time_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> time_elapsed_rpe_reinit = time_end - time_start;

    pcout << "time: RPE reinit         " << time_elapsed_rpe_reinit.count() << " s\n";

    if(not rpe_source->all_points_found())
    {
      write_points_in_dummy_triangulation(integration_points_target,
                                          "./",
                                          "points_all",
                                          dealii::Utilities::MPI::this_mpi_process(mpi_comm),
                                          mpi_comm);

      std::vector<dealii::Point<dim>> points_not_found;
      points_not_found.reserve(integration_points_target.size());
      for(unsigned int i = 0; i < integration_points_target.size(); ++i)
      {
        if(not rpe_source->point_found(i))
        {
          points_not_found.push_back(integration_points_target[i]);
        }
      }

      write_points_in_dummy_triangulation(points_not_found, "./", "points_not_found", 0, mpi_comm);

      MPI_Barrier(mpi_comm);

      AssertThrow(
        rpe_source->all_points_found(),
        dealii::ExcMessage(
          "Could not interpolate source grid vector in target grid. "
          "Points exported to `./points_all_points.pvtu` and `./points_not_found_points.pvtu`"));
    }
  }

  // Loop over vectors and project.
  double time_elapsed_rpe_interpolate    = 0.0;
  double time_elapsed_inverse_mass_rhs   = 0.0;
  double time_elapsed_inverse_mass_apply = 0.0;
  for(unsigned int i = 0; i < target_vectors.size(); ++i)
  {
    // Evaluate the source vector at the target integration points.
    VectorType const & source_vector = *source_vectors.at(i);
    source_vector.update_ghost_values();

    // Assemble right-hand side for the projection.
    VectorType system_rhs;
    if(data.grids_and_maps_identical)
    {
      time_start = std::chrono::high_resolution_clock::now();
      target_matrix_free.initialize_dof_vector(system_rhs, dof_index);
      assemble_projection_rhs<dim, n_components, Number, VectorType>(
        system_rhs,
        source_vector,
        source_dof_handler,
        *source_mapping /* source_and_target_mapping */,
        target_matrix_free.get_dof_handler(dof_index),
        target_matrix_free.get_affine_constraints(dof_index),
        target_matrix_free.get_quadrature(quad_index));
      time_end = std::chrono::high_resolution_clock::now();
      time_elapsed_inverse_mass_rhs += std::chrono::duration<double>(time_end - time_start).count();
    }
    else
    {
      time_start = std::chrono::high_resolution_clock::now();
      std::vector<
        typename dealii::FEPointEvaluation<n_components, dim, dim, Number>::value_type> const
        values_source_in_q_points_target = dealii::VectorTools::point_values<n_components>(
          *rpe_source,
          source_dof_handler,
          source_vector,
          dealii::VectorTools::EvaluationFlags::avg);
      time_end = std::chrono::high_resolution_clock::now();
      time_elapsed_rpe_interpolate += std::chrono::duration<double>(time_end - time_start).count();

      time_start = std::chrono::high_resolution_clock::now();
      system_rhs = assemble_projection_rhs<dim, n_components, Number, VectorType>(
        target_matrix_free, values_source_in_q_points_target, dof_index, quad_index);
      time_end = std::chrono::high_resolution_clock::now();
      time_elapsed_inverse_mass_rhs += std::chrono::duration<double>(time_end - time_start).count();
    }

    // Solve linear system starting from zero initial guess.
    VectorType sol;
    sol.reinit(system_rhs, false /* omit_zeroing_entries */);

    time_start = std::chrono::high_resolution_clock::now();
    inverse_mass_operator.apply(sol, system_rhs);
    time_end = std::chrono::high_resolution_clock::now();
    time_elapsed_inverse_mass_apply += std::chrono::duration<double>(time_end - time_start).count();

    pcout << "global CG iterations in projection : "
          << inverse_mass_operator.get_n_iter_global_last() << "\n";

    // Copy solution to target vector.
    *target_vectors[i] = sol;
  }

  pcout << "time: RPE interpolate    " << time_elapsed_rpe_interpolate << " s\n";
  pcout << "time: inverse mass rhs   " << time_elapsed_inverse_mass_rhs << " s\n";
  pcout << "time: inverse mass apply " << time_elapsed_inverse_mass_apply << " s\n";
}

/**
 * Utility function to perform matrix-free grid-to-grid projection. We assume we only have a single
 * `dealii::FiniteElement` per `dealii::DoFHandler`. This function requires a suitable `MatrixFree`
 * object.
 */
template<int dim, typename Number, typename VectorType>
void
do_grid_to_grid_projection(
  std::shared_ptr<dealii::Mapping<dim> const> const &  source_mapping,
  std::vector<dealii::DoFHandler<dim> const *> const & source_dof_handlers,
  std::vector<std::vector<VectorType *>> const &       source_vectors_per_dof_handler,
  std::vector<dealii::DoFHandler<dim> const *> const & target_dof_handlers,
  dealii::MatrixFree<dim, Number, dealii::VectorizedArray<Number>> const & target_matrix_free,
  std::vector<std::vector<VectorType *>> & target_vectors_per_dof_handler,
  GridToGridProjectionData<dim> const &    data)
{
  // Check input dimensions.
  AssertThrow(source_vectors_per_dof_handler.size() == source_dof_handlers.size(),
              dealii::ExcMessage("First dimension of source vector of vectors "
                                 "has to match source DoFHandler count."));
  AssertThrow(target_vectors_per_dof_handler.size() == target_dof_handlers.size(),
              dealii::ExcMessage("First dimension of target vector of vectors "
                                 "has to match target DoFHandler count."));
  AssertThrow(source_dof_handlers.size() == target_dof_handlers.size(),
              dealii::ExcMessage("Target and source DoFHandler counts have to match"));
  AssertThrow(source_vectors_per_dof_handler.size() > 0,
              dealii::ExcMessage("Vector of source vectors empty."));
  for(unsigned int i = 0; i < source_vectors_per_dof_handler.size(); ++i)
  {
    AssertThrow(source_vectors_per_dof_handler[i].size() ==
                  target_vectors_per_dof_handler.at(i).size(),
                dealii::ExcMessage("Vectors of source and target vectors need to have same size."));
  }

  // Project vectors per `dealii::DoFHandler`.
  for(unsigned int i = 0; i < target_dof_handlers.size(); ++i)
  {
    unsigned int const n_components = target_dof_handlers[i]->get_fe().n_components();
    if(n_components == 1)
    {
      project_vectors<dim, Number, 1 /* n_components */, VectorType>(
        source_mapping,
        *source_dof_handlers.at(i),
        source_vectors_per_dof_handler.at(i),
        target_matrix_free,
        target_vectors_per_dof_handler.at(i),
        i /* dof_index */,
        i /* quad_index */,
        data);
    }
    else if(n_components == dim)
    {
      project_vectors<dim, Number, dim /* n_components */, VectorType>(
        source_mapping,
        *source_dof_handlers.at(i),
        source_vectors_per_dof_handler.at(i),
        target_matrix_free,
        target_vectors_per_dof_handler.at(i),
        i /* dof_index */,
        i /* quad_index */,
        data);
    }
    else if(n_components == dim + 2)
    {
      project_vectors<dim, Number, dim + 2 /* n_components */, VectorType>(
        source_mapping,
        *source_dof_handlers.at(i),
        source_vectors_per_dof_handler.at(i),
        target_matrix_free,
        target_vectors_per_dof_handler.at(i),
        i /* dof_index */,
        i /* quad_index */,
        data);
    }
    else
    {
      AssertThrow(n_components == 1 or n_components == dim,
                  dealii::ExcMessage("The requested number of components is not"
                                     "supported in `grid_to_grid_projection()`."));
    }
  }
}

/**
 * Same as the function above, but creates a suitable `MatrixFree` object, which is expensive.
 */
template<int dim, typename Number, typename VectorType>
void
grid_to_grid_projection(
  std::shared_ptr<dealii::Mapping<dim> const> const &  source_mapping,
  std::vector<dealii::DoFHandler<dim> const *> const & source_dof_handlers,
  std::vector<std::vector<VectorType *>> const &       source_vectors_per_dof_handler,
  std::shared_ptr<dealii::Mapping<dim> const> const &  target_mapping,
  std::vector<dealii::DoFHandler<dim> const *> const & target_dof_handlers,
  std::vector<std::vector<VectorType *>> &             target_vectors_per_dof_handler,
  GridToGridProjectionData<dim> const &                data)
{
  // Setup a single `dealii::MatrixFree` object with multiple `dealii::DoFHandler`s.
  MatrixFreeData<dim, Number> target_matrix_free_data;

  MappingFlags mapping_flags;
  mapping_flags.cells =
    dealii::update_quadrature_points | dealii::update_values | dealii::update_JxW_values;
  target_matrix_free_data.append_mapping_flags(mapping_flags);

  dealii::AffineConstraints<Number> empty_constraints;
  empty_constraints.clear();
  empty_constraints.close();
  for(unsigned int i = 0; i < target_dof_handlers.size(); ++i)
  {
    target_matrix_free_data.insert_dof_handler(target_dof_handlers[i], std::to_string(i));
    target_matrix_free_data.insert_constraint(&empty_constraints, std::to_string(i));

    ElementType element_type = get_element_type(target_dof_handlers[i]->get_triangulation());

    std::shared_ptr<dealii::Quadrature<dim>> quadrature = create_quadrature<dim>(
      element_type, target_dof_handlers[i]->get_fe().degree + data.additional_quadrature_points);

    target_matrix_free_data.insert_quadrature(*quadrature, std::to_string(i));
  }

  dealii::MatrixFree<dim, Number, dealii::VectorizedArray<Number>> target_matrix_free;

  target_matrix_free.reinit(*target_mapping,
                            target_matrix_free_data.get_dof_handler_vector(),
                            target_matrix_free_data.get_constraint_vector(),
                            target_matrix_free_data.get_quadrature_vector(),
                            target_matrix_free_data.data);

  do_grid_to_grid_projection<dim, Number, VectorType>(source_mapping,
                                                      source_dof_handlers,
                                                      source_vectors_per_dof_handler,
                                                      target_dof_handlers,
                                                      target_matrix_free,
                                                      target_vectors_per_dof_handler,
                                                      data);
}

} // namespace GridToGridProjection
} // namespace ExaDG

#endif /* EXADG_OPERATORS_SOLUTION_PROJECTION_BETWEEN_TRIANGULATIONS_H_ */
