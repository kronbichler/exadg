/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2025 by Martin Kronbichler, Shubham Goswami,
 *  Richard Schussnig
 *
 *  This file is dual-licensed under the Apache-2.0 with LLVM Exception (see
 *  https://spdx.org/licenses/Apache-2.0.html and
 *  https://spdx.org/licenses/LLVM-exception.html) and the GNU General Public
 *  License as published by the Free Software Foundation, either version 3 of
 *  the License, or (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License in the top-level LICENSE file for
 *  more details.
 *  ______________________________________________________________________
 */

#pragma once

#include <deal.II/base/floating_point_comparator.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/memory_space_data.h>
#include <deal.II/distributed/repartitioning_policy_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tensor_product_kernels.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/numerics/vector_tools.h>

#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/laplace_operator_extruded.h>


namespace LaplaceOperator
{
template<typename Number>
void
make_zero_mean(const std::vector<unsigned int> &                    constrained_dofs,
               dealii::LinearAlgebra::distributed::Vector<Number> & vec)
{
  // set constrained entries to zero
  for(const unsigned int index : constrained_dofs)
    vec.local_element(index) = 0.;

  // rescale mean value computed among all vector entries to the vector size
  // without constraints
  const unsigned int n_unconstrained_dofs = vec.locally_owned_size() - constrained_dofs.size();
  vec.add(-vec.mean_value() * vec.size() /
          dealii::Utilities::MPI::sum(n_unconstrained_dofs, vec.get_mpi_communicator()));

  // set constrained entries to zero again, this should now have zero mean
  for(const unsigned int index : constrained_dofs)
    vec.local_element(index) = 0.;
}



template<int dim, typename Number = float>
class PoissonPreconditionerMG
{
public:
  using VectorizedArrayType        = dealii::VectorizedArray<Number>;
  using MatrixType                 = LaplaceOperator::LaplaceOperatorFE<dim, Number>;
  using MatrixTypeDG               = LaplaceOperator::LaplaceOperatorDG<dim, Number>;
  using VectorType                 = dealii::LinearAlgebra::distributed::Vector<Number>;
  using SmootherPreconditionerType = dealii::DiagonalMatrix<VectorType>;
  using SmootherType =
    dealii::PreconditionChebyshev<MatrixType, VectorType, SmootherPreconditionerType>;
  using SmootherTypeDG =
    dealii::PreconditionChebyshev<MatrixTypeDG, VectorType, SmootherPreconditionerType>;
  using MGTransferType = dealii::MGTransferGlobalCoarsening<dim, VectorType>;

  PoissonPreconditionerMG(
    const dealii::Mapping<dim> &                                   mapping_fine,
    const dealii::DoFHandler<dim> &                                dof_handler,
    const std::vector<unsigned int> &                              cell_vectorization_category,
    const std::function<std::vector<dealii::Point<dim>>(
                                                        typename dealii::Triangulation<dim>::cell_iterator const)> & mapping_function,
                          const Number ip_factor,
                          const bool is_test)
    : coarse_triangulations(
        dealii::MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(
          dof_handler.get_triangulation()/*,
                                           dealii::RepartitioningPolicyTools::MinimalGranularityPolicy<dim>(64)*/)),
      fe_hierarchy(create_fe_hierarchy(dof_handler.get_fe())),
      min_level(0),
      max_level(dof_handler.get_triangulation().n_global_levels() - 1 + fe_hierarchy.max_level()),
      is_test(is_test)
  {
    dof_handler_hierarchy.resize(min_level, max_level);

    level_constraints.resize(min_level, max_level);
    mg_matrices.resize(min_level, max_level);
    mg_smoother.resize(min_level, max_level);
    mg_transfers.resize(min_level, max_level);
    rhs.resize(min_level, max_level);
    temp_vector.resize(min_level, max_level);
    solution_update.resize(min_level, max_level);

    // initialize levels
    for(unsigned int level = min_level; level <= max_level; level++)
    {
      dealii::AffineConstraints<Number> constraints;
      dealii::DoFHandler<dim> &         dof_h = dof_handler_hierarchy[level];
      dof_h.reinit(
        *coarse_triangulations[std::min(level,
                                        dof_handler.get_triangulation().n_global_levels() - 1)]);
      if(level < coarse_triangulations.size())
        dof_h.distribute_dofs(*fe_hierarchy[0]);
      else
        dof_h.distribute_dofs(*fe_hierarchy[level - coarse_triangulations.size() + 1]);

      level_constraints[level].reinit(dof_h.locally_owned_dofs(),
                                      dealii::DoFTools::extract_locally_relevant_dofs(dof_h));

      dealii::ndarray<unsigned int, dim, 2> periodic_ids;
      for(unsigned int d = 0; d < dim; ++d)
        for(unsigned int e = 0; e < 2; ++e)
          periodic_ids[d][e] = numbers::invalid_unsigned_int;
      {
        for(const auto & cell : dof_h.cell_iterators_on_level(0))
          for(unsigned int d = 0; d < dim; ++d)
            if(cell->at_boundary(2 * d) && cell->has_periodic_neighbor(2 * d))
            {
              periodic_ids[d][0] = cell->face(2 * d)->boundary_id();
              periodic_ids[d][1] = cell->periodic_neighbor(2 * d)
                                     ->face(cell->periodic_neighbor_face_no(2 * d))
                                     ->boundary_id();
            }
        for(unsigned int d = 0; d < dim; ++d)
          if(periodic_ids[d][0] != numbers::invalid_unsigned_int)
            dealii::DoFTools::make_periodicity_constraints(
              dof_h, periodic_ids[d][0], periodic_ids[d][1], d, level_constraints[level]);
      }
      level_constraints[level].close();

      typename dealii::MatrixFree<dim, Number>::AdditionalData mf_data;
      if(level >= coarse_triangulations.size())
        mf_data.cell_vectorization_category = cell_vectorization_category;

      // renumber Dofs to minimize the number of partitions in import
      // indices of partitioner
      dealii::DoFRenumbering::matrix_free_data_locality(dof_h, level_constraints[level], mf_data);
      level_constraints[level].reinit(dof_h.locally_owned_dofs(),
                                      dealii::DoFTools::extract_locally_relevant_dofs(dof_h));
      for(unsigned int d = 0; d < dim; ++d)
        if(periodic_ids[d][0] != numbers::invalid_unsigned_int)
          dealii::DoFTools::make_periodicity_constraints(
            dof_h, periodic_ids[d][0], periodic_ids[d][1], d, level_constraints[level]);

      level_constraints[level].close();
      if(level < coarse_triangulations.size())
      {
        if(mapping_function)
        {
          dealii::MappingQCache<dim> mapping_coarse(1);
          mapping_coarse.initialize(dof_h.get_triangulation(), mapping_function);
          mg_matrices[level].reinit(mapping_coarse,
                                    dof_h,
                                    level_constraints[level],
                                    {},
                                    dealii::QGauss<1>(dof_h.get_fe().degree + 1));
        }
        else
        {
          dealii::MappingQ1<dim> mapping_coarse;
          mg_matrices[level].reinit(mapping_coarse,
                                    dof_h,
                                    level_constraints[level],
                                    {},
                                    dealii::QGauss<1>(dof_h.get_fe().degree + 1));
        }
      }
      else
        mg_matrices[level].reinit(mapping_fine,
                                  dof_h,
                                  level_constraints[level],
                                  cell_vectorization_category,
                                  dealii::QGauss<1>(dof_h.get_fe().degree + 1));

      // initialize transfer operator
      if(level > 0)
      {
        auto transfer = std::make_unique<dealii::MGTwoLevelTransfer<dim, VectorType>>();
        transfer->reinit(dof_h,
                         dof_handler_hierarchy[level - 1],
                         level_constraints[level],
                         level_constraints[level - 1]);
        transfer->enable_inplace_operations_if_possible(
          mg_matrices[level - 1].get_matrix_free().get_dof_info().vector_partitioner,
          mg_matrices[level].get_matrix_free().get_dof_info().vector_partitioner);

        mg_transfers[level - 1] = std::move(transfer);
      }
    }

    // initialize levels
    for(unsigned int level = min_level; level <= max_level; level++)
    {
      // ... initialize smoother
      typename SmootherType::AdditionalData smoother_data;
      smoother_data.preconditioner = std::make_shared<SmootherPreconditionerType>();
      mg_matrices[level].compute_inverse_diagonal(smoother_data.preconditioner->get_vector());
      smoother_data.smoothing_range = 20.;
      smoother_data.degree          = 4;

      // manually compute the eigenvalue estimate for Chebyshev because we
      // need to be careful with the constrained indices
      dealii::IterationNumberControl control(12, 1e-6, false, false);

      dealii::SolverCG<VectorType>        solver(control);
      dealii::internal::EigenvalueTracker eigenvalue_tracker;
      solver.connect_eigenvalues_slot(
        [&eigenvalue_tracker](const std::vector<double> & eigenvalues) {
          eigenvalue_tracker.slot(eigenvalues);
        });

      mg_matrices[level].initialize_dof_vector(solution_update[level]);
      mg_matrices[level].initialize_dof_vector(temp_vector[level]);
      mg_matrices[level].initialize_dof_vector(rhs[level]);

      dealii::internal::set_initial_guess(rhs[level]);
      make_zero_mean(mg_matrices[level].get_matrix_free().get_constrained_dofs(), rhs[level]);

      // TODO: this call raises an exception when running with many MPI processes on coarse grids.
      solver.solve(mg_matrices[level],
                   temp_vector[level],
                   rhs[level],
                   *smoother_data.preconditioner);

      smoother_data.eig_cg_n_iterations = 0;
      if(eigenvalue_tracker.values.empty())
        smoother_data.max_eigenvalue = 1.0;
      else
        smoother_data.max_eigenvalue = eigenvalue_tracker.values.back();

      mg_smoother[level].initialize(mg_matrices[level], smoother_data);
    }

    // create a different matrix on the finest level due to enable the
    // efficient implementation of the DG discretization
    dealii::AffineConstraints<Number> empty_constraints;
    empty_constraints.close();
    dg_matrix.reinit(mapping_fine,
                     dof_handler,
                     empty_constraints,
                     cell_vectorization_category,
                     dealii::QGauss<1>(dof_handler.get_fe().degree + 1),
                     this->is_test);
    dg_matrix.set_penalty_parameters(ip_factor);
    {
      typename SmootherTypeDG::AdditionalData smoother_data_dg;
      smoother_data_dg.preconditioner = std::make_shared<SmootherPreconditionerType>();
      dg_matrix.compute_inverse_diagonal(smoother_data_dg.preconditioner->get_vector());
      smoother_data_dg.smoothing_range = 20.;
      smoother_data_dg.degree          = 4;

      // manually compute the eigenvalue estimate for Chebyshev because of
      // mean values
      dealii::IterationNumberControl control(12, 1e-6, false, false);

      dealii::SolverCG<VectorType>        solver(control);
      dealii::internal::EigenvalueTracker eigenvalue_tracker;
      solver.connect_eigenvalues_slot(
        [&eigenvalue_tracker](const std::vector<double> & eigenvalues) {
          eigenvalue_tracker.slot(eigenvalues);
        });

      dg_matrix.initialize_dof_vector(solution_update_dg);
      dg_matrix.initialize_dof_vector(rhs_dg);

      dealii::internal::set_initial_guess(rhs_dg);
      make_zero_mean({}, rhs_dg);
      solver.solve(dg_matrix, solution_update_dg, rhs_dg, *smoother_data_dg.preconditioner);

      smoother_data_dg.eig_cg_n_iterations = 0;
      if(eigenvalue_tracker.values.empty())
        smoother_data_dg.max_eigenvalue = 1.0;
      else
        smoother_data_dg.max_eigenvalue = eigenvalue_tracker.values.back();
      mg_smoother_dg.initialize(dg_matrix, smoother_data_dg);
    }
    auto transfer = std::make_unique<dealii::MGTwoLevelTransfer<dim, VectorType>>();
    transfer->reinit(dof_handler,
                     dof_handler_hierarchy[max_level],
                     empty_constraints,
                     level_constraints[max_level]);
    transfer->enable_inplace_operations_if_possible(
      mg_matrices[max_level].get_matrix_free().get_dof_info().vector_partitioner,
      dg_matrix.get_matrix_free().get_dof_info().vector_partitioner);

    mg_transfers[max_level] = std::move(transfer);

    timings.clear();
    timings.resize(max_level + 2);
    count_times = 0;
  }

  ~PoissonPreconditionerMG()
  {
    if(not this->is_test and Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      double all_time = 0;
      for(const auto & array : timings)
        for(const auto time : array)
          all_time += time;
      std::cout << "Collected multigrid timings in " << count_times
                << " evaluations [t_total=" << all_time << "s]" << std::endl;
      std::cout << "Level   smooth      residual    restrict    prolongate   [rel. share]"
                << std::endl;
      for(unsigned int i = 0; i < timings.size(); ++i)
      {
        std::cout << std::left << std::scientific << std::setw(8) << i << std::setw(12)
                  << std::setprecision(3) << timings[i][1] << std::setw(12) << timings[i][2]
                  << std::setw(12) << timings[i][0] << std::setw(13) << timings[i][3]
                  << std::defaultfloat
                  << (timings[i][0] + timings[i][1] + timings[i][2] + timings[i][3]) / all_time *
                       100
                  << "%" << std::endl;
      }
      std::cout << std::endl;
    }
  }

  static dealii::MGLevelObject<std::unique_ptr<dealii::FE_Q<dim>>>
  create_fe_hierarchy(const dealii::FiniteElement<dim> & fe)
  {
    std::vector<unsigned int> p_levels({fe.degree});
    while(p_levels.back() > 1)
    {
      // pick the next coarser degree as half the previous degree; if
      // integer division has remainder, use the nearest even degree (i.e.,
      // we do steps like 2-1, 3-2-1, 4-2-1, 5-2-1, 6-3-2-1, 7-4-2-1, etc)
      const unsigned int tentative_degree = p_levels.back() / 2;
      if(p_levels.back() % 2 == 1 && tentative_degree % 2 == 1)
        p_levels.push_back(tentative_degree + 1);
      else
        p_levels.push_back(tentative_degree);
    }
    dealii::MGLevelObject<std::unique_ptr<dealii::FE_Q<dim>>> fes(0, p_levels.size() - 1);
    for(unsigned int level = 0; level < p_levels.size(); ++level)
      fes[level] = std::make_unique<dealii::FE_Q<dim>>(p_levels[p_levels.size() - 1 - level]);
    return fes;
  }

  template<typename VectorTypeOuter>
  void
  vmult(VectorTypeOuter & dst, const VectorTypeOuter & src) const
  {
    ++count_times;
    Timer time;
    rhs_dg.copy_locally_owned_data_from(src);
    timings.back()[0] += time.wall_time();
    time.restart();

    mg_smoother_dg.vmult(solution_update_dg, rhs_dg);
    timings.back()[1] += time.wall_time();
    time.restart();

    dg_matrix.vmult_residual_and_restrict_to_fe(rhs_dg,
                                                solution_update_dg,
                                                mg_matrices[max_level],
                                                rhs[max_level]);
    timings.back()[2] += time.wall_time();
    time.restart();

    for(unsigned int level = max_level; level > min_level; --level)
    {
      mg_smoother[level].vmult(solution_update[level], rhs[level]);
      timings[level][1] += time.wall_time();
      time.restart();

      mg_matrices[level].vmult(temp_vector[level], solution_update[level]);
      temp_vector[level].sadd(-1.0, 1.0, rhs[level]);
      timings[level][2] += time.wall_time();
      time.restart();

      rhs[level - 1] = 0;
      mg_transfers[level - 1]->restrict_and_add(rhs[level - 1], temp_vector[level]);
      timings[level][0] += time.wall_time();
      time.restart();
    }

    // coarse solver, taking into account zero mean
    make_zero_mean(mg_matrices[min_level].get_matrix_free().get_constrained_dofs(), rhs[min_level]);
    mg_smoother[min_level].vmult(solution_update[min_level], rhs[min_level]);
    make_zero_mean(mg_matrices[min_level].get_matrix_free().get_constrained_dofs(),
                   solution_update[min_level]);
    timings[min_level][1] += time.wall_time();
    time.restart();

    for(unsigned int level = min_level; level < max_level; ++level)
    {
      mg_transfers[level]->prolongate_and_add(solution_update[level + 1], solution_update[level]);
      timings[level + 1][3] += time.wall_time();
      time.restart();

      mg_smoother[level + 1].step(solution_update[level + 1], rhs[level + 1]);
      timings[level + 1][1] += time.wall_time();
      time.restart();
    }
    dg_matrix.prolongate_and_add(solution_update_dg,
                                 solution_update[max_level],
                                 mg_matrices[max_level]);
    timings.back()[3] += time.wall_time();
    time.restart();

    mg_smoother_dg.step(solution_update_dg, rhs_dg);
    timings.back()[1] += time.wall_time();
    time.restart();

    dst.copy_locally_owned_data_from(solution_update_dg);
    timings.back()[0] += time.wall_time();
  }

  const MatrixTypeDG &
  get_dg_matrix() const
  {
    return dg_matrix;
  }

private:
  std::vector<std::shared_ptr<const dealii::Triangulation<dim>>> coarse_triangulations;

  const dealii::MGLevelObject<std::unique_ptr<dealii::FE_Q<dim>>> fe_hierarchy;

  const unsigned int min_level;
  const unsigned int max_level;

  bool const is_test;

  dealii::MGLevelObject<dealii::DoFHandler<dim>> dof_handler_hierarchy;

  dealii::MGLevelObject<dealii::AffineConstraints<Number>> level_constraints;
  dealii::MGLevelObject<MatrixType>                        mg_matrices;

  MatrixTypeDG dg_matrix;

  SmootherType                        mg_coarse_grid_smoother;
  dealii::MGLevelObject<SmootherType> mg_smoother;
  SmootherTypeDG                      mg_smoother_dg;

  dealii::MGLevelObject<std::unique_ptr<dealii::MGTwoLevelTransferBase<dim, VectorType>>>
    mg_transfers;

  mutable dealii::MGLevelObject<VectorType> rhs;
  mutable dealii::MGLevelObject<VectorType> temp_vector;
  mutable dealii::MGLevelObject<VectorType> solution_update;

  mutable std::size_t                        count_times;
  mutable std::vector<std::array<double, 4>> timings;

  mutable VectorType rhs_dg;
  mutable VectorType solution_update_dg;
};

} // namespace LaplaceOperator
