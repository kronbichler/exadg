/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2025 - 2026 by Martin Kronbichler, Shubham Goswami,
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
#include <deal.II/fe/mapping_q_cache.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tensor_product_kernels.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/numerics/vector_tools.h>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <map>
#include <random>
#include <set>

#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/laplace_operator_extruded.h>
#include <exadg/operators/constraints.h>
#include <exadg/solvers_and_preconditioners/preconditioners/chebyshev_scratch.h>
#include <exadg/utilities/mpi_shared_memory.h>


namespace LaplaceOperator
{
template<typename Number>
void
make_zero_mean(const Number                                         scaling_factor_for_zero_mean,
               const std::vector<unsigned int> &                    constrained_dofs,
               dealii::LinearAlgebra::distributed::Vector<Number> & vec)
{
  // set constrained entries to zero
  for(const unsigned int index : constrained_dofs)
    vec.local_element(index) = 0.;

  vec.add(-vec.mean_value() * scaling_factor_for_zero_mean);

  // set constrained entries to zero again, the resulting vector should now
  // have zero mean
  for(const unsigned int index : constrained_dofs)
    vec.local_element(index) = 0.;
}

template<typename Number>
void
make_zero_mean(const std::vector<unsigned int> &                    constrained_dofs,
               dealii::LinearAlgebra::distributed::Vector<Number> & vec)
{
  const unsigned int n_unconstrained_dofs = vec.locally_owned_size() - constrained_dofs.size();
  const Number       scaling_factor_for_zero_mean =
    static_cast<Number>(vec.size()) /
    dealii::Utilities::MPI::sum(n_unconstrained_dofs, vec.get_mpi_communicator());
  make_zero_mean(scaling_factor_for_zero_mean, constrained_dofs, vec);
}


/**
 * Interface of the transfer operators of the Poisson multigrid that can
 * compute the residual b - A u on the fine level and restrict it in a single
 * cell loop. The operator part is computed cell by cell without summation
 * into a global vector and restricted directly, whereas the right hand side
 * is read with the masked indices of the restriction, where each DoF is
 * visited by exactly one cell. Compared to the separate matrix-vector
 * product, vector update and restriction, this avoids four passes through
 * the fine vector and the communication of the matrix-vector product.
 */
template<int dim, typename Number>
class MGTwoLevelTransferResidualRestriction
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  virtual ~MGTwoLevelTransferResidualRestriction() = default;

  /**
   * Compute the residual rhs - A * solution on the fine level, with A the
   * operator returned by get_operator_fine(), and restrict it to
   * the coarse level vector 'dst'.
   */
  virtual void
  restrict_residual(VectorType &       dst,
                    const VectorType & rhs,
                    const VectorType & solution) const = 0;

  virtual const LaplaceOperatorFE<dim, Number> &
  get_operator_fine() const = 0;
};



/**
 * This is a specialized transfer operator, designed for transfer between
 * p-levels only, that aims to utilize the fact that we have the same cells
 * and the same cell order to design a highly efficient transfer operation.
 */
template<int dim, typename Number>
class MGTwoLevelTransferFromOperator
  : public MGTwoLevelTransferBase<dim, LinearAlgebra::distributed::Vector<Number>>,
    public MGTwoLevelTransferResidualRestriction<dim, Number>
{
public:
  using VectorType                      = LinearAlgebra::distributed::Vector<Number>;
  static constexpr unsigned int n_lanes = VectorizedArray<Number>::size();

  MGTwoLevelTransferFromOperator(const LaplaceOperatorFE<dim, Number> & operator_fine,
                                 const LaplaceOperatorFE<dim, Number> & operator_coarse)
    : operator_fine(operator_fine), operator_coarse(operator_coarse)
  {
    AssertThrow(operator_fine.get_matrix_free()
                    .get_dof_handler()
                    .get_triangulation()
                    .n_global_active_cells() == operator_coarse.get_matrix_free()
                                                  .get_dof_handler()
                                                  .get_triangulation()
                                                  .n_global_active_cells(),
                ExcMessage("This transfer is designed for polynomial transfer between "
                           "matrix-free operators defined on the same mesh, but found "
                           "meshes with " +
                           std::to_string(operator_fine.get_matrix_free()
                                            .get_dof_handler()
                                            .get_triangulation()
                                            .n_global_active_cells()) +
                           " and " +
                           std::to_string(operator_coarse.get_matrix_free()
                                            .get_dof_handler()
                                            .get_triangulation()
                                            .n_global_active_cells()) +
                           " global cells, respectively."));
    AssertThrow(
      operator_fine.get_matrix_free().get_dof_handler().get_triangulation().n_active_cells() ==
        operator_coarse.get_matrix_free().get_dof_handler().get_triangulation().n_active_cells(),
      ExcMessage(
        "This transfer is designed for polynomial transfer between "
        "matrix-free operators defined on the same mesh partitioned "
        "in the same way, but found meshes with " +
        std::to_string(
          operator_fine.get_matrix_free().get_dof_handler().get_triangulation().n_active_cells()) +
        " and " +
        std::to_string(operator_coarse.get_matrix_free()
                         .get_dof_handler()
                         .get_triangulation()
                         .n_active_cells()) +
        " cells owned on the current MPI process, respectively."));
    AssertDimension(operator_fine.get_matrix_free().n_cell_batches(),
                    operator_coarse.get_matrix_free().n_cell_batches());
    for(unsigned int cell = 0; cell < operator_fine.get_matrix_free().n_cell_batches(); ++cell)
    {
      for(unsigned int v = 0;
          v < operator_fine.get_matrix_free().n_active_entries_per_cell_batch(cell);
          ++v)
      {
        AssertThrow(
          operator_fine.get_matrix_free().get_cell_iterator(cell, v)->level() ==
              operator_coarse.get_matrix_free().get_cell_iterator(cell, v)->level() &&
            operator_fine.get_matrix_free().get_cell_iterator(cell, v)->index() ==
              operator_coarse.get_matrix_free().get_cell_iterator(cell, v)->index(),
          ExcMessage(
            "The cell iterators on refined / coarse side did not match, "
            "do both MatrixFree objects use the same ordering of cells? Got: " +
            operator_fine.get_matrix_free().get_cell_iterator(cell, v)->id().to_string() + " vs " +
            operator_coarse.get_matrix_free().get_cell_iterator(cell, v)->id().to_string() + " " +
            std::to_string(cell) + " " + std::to_string(v)));
      }
    }

    // find unique indices of prolongation to identify the actual read
    // pattern; on cells with hanging nodes, only the DoFs actually owned by
    // the cell are considered, because the indices of the coarse side of
    // hanging-node constraints do not belong to the position within the cell
    fine_dof_indices = operator_fine.get_compressed_dof_indices_without_hanging_nodes();
    const unsigned int degree_fine =
      operator_fine.get_matrix_free().get_dof_handler().get_fe().degree;
    const unsigned int degree_coarse =
      operator_coarse.get_matrix_free().get_dof_handler().get_fe().degree;
    const unsigned int n_owned_dofs =
      operator_fine.get_matrix_free().get_dof_handler().locally_owned_dofs().n_elements();
    std::vector<char> touched_dof(n_owned_dofs, 0);
    for(unsigned int & dof : fine_dof_indices)
      if(dof >= n_owned_dofs)
        dof = numbers::invalid_unsigned_int;
      else if(touched_dof[dof] == 0)
        touched_dof[dof] = 1;
      else
        dof = numbers::invalid_unsigned_int;

    all_indices_unconstrained.resize(fine_dof_indices.size() / n_lanes, 0);

    std::vector<Polynomials::Polynomial<double>> poly_coarse =
      Polynomials::generate_complete_Lagrange_basis(
        QGaussLobatto<1>(degree_coarse + 1).get_points());
    std::vector<Point<1>> points_fine(QGaussLobatto<1>(degree_fine + 1).get_points());
    prolongation_matrix.reinit(degree_coarse + 1, degree_fine + 1);
    for(unsigned int i = 0; i < prolongation_matrix.m(); ++i)
      for(unsigned int j = 0; j < prolongation_matrix.n(); ++j)
        prolongation_matrix(i, j) = poly_coarse[i].value(points_fine[j][0]);
    prolongation_matrix_data.resize(prolongation_matrix.m() * prolongation_matrix.n());
    for(unsigned int i = 0, c = 0; i < prolongation_matrix.m(); ++i)
      for(unsigned int j = 0; j < prolongation_matrix.n(); ++j, ++c)
        prolongation_matrix_data[c] = prolongation_matrix(i, j);
  }

  /**
   * @copydoc MGTwoLevelTransferBase::prolongate_and_add
   */
  void
  prolongate_and_add(VectorType & dst, const VectorType & src) const override
  {
    src.update_ghost_values();

    const unsigned int size_fine      = prolongation_matrix.n();
    const unsigned int n_cell_batches = operator_fine.get_matrix_free().n_cell_batches();
    for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
    {
      if(size_fine == 2)
        do_prolongate_and_add_on_cell<2>(cell, dst, src);
      else if(size_fine == 3)
        do_prolongate_and_add_on_cell<3>(cell, dst, src);
      else if(size_fine == 4)
        do_prolongate_and_add_on_cell<4>(cell, dst, src);
      else if(size_fine == 5)
        do_prolongate_and_add_on_cell<5>(cell, dst, src);
      else if(size_fine == 6)
        do_prolongate_and_add_on_cell<6>(cell, dst, src);
      else if(size_fine == 7)
        do_prolongate_and_add_on_cell<7>(cell, dst, src);
      else if(size_fine == 8)
        do_prolongate_and_add_on_cell<8>(cell, dst, src);
      else if(size_fine == 9)
        do_prolongate_and_add_on_cell<9>(cell, dst, src);
      else
        AssertThrow(false,
                    ExcMessage("Fine degree " + std::to_string(size_fine - 1) +
                               " not instantiated"));
    }

    src.zero_out_ghost_values();
  }

  /**
   * @copydoc MGTwoLevelTransferBase::restrict_and_add
   */
  void
  restrict_and_add(VectorType & dst, const VectorType & src) const override
  {
    restrict_cells<false>(dst, src, nullptr);
  }

  /**
   * @copydoc MGTwoLevelTransferResidualRestriction::restrict_residual
   */
  void
  restrict_residual(VectorType &       dst,
                    const VectorType & rhs,
                    const VectorType & solution) const override
  {
    const bool has_ghosts = solution.has_ghost_elements();
    if(!has_ghosts)
      solution.update_ghost_values();
    restrict_cells<true>(dst, solution, &rhs);
    if(!has_ghosts)
      solution.zero_out_ghost_values();
  }

  const LaplaceOperatorFE<dim, Number> &
  get_operator_fine() const override
  {
    return operator_fine;
  }

  void
  interpolate(VectorType &, const VectorType &) const override
  {
    AssertThrow(false, ExcNotImplemented());
  }

  std::pair<bool, bool>
  enable_inplace_operations_if_possible(
    const std::shared_ptr<const Utilities::MPI::Partitioner> &,
    const std::shared_ptr<const Utilities::MPI::Partitioner> &) override
  {
    return std::make_pair(true, true);
  }

  std::size_t
  memory_consumption() const override
  {
    return MemoryConsumption::memory_consumption(fine_dof_indices) +
           MemoryConsumption::memory_consumption(all_indices_unconstrained) +
           prolongation_matrix.memory_consumption() + prolongation_matrix_data.memory_consumption();
  }

  std::pair<const DoFHandler<dim> *, unsigned int>
  get_dof_handler_fine() const override
  {
    return std::make_pair(&operator_fine.get_matrix_free().get_dof_handler(),
                          numbers::invalid_unsigned_int);
  }

private:
  const LaplaceOperatorFE<dim, Number> & operator_fine;
  const LaplaceOperatorFE<dim, Number> & operator_coarse;
  std::vector<unsigned int>              fine_dof_indices;
  std::vector<unsigned char>             all_indices_unconstrained;
  FullMatrix<Number>                     prolongation_matrix;
  AlignedVector<Number>                  prolongation_matrix_data;

  // Restriction of a vector src (residual == false) or of the residual *rhs - A src
  // with 'A src' evaluated within the same loop (residual == true), and summed into dst
  template<bool residual>
  void
  restrict_cells(VectorType & coarse_dst, const VectorType & src, const VectorType * rhs) const
  {
    int dummy = 0;
    operator_coarse.get_matrix_free().template cell_loop<VectorType, int>(
      [&](const auto &,
          VectorType & dst,
          const int &,
          const std::pair<unsigned int, unsigned int> & cell_range) {
        const unsigned int size_fine = prolongation_matrix.n();
        for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          if(size_fine == 2)
            do_restrict_and_add_on_cell<2, residual>(cell, dst, src, rhs);
          else if(size_fine == 3)
            do_restrict_and_add_on_cell<3, residual>(cell, dst, src, rhs);
          else if(size_fine == 4)
            do_restrict_and_add_on_cell<4, residual>(cell, dst, src, rhs);
          else if(size_fine == 5)
            do_restrict_and_add_on_cell<5, residual>(cell, dst, src, rhs);
          else if(size_fine == 6)
            do_restrict_and_add_on_cell<6, residual>(cell, dst, src, rhs);
          else if(size_fine == 7)
            do_restrict_and_add_on_cell<7, residual>(cell, dst, src, rhs);
          else if(size_fine == 8)
            do_restrict_and_add_on_cell<8, residual>(cell, dst, src, rhs);
          else if(size_fine == 9)
            do_restrict_and_add_on_cell<9, residual>(cell, dst, src, rhs);
          else
            AssertThrow(false,
                        ExcMessage("Fine degree " + std::to_string(size_fine - 1) +
                                   " not instantiated"));
        }
      },
      coarse_dst,
      dummy,
      /* zero_dst */ residual);
  }

  template<int size_fine>
  void
  do_prolongate_and_add_on_cell(const unsigned int cell,
                                VectorType &       dst,
                                const VectorType & src) const
  {
    constexpr int size_coarse = get_coarser_fe_degree(size_fine - 1) + 1;
    AssertDimension(prolongation_matrix.m(), size_coarse);
    VectorizedArray<Number> tmp[Utilities::pow(size_fine, dim)];

    operator_coarse.template read_dof_values<size_coarse - 1>(cell, src, tmp);
    dealii::internal::FEEvaluationImplBasisChange<dealii::internal::evaluate_general,
                                                  dealii::internal::EvaluatorQuantity::value,
                                                  dim,
                                                  size_coarse,
                                                  size_fine>::do_forward(1,
                                                                         prolongation_matrix_data,
                                                                         tmp,
                                                                         tmp);

    operator_fine.template distribute_local_to_global_compressed<size_fine - 1, size_fine, 1>(
      dst, fine_dof_indices, all_indices_unconstrained, cell, {}, true, tmp);
  }

  template<int size_fine, bool residual>
  void
  do_restrict_and_add_on_cell(const unsigned int cell,
                              VectorType &       dst,
                              const VectorType & src,
                              const VectorType * rhs) const
  {
    constexpr int size_coarse = get_coarser_fe_degree(size_fine - 1) + 1;
    AssertDimension(prolongation_matrix.m(), size_coarse);
    constexpr unsigned int  dofs_per_cell = Utilities::pow(size_fine, dim);
    VectorizedArray<Number> tmp[dofs_per_cell];

    operator_fine.template read_dof_values_compressed<size_fine - 1, size_fine, 1>(
      residual ? *rhs : src, fine_dof_indices, all_indices_unconstrained, cell, {}, true, tmp);
    if constexpr(residual)
    {
      // The operator is applied without the transpose of the hanging-node
      // interpolation and the cell values are restricted at all nodes: the
      // interpolation of the coarse space to the nodes of a fine cell
      // already contains the hanging-node interpolation and constraints of
      // the fine level, because the coarse space is a subspace.
      VectorizedArray<Number> cell_result[dofs_per_cell];
      operator_fine.template compute_cell_operation_nodal<size_fine - 1>(cell, src, cell_result);
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        tmp[i] -= cell_result[i];
    }
    dealii::internal::FEEvaluationImplBasisChange<dealii::internal::evaluate_general,
                                                  dealii::internal::EvaluatorQuantity::value,
                                                  dim,
                                                  size_coarse,
                                                  size_fine>::do_backward(1,
                                                                          prolongation_matrix_data,
                                                                          false,
                                                                          tmp,
                                                                          tmp);
    operator_coarse.template distribute_local_to_global<size_coarse - 1>(cell, tmp, dst);
  }
};



/**
 * This is a specialized transfer operator for doing anisotropic transfer,
 * implemented for nested meshes with all information embedded into the meshes
 * of operator_fine and operator_coarse. Compared to the deal.II class
 * MGTwoLevelTransferNonnested, this class aims to be more efficient by
 * computing a nested transfer.
 */
template<int dim, typename Number>
class MGTwoLevelTransferAnisotropicNested
  : public MGTwoLevelTransferBase<dim, LinearAlgebra::distributed::Vector<Number>>,
    public MGTwoLevelTransferResidualRestriction<dim, Number>
{
public:
  using VectorType                      = LinearAlgebra::distributed::Vector<Number>;
  static constexpr unsigned int n_lanes = VectorizedArray<Number>::size();

  MGTwoLevelTransferAnisotropicNested(
    const LaplaceOperator::LaplaceOperatorFE<dim, Number> & operator_fine,
    const LaplaceOperator::LaplaceOperatorFE<dim, Number> & operator_coarse,
    const bool                                              print_statistics = false)
    : operator_fine(operator_fine),
      operator_coarse(operator_coarse),
      print_timings(print_statistics),
      barrier_before_transfer(std::getenv("EXADG_TRANSFER_PROFILE_BARRIER") != nullptr)
  {
    // find unique indices of prolongation to identify the actual read
    // pattern; on cells with hanging nodes, only the DoFs actually owned by
    // the cell are considered, because the indices of the coarse side of
    // hanging-node constraints do not belong to the position within the cell
    fine_dof_indices = operator_fine.get_compressed_dof_indices_without_hanging_nodes();
    const unsigned int degree_fine =
      operator_fine.get_matrix_free().get_dof_handler().get_fe().degree;
    const unsigned int n_owned_dofs =
      operator_fine.get_matrix_free().get_dof_handler().locally_owned_dofs().n_elements();
    std::vector<char> touched_dof(n_owned_dofs, 0);
    for(unsigned int & dof : fine_dof_indices)
      if(dof >= n_owned_dofs)
        dof = numbers::invalid_unsigned_int;
      else if(touched_dof[dof] == 0)
        touched_dof[dof] = 1;
      else
        dof = numbers::invalid_unsigned_int;

    all_indices_unconstrained.resize(fine_dof_indices.size() / n_lanes, 0);

    const DoFHandler<dim> & dof_handler_coarse =
      operator_coarse.get_matrix_free().get_dof_handler();
    const MPI_Comm communicator = dof_handler_coarse.get_mpi_communicator();

    // Identify cells on coarser level for data import
    const Triangulation<dim> & tria_coarse(dof_handler_coarse.get_triangulation());

    // Step 1: For each cell, identify the cell on the coarse level where a
    // cell from the fine level is contained in. We do this by a geometric
    // search, whose result is the global index of cells queried from the
    // process owing those cells, and the respective reference coordinates,
    // which we use for the computation of the interpolation matrix.
    const auto &            mf      = operator_coarse.get_matrix_free();
    const auto &            mf_fine = operator_fine.get_matrix_free();
    std::vector<Point<dim>> cell_centers;
    cell_centers.reserve(mf_fine.n_cell_batches() * n_lanes);
    for(unsigned int c = 0; c < mf_fine.n_cell_batches(); ++c)
      for(unsigned int v = 0; v < mf_fine.n_active_entries_per_cell_batch(c); ++v)
        cell_centers.push_back(mf_fine.get_cell_iterator(c, v)->center());

    Utilities::MPI::RemotePointEvaluation<dim> point_eval_cache;
    point_eval_cache.reinit(cell_centers, tria_coarse, MappingQ1<dim>());
    using data_pair = std::pair<types::global_cell_index, Point<dim>>;
    std::vector<data_pair> indices_of_needed_cells, buffer;
    const auto             evaluation_function =
      [&](const ArrayView<data_pair> &                                          values,
          const typename Utilities::MPI::RemotePointEvaluation<dim>::CellData & cell_data) {
        for(unsigned int i = 0; i < cell_data.cells.size(); ++i)
        {
          typename Triangulation<dim>::active_cell_iterator cell = {&tria_coarse,
                                                                    cell_data.cells[i].first,
                                                                    cell_data.cells[i].second};
          for(unsigned int q = cell_data.reference_point_ptrs[i];
              q < cell_data.reference_point_ptrs[i + 1];
              ++q)
          {
            values[q] =
              std::make_pair(cell->global_active_cell_index(), cell_data.reference_point_values[q]);
          }
        }
      };

    point_eval_cache.template evaluate_and_process<data_pair>(indices_of_needed_cells,
                                                              buffer,
                                                              evaluation_function);

    // Collect all requested cells in the form of an IndexSet, which we use
    // for communicating the solution values on the coarse side to the owner
    // on the refined side, where the interpolation process is performed
    IndexSet owned_cells =
      tria_coarse.global_active_cell_index_partitioner().lock()->locally_owned_range();
    IndexSet requested_cells(tria_coarse.n_global_active_cells());
    for(const data_pair & item : indices_of_needed_cells)
      requested_cells.add_index(item.first);

    const unsigned int dofs_per_cell = mf.get_dof_handler().get_fe().dofs_per_cell;
    partitioner_coarse_cells =
      std::make_shared<Utilities::MPI::Partitioner>(owned_cells, requested_cells, communicator);
    setup_communication(dofs_per_cell);

    std::vector<unsigned int> active_cell_index_to_index(
      partitioner_coarse_cells->locally_owned_size());
    for(unsigned int cell = 0; cell < mf.n_cell_batches(); ++cell)
      for(unsigned int v = 0; v < mf.n_active_entries_per_cell_batch(cell); ++v)
      {
        const auto dcell                      = mf.get_cell_iterator(cell, v);
        active_cell_index_to_index[partitioner_coarse_cells->global_to_local(
          dcell->global_active_cell_index())] = cell * n_lanes + v;
      }

    unsigned int count = 0;
    cell_indices_to_send.resize(partitioner_coarse_cells->n_import_indices());
    for(const auto & import_range : partitioner_coarse_cells->import_indices())
      for(unsigned int j = import_range.first; j < import_range.second; ++j, ++count)
        cell_indices_to_send[count] = active_cell_index_to_index[j];
    AssertDimension(count, cell_indices_to_send.size());

    cell_indices_to_be_read.clear();
    cell_indices_to_be_read.resize(mf_fine.n_cell_batches() * n_lanes,
                                   numbers::invalid_unsigned_int);

    count                                   = 0;
    unsigned int n_fine_cells_remote_coarse = 0;
    interpolation_index_of_cells.resize(mf_fine.n_cell_batches());
    const unsigned int size_prol = (dof_handler_coarse.get_fe().degree + 1) * (degree_fine + 1);
    // the first fine cell (in the order of the restriction loop) contributing
    // to a coarse cell of another process sets the values in the receive
    // buffer, the others add to them, which avoids zeroing the buffer
    static_assert(n_lanes <= 32, "Lane mask must fit into 32 bits");
    first_contribution_lanes.assign(mf_fine.n_cell_batches(), 0);
    std::vector<bool> ghost_cell_touched(partitioner_coarse_cells->n_ghost_indices(), false);
    for(unsigned int c = 0; c < mf_fine.n_cell_batches(); ++c)
      for(unsigned int v = 0; v < mf_fine.n_active_entries_per_cell_batch(c); ++v, ++count)
      {
        const auto & item        = indices_of_needed_cells[count];
        unsigned int local_index = partitioner_coarse_cells->global_to_local(item.first);
        if(local_index < partitioner_coarse_cells->locally_owned_size())
        {
          cell_indices_to_be_read[c * n_lanes + v] = active_cell_index_to_index[local_index];
          // local and remote cells are distinguished by comparing with the
          // number of locally owned cells
          AssertThrow(active_cell_index_to_index[local_index] <
                        partitioner_coarse_cells->locally_owned_size(),
                      ExcMessage("Cell batches of the coarse level contain padded lanes"));
        }
        else
        {
          cell_indices_to_be_read[c * n_lanes + v] = local_index;
          ++n_fine_cells_remote_coarse;
          const unsigned int ghost = local_index - partitioner_coarse_cells->locally_owned_size();
          if(!ghost_cell_touched[ghost])
          {
            ghost_cell_touched[ghost] = true;
            first_contribution_lanes[c] |= 1u << v;
          }
        }
        const Point<dim> ref_point = item.second;
        for(unsigned int d = 0; d < dim; ++d)
        {
          if(std::abs(ref_point[d] - 0.25) < 1e-10)
            interpolation_index_of_cells[c][d][v] = 1 * size_prol;
          else if(std::abs(ref_point[d] - 0.75) < 1e-10)
            interpolation_index_of_cells[c][d][v] = 2 * size_prol;
          else if(std::abs(ref_point[d] - 0.5) < 1e-10)
            interpolation_index_of_cells[c][d][v] = 0;
          else
            AssertThrow(false,
                        ExcMessage("Could not detect coarsening pattern, "
                                   "got position " +
                                   std::to_string(ref_point[d]) + ", expected 0.25, 0.5 or 0.75"));
        }
      }

    if(print_statistics)
      print_ownership_statistics(count, n_fine_cells_remote_coarse, dofs_per_cell);

    std::vector<Polynomials::Polynomial<double>> poly =
      Polynomials::generate_complete_Lagrange_basis(
        QGaussLobatto<1>(dof_handler_coarse.get_fe().degree + 1).get_points());
    std::vector<Point<1>> points_fine(QGaussLobatto<1>(degree_fine + 1).get_points());
    prolongation_matrix_data.resize(3 * size_prol);
    for(unsigned int i = 0; i < poly.size(); ++i)
      for(unsigned int j = 0; j < degree_fine + 1; ++j)
      {
        prolongation_matrix_data[i * (degree_fine + 1) + j] = static_cast<Number>(i == j);
        prolongation_matrix_data[size_prol + i * (degree_fine + 1) + j] =
          poly[i].value(0.5 * points_fine[j][0]);
        prolongation_matrix_data[2 * size_prol + i * (degree_fine + 1) + j] =
          poly[i].value(0.5 + 0.5 * points_fine[j][0]);
      }
  }

  ~MGTwoLevelTransferAnisotropicNested() override
  {
    if(print_timings)
      print_phase_timings();
    for(std::vector<MPI_Request> * requests : {&prolongation_done_send,
                                               &prolongation_done_recv,
                                               &restriction_done_send,
                                               &restriction_done_recv})
      wait_for_signals(*requests);
    if(comm_done != MPI_COMM_NULL)
    {
      const int ierr = MPI_Comm_free(&comm_done);
      AssertThrowMPI(ierr);
    }
  }

  /**
   * @copydoc MGTwoLevelTransferBase::prolongate_and_add
   */
  void
  prolongate_and_add(VectorType & dst, const VectorType & src) const override
  {
    // Retrieve solution from coarse space into a separate field that gets
    // communicated (cells not present in the local range of the solution
    // vector) or by reading directly on the locally owned cells
    PhaseTimer<7> timer(time_prolongation);
    ++n_prolongations;
    if(barrier_before_transfer)
    {
      const int ierr = MPI_Barrier(src.get_mpi_communicator());
      AssertThrowMPI(ierr);
    }
    timer.lap(0);
    src.update_ghost_values();
    timer.lap(1);
    const unsigned int dofs_per_cell =
      operator_coarse.get_matrix_free().get_dof_handler().get_fe().dofs_per_cell;

    // the partners on this node must have finished reading the cells packed
    // in the last prolongation before the send buffer is overwritten
    wait_for_signals(prolongation_done_recv);
    wait_for_signals(prolongation_done_send);
    timer.lap(2);

    // partners on the same node only exchange an empty message signaling that
    // the send buffer is ready, and read the data in place
    std::vector<MPI_Request> requests(partitioner_coarse_cells->import_targets().size() +
                                      partitioner_coarse_cells->ghost_targets().size());
    unsigned int             offset = 0, count_request = 0;
    for(unsigned int t = 0; t < partitioner_coarse_cells->ghost_targets().size(); ++t)
    {
      const auto & target = partitioner_coarse_cells->ghost_targets()[t];
      const auto   ierr =
        MPI_Irecv(recv_buffer + offset,
                  ghost_target_shared[t] ? 0 : target.second * dofs_per_cell * sizeof(Number),
                  MPI_BYTE,
                  target.first,
                  15,
                  src.get_mpi_communicator(),
                  &requests[count_request]);
      AssertThrowMPI(ierr);
      offset += target.second * dofs_per_cell;
      ++count_request;
    }

    unsigned int count_cell = 0;
    offset                  = 0;
    for(unsigned int t = 0; t < partitioner_coarse_cells->import_targets().size(); ++t)
    {
      const auto & target = partitioner_coarse_cells->import_targets()[t];
      for(unsigned int i = 0; i < target.second; ++i, ++count_cell)
        operator_coarse.read_dof_values_compressed(src,
                                                   cell_indices_to_send[count_cell] / n_lanes,
                                                   cell_indices_to_send[count_cell] % n_lanes,
                                                   send_buffer + offset + i * dofs_per_cell);
      std::atomic_thread_fence(std::memory_order_release);
      const auto ierr =
        MPI_Isend(send_buffer + offset,
                  import_target_shared[t] ? 0 : target.second * dofs_per_cell * sizeof(Number),
                  MPI_BYTE,
                  target.first,
                  15,
                  src.get_mpi_communicator(),
                  &requests[count_request]);
      AssertThrowMPI(ierr);
      offset += target.second * dofs_per_cell;
      ++count_request;
    }
    AssertDimension(offset, send_buffer_size);
    AssertDimension(count_request, requests.size());

    // receive the signals that the partners on this node are done reading
    post_signals(partitioner_coarse_cells->import_targets(),
                 import_target_shared,
                 tag_prolongation_done,
                 false,
                 prolongation_done_recv);
    timer.lap(3);

    if(requests.size() > 0)
    {
      const int ierr = MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
      AssertThrowMPI(ierr);
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    timer.lap(4);

    const unsigned int size_fine =
      operator_fine.get_matrix_free().get_dof_handler().get_fe().degree + 1;
    const unsigned int n_cell_batches =
      fine_dof_indices.size() / n_lanes / Utilities::pow(std::min(3u, size_fine), dim);
    for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
    {
      if(size_fine == 2)
        do_prolongate_and_add_on_cell<2>(cell, dst, src);
      else if(size_fine == 3)
        do_prolongate_and_add_on_cell<3>(cell, dst, src);
      else if(size_fine == 4)
        do_prolongate_and_add_on_cell<4>(cell, dst, src);
      else if(size_fine == 5)
        do_prolongate_and_add_on_cell<5>(cell, dst, src);
      else if(size_fine == 6)
        do_prolongate_and_add_on_cell<6>(cell, dst, src);
      else if(size_fine == 7)
        do_prolongate_and_add_on_cell<7>(cell, dst, src);
      else if(size_fine == 8)
        do_prolongate_and_add_on_cell<8>(cell, dst, src);
      else if(size_fine == 9)
        do_prolongate_and_add_on_cell<9>(cell, dst, src);
      else
        AssertThrow(false,
                    ExcMessage("Fine degree " + std::to_string(size_fine - 1) +
                               " not instantiated"));
    }

    timer.lap(5);

    // signal to the partners on the same node that their send buffer may be
    // overwritten
    post_signals(partitioner_coarse_cells->ghost_targets(),
                 ghost_target_shared,
                 tag_prolongation_done,
                 true,
                 prolongation_done_send);

    src.zero_out_ghost_values();
    timer.lap(6);
  }

  /**
   * @copydoc MGTwoLevelTransferBase::restrict_and_add
   */
  void
  restrict_and_add(VectorType & dst, const VectorType & src) const override
  {
    restrict_cells<false>(dst, src, nullptr);
  }

  /**
   * @copydoc MGTwoLevelTransferResidualRestriction::restrict_residual
   */
  void
  restrict_residual(VectorType &       dst,
                    const VectorType & rhs,
                    const VectorType & solution) const override
  {
    dst                   = 0;
    const bool has_ghosts = solution.has_ghost_elements();
    if(!has_ghosts)
      solution.update_ghost_values();
    restrict_cells<true>(dst, solution, &rhs);
    if(!has_ghosts)
      solution.zero_out_ghost_values();
  }

  const LaplaceOperatorFE<dim, Number> &
  get_operator_fine() const override
  {
    return operator_fine;
  }

private:
  // Restriction of src (residual == false) or of the residual *rhs - A src
  // (residual == true), summed into dst
  template<bool residual>
  void
  restrict_cells(VectorType & dst, const VectorType & src, const VectorType * rhs) const
  {
    const unsigned int size_fine =
      operator_fine.get_matrix_free().get_dof_handler().get_fe().degree + 1;
    const unsigned int n_cell_batches = fine_dof_indices.size() / n_lanes /
                                        Utilities::pow(std::min<unsigned int>(3u, size_fine), dim);
    PhaseTimer<6> timer(time_restriction);
    ++n_restrictions;
    if(barrier_before_transfer)
    {
      const int ierr = MPI_Barrier(dst.get_mpi_communicator());
      AssertThrowMPI(ierr);
    }
    timer.lap(0);

    // the partners on the same node must have finished reading the contributions
    // of the last restriction before the buffer is overwritten
    wait_for_signals(restriction_done_recv);
    wait_for_signals(restriction_done_send);
    timer.lap(1);
    for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
    {
      if(size_fine == 2)
        do_restrict_and_add_on_cell<2, residual>(cell, dst, src, rhs);
      else if(size_fine == 3)
        do_restrict_and_add_on_cell<3, residual>(cell, dst, src, rhs);
      else if(size_fine == 4)
        do_restrict_and_add_on_cell<4, residual>(cell, dst, src, rhs);
      else if(size_fine == 5)
        do_restrict_and_add_on_cell<5, residual>(cell, dst, src, rhs);
      else if(size_fine == 6)
        do_restrict_and_add_on_cell<6, residual>(cell, dst, src, rhs);
      else if(size_fine == 7)
        do_restrict_and_add_on_cell<7, residual>(cell, dst, src, rhs);
      else if(size_fine == 8)
        do_restrict_and_add_on_cell<8, residual>(cell, dst, src, rhs);
      else if(size_fine == 9)
        do_restrict_and_add_on_cell<9, residual>(cell, dst, src, rhs);
      else
        AssertThrow(false,
                    ExcMessage("Fine degree " + std::to_string(size_fine - 1) +
                               " not instantiated"));
    }

    timer.lap(2);
    const unsigned int dofs_per_cell =
      operator_coarse.get_matrix_free().get_dof_handler().get_fe().dofs_per_cell;

    std::vector<MPI_Request> requests(partitioner_coarse_cells->import_targets().size() +
                                      partitioner_coarse_cells->ghost_targets().size());
    unsigned int             offset = 0, count_request = 0;
    for(unsigned int t = 0; t < partitioner_coarse_cells->import_targets().size(); ++t)
    {
      const auto & target = partitioner_coarse_cells->import_targets()[t];
      const auto   ierr =
        MPI_Irecv(send_buffer + offset,
                  import_target_shared[t] ? 0 : target.second * dofs_per_cell * sizeof(Number),
                  MPI_BYTE,
                  target.first,
                  15,
                  dst.get_mpi_communicator(),
                  &requests[count_request]);
      AssertThrowMPI(ierr);
      offset += target.second * dofs_per_cell;
      ++count_request;
    }

    std::atomic_thread_fence(std::memory_order_release);
    offset = 0;
    for(unsigned int t = 0; t < partitioner_coarse_cells->ghost_targets().size(); ++t)
    {
      const auto & target = partitioner_coarse_cells->ghost_targets()[t];
      const auto   ierr =
        MPI_Isend(recv_buffer + offset,
                  ghost_target_shared[t] ? 0 : target.second * dofs_per_cell * sizeof(Number),
                  MPI_BYTE,
                  target.first,
                  15,
                  dst.get_mpi_communicator(),
                  &requests[count_request]);
      AssertThrowMPI(ierr);
      offset += target.second * dofs_per_cell;
      ++count_request;
    }
    AssertDimension(count_request, requests.size());

    // receive the signals that the partners on this node are done reading
    post_signals(partitioner_coarse_cells->ghost_targets(),
                 ghost_target_shared,
                 tag_restriction_done,
                 false,
                 restriction_done_recv);

    if(requests.size() > 0)
    {
      const int ierr = MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
      AssertThrowMPI(ierr);
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    timer.lap(3);
    for(unsigned int i = 0; i < partitioner_coarse_cells->n_import_indices(); ++i)
      operator_coarse.distribute_local_to_global_compressed(dst,
                                                            cell_indices_to_send[i] / n_lanes,
                                                            cell_indices_to_send[i] % n_lanes,
                                                            import_cell_contributions[i]);

    // signal to the partners on this node that their buffer may be
    // overwritten
    post_signals(partitioner_coarse_cells->import_targets(),
                 import_target_shared,
                 tag_restriction_done,
                 true,
                 restriction_done_send);
    timer.lap(4);

    dst.compress(VectorOperation::add);
    timer.lap(5);
  }

public:
  void
  interpolate(VectorType &, const VectorType &) const override
  {
    AssertThrow(false, ExcNotImplemented());
  }

  std::pair<bool, bool>
  enable_inplace_operations_if_possible(
    const std::shared_ptr<const Utilities::MPI::Partitioner> &,
    const std::shared_ptr<const Utilities::MPI::Partitioner> &) override
  {
    return std::make_pair(true, true);
  }

  std::size_t
  memory_consumption() const override
  {
    return MemoryConsumption::memory_consumption(fine_dof_indices) +
           MemoryConsumption::memory_consumption(all_indices_unconstrained) +
           MemoryConsumption::memory_consumption(prolongation_matrix_data) +
           MemoryConsumption::memory_consumption(cell_indices_to_be_read) +
           MemoryConsumption::memory_consumption(cell_indices_to_send) +
           MemoryConsumption::memory_consumption(interpolation_index_of_cells);
  }

  std::pair<const DoFHandler<dim> *, unsigned int>
  get_dof_handler_fine() const override
  {
    return std::make_pair(&operator_fine.get_matrix_free().get_dof_handler(),
                          numbers::invalid_unsigned_int);
  }

private:
  /**
   * Print how many of the coarse cells needed by the locally owned fine cells
   * are owned by other processes, and which part of that data has to cross
   * node boundaries. This measures how well the partitions of the fine and
   * coarse triangulations are aligned. Both prolongation and restriction move
   * the same amount of data (in opposite directions).
   */
  void
  print_ownership_statistics(const unsigned int n_fine_cells,
                             const unsigned int n_fine_cells_remote_coarse,
                             const unsigned int dofs_per_cell) const
  {
    const Utilities::MPI::Partitioner & part    = *partitioner_coarse_cells;
    const MPI_Comm                      comm    = part.get_mpi_communicator();
    const unsigned int                  my_rank = Utilities::MPI::this_mpi_process(comm);

    // identify the node of each process by the rank of the first process in
    // its shared-memory communicator
    MPI_Comm comm_shared;
    int      ierr =
      MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, my_rank, MPI_INFO_NULL, &comm_shared);
    AssertThrowMPI(ierr);
    unsigned int node_id = my_rank;
    ierr                 = MPI_Bcast(&node_id, 1, MPI_UNSIGNED, 0, comm_shared);
    AssertThrowMPI(ierr);
    ierr = MPI_Comm_free(&comm_shared);
    AssertThrowMPI(ierr);
    // emulate a layout with several nodes on a single machine
    if(const char * env = std::getenv("EXADG_RANKS_PER_NODE"))
    {
      const unsigned int ranks_per_node = std::max(1, std::atoi(env));
      node_id                           = my_rank / ranks_per_node * ranks_per_node;
    }
    const std::vector<unsigned int> node_of_rank = Utilities::MPI::all_gather(comm, node_id);
    const std::size_t               n_nodes =
      std::set<unsigned int>(node_of_rank.begin(), node_of_rank.end()).size();

    unsigned int cells_on_node = 0, cells_off_node = 0, partners_off_node = 0;
    for(const auto & [rank, n_cells] : part.ghost_targets())
      if(node_of_rank[rank] == node_id)
        cells_on_node += n_cells;
      else
      {
        cells_off_node += n_cells;
        ++partners_off_node;
      }

    // Count the distinct vertices, lines, faces and cells among the coarse
    // cells sent to each target. With k the degree, a message containing each
    // shared DoF only once has V + (k-1) E + (k-1)^2 F + (k-1)^3 C entries,
    // compared to C (k+1)^3 for the current per-cell messages.
    std::array<std::uint64_t, 4> entities_on_node{}, entities_off_node{};
    if constexpr(dim == 3)
    {
      const auto & mf_coarse  = operator_coarse.get_matrix_free();
      unsigned int count_cell = 0;
      for(const auto & [rank, n_cells] : part.import_targets())
      {
        std::set<unsigned int> vertices, lines, faces;
        for(unsigned int i = 0; i < n_cells; ++i, ++count_cell)
        {
          const auto cell = mf_coarse.get_cell_iterator(cell_indices_to_send[count_cell] / n_lanes,
                                                        cell_indices_to_send[count_cell] % n_lanes);
          for(const unsigned int v : cell->vertex_indices())
            vertices.insert(cell->vertex_index(v));
          for(const unsigned int l : cell->line_indices())
            lines.insert(cell->line(l)->index());
          for(const unsigned int f : cell->face_indices())
            faces.insert(cell->face(f)->index());
        }
        auto & entities = (node_of_rank[rank] == node_id) ? entities_on_node : entities_off_node;
        entities[0] += vertices.size();
        entities[1] += lines.size();
        entities[2] += faces.size();
        entities[3] += n_cells;
      }
      for(auto & e : entities_on_node)
        e = Utilities::MPI::sum(e, comm);
      for(auto & e : entities_off_node)
        e = Utilities::MPI::sum(e, comm);
    }
    const auto unique_fraction = [](const std::array<std::uint64_t, 4> & e, const unsigned int k) {
      const double km = k - 1.;
      return 100. * (e[0] + km * e[1] + km * km * e[2] + km * km * km * e[3]) /
             std::max(1., e[3] * Utilities::fixed_power<3>(k + 1.));
    };
    const unsigned int degree_coarse =
      operator_coarse.get_matrix_free().get_dof_handler().get_fe().degree;
    const auto kb_ghosts_vector = Utilities::MPI::min_max_avg(
      operator_coarse.get_matrix_free().get_dof_info().vector_partitioner->n_ghost_indices() *
        sizeof(Number) / 1024.,
      comm);

    // Load of the coarse level if each coarse cell was owned by the owner of
    // its fine children (aligned partition): every fine cell contributes the
    // inverse of the number of children of its coarse cell. Compare the
    // maximum with the average per process and per node.
    double aligned_load = 0;
    for(unsigned int c = 0; c < interpolation_index_of_cells.size(); ++c)
      for(unsigned int v = 0;
          v < operator_fine.get_matrix_free().n_active_entries_per_cell_batch(c);
          ++v)
      {
        unsigned int n_children = 1;
        for(unsigned int d = 0; d < dim; ++d)
          if(interpolation_index_of_cells[c][d][v] != 0)
            n_children *= 2;
        aligned_load += 1. / n_children;
      }
    const auto                     aligned_rank = Utilities::MPI::min_max_avg(aligned_load, comm);
    const std::vector<double>      aligned_all  = Utilities::MPI::all_gather(comm, aligned_load);
    std::map<unsigned int, double> aligned_node;
    for(unsigned int r = 0; r < aligned_all.size(); ++r)
      aligned_node[node_of_rank[r]] += aligned_all[r];
    double aligned_node_max = 0, aligned_node_sum = 0;
    for(const auto & [node, load] : aligned_node)
    {
      aligned_node_max = std::max(aligned_node_max, load);
      aligned_node_sum += load;
    }
    const double aligned_node_avg =
      aligned_node_sum / std::max<std::size_t>(1, aligned_node.size());

    const auto sum = [&](const unsigned int value) {
      return Utilities::MPI::sum<std::uint64_t>(value, comm);
    };
    const std::uint64_t n_fine_global          = sum(n_fine_cells);
    const std::uint64_t n_fine_remote_global   = sum(n_fine_cells_remote_coarse);
    const std::uint64_t n_coarse_global        = sum(part.locally_owned_size());
    const std::uint64_t n_received_on_global   = sum(cells_on_node);
    const std::uint64_t n_received_off_global  = sum(cells_off_node);
    const std::uint64_t n_received_cell_global = n_received_on_global + n_received_off_global;

    const double kb_per_cell = dofs_per_cell * sizeof(Number) / 1024.;
    const auto   kb_received =
      Utilities::MPI::min_max_avg(part.n_ghost_indices() * kb_per_cell, comm);
    const auto kb_off_node  = Utilities::MPI::min_max_avg(cells_off_node * kb_per_cell, comm);
    const auto partners     = Utilities::MPI::min_max_avg(part.ghost_targets().size(), comm);
    const auto partners_off = Utilities::MPI::min_max_avg(partners_off_node, comm);

    if(my_rank == 0)
    {
      const auto percent = [](const std::uint64_t value, const std::uint64_t whole) {
        return 100. * value / std::max<std::uint64_t>(whole, 1);
      };
      std::cout << std::defaultfloat << std::setprecision(4) << "Anisotropic transfer, "
                << n_fine_global << " fine cells -> " << n_coarse_global << " coarse cells on "
                << Utilities::MPI::n_mpi_processes(comm) << " processes, " << n_nodes
                << " nodes:" << std::endl
                << "  fine cells whose coarse cell is on another process: " << n_fine_remote_global
                << " (" << percent(n_fine_remote_global, n_fine_global) << "%)" << std::endl
                << "  coarse cells moved per transfer:                   " << n_received_cell_global
                << " (" << percent(n_received_cell_global, n_coarse_global)
                << "% of all coarse cells), off-node "
                << percent(n_received_off_global, n_received_cell_global) << "%" << std::endl
                << "  KB received per process and transfer:   min " << kb_received.min << " avg "
                << kb_received.avg << " max " << kb_received.max << std::endl
                << "  KB received from other nodes:           min " << kb_off_node.min << " avg "
                << kb_off_node.avg << " max " << kb_off_node.max << std::endl
                << "  unique DoFs vs. per-cell entries, degree " << degree_coarse << ": "
                << unique_fraction(entities_on_node, degree_coarse) << "% on-node, "
                << unique_fraction(entities_off_node, degree_coarse)
                << "% off-node; degree 7: " << unique_fraction(entities_on_node, 7) << "% / "
                << unique_fraction(entities_off_node, 7) << "%" << std::endl
                << "  coarse cells per process if aligned:    min " << aligned_rank.min << " avg "
                << aligned_rank.avg << " max " << aligned_rank.max << " (max/avg "
                << aligned_rank.max / aligned_rank.avg << "), per node max/avg "
                << aligned_node_max / aligned_node_avg << std::endl
                << "  KB of ghost update of the coarse vector: avg " << kb_ghosts_vector.avg
                << " max " << kb_ghosts_vector.max << std::endl
                << "  partner processes (all / off-node):     avg " << partners.avg << " / "
                << partners_off.avg << ", max " << partners.max << " / " << partners_off.max
                << std::endl;
    }
  }

  const LaplaceOperator::LaplaceOperatorFE<dim, Number> & operator_fine;
  const LaplaceOperator::LaplaceOperatorFE<dim, Number> & operator_coarse;

  // Accumulated wall time of the phases of prolongation and restriction,
  // printed at destruction; with EXADG_TRANSFER_PROFILE_BARRIER, a barrier
  // before each transfer separates the waiting for late processes (phase 0)
  // from the transfer itself
  const bool                     print_timings;
  const bool                     barrier_before_transfer;
  mutable std::array<double, 7>  time_prolongation = {};
  mutable std::array<double, 6>  time_restriction  = {};
  mutable unsigned long long int n_prolongations   = 0;
  mutable unsigned long long int n_restrictions    = 0;

  // Adds the time since the last lap (or construction) to the given phase
  template<std::size_t n_phases>
  class PhaseTimer
  {
  public:
    PhaseTimer(std::array<double, n_phases> & times)
      : times(times), last(std::chrono::steady_clock::now())
    {
    }

    void
    lap(const unsigned int phase)
    {
      AssertIndexRange(phase, n_phases);
      const auto now = std::chrono::steady_clock::now();
      times[phase] += std::chrono::duration<double>(now - last).count();
      last = now;
    }

  private:
    std::array<double, n_phases> &                     times;
    std::chrono::time_point<std::chrono::steady_clock> last;
  };

  void
  print_phase_timings() const
  {
    const MPI_Comm comm = partitioner_coarse_cells->get_mpi_communicator();
    const auto     print =
      [&](const auto & times, const auto & names, const unsigned long long int n_calls) {
        double total = 0;
        for(const double t : times)
          total += t;
        const auto total_stat = Utilities::MPI::min_max_avg(total, comm);
        if(Utilities::MPI::this_mpi_process(comm) == 0)
          std::cout << "  " << names[0] << " (" << n_calls
                    << " calls), microseconds per call, avg / max over processes:" << std::endl;
        for(unsigned int i = 0; i < times.size(); ++i)
        {
          const auto stat = Utilities::MPI::min_max_avg(times[i], comm);
          if(Utilities::MPI::this_mpi_process(comm) == 0)
            std::cout << "    " << std::left << std::setw(36) << names[i + 1] << std::right
                      << std::fixed << std::setprecision(1) << std::setw(9)
                      << 1e6 * stat.avg / std::max(1ull, n_calls) << std::setw(9)
                      << 1e6 * stat.max / std::max(1ull, n_calls) << std::defaultfloat << std::endl;
        }
        if(Utilities::MPI::this_mpi_process(comm) == 0)
          std::cout << "    " << std::left << std::setw(36) << "total" << std::right << std::fixed
                    << std::setprecision(1) << std::setw(9)
                    << 1e6 * total_stat.avg / std::max(1ull, n_calls) << std::setw(9)
                    << 1e6 * total_stat.max / std::max(1ull, n_calls) << std::defaultfloat
                    << std::endl;
      };
    if(Utilities::MPI::this_mpi_process(comm) == 0)
      std::cout << "Phases of anisotropic transfer, "
                << operator_fine.get_matrix_free().get_dof_handler().n_dofs() << " fine DoFs -> "
                << operator_coarse.get_matrix_free().get_dof_handler().n_dofs()
                << " coarse DoFs:" << std::endl;
    print(time_prolongation,
          std::array<std::string, 8>{{"prolongation",
                                      "barrier (waiting for late processes)",
                                      "ghost update coarse vector",
                                      "wait for done signals",
                                      "pack and post messages",
                                      "wait for messages",
                                      "cell loop",
                                      "signal done, zero ghosts"}},
          n_prolongations);
    print(time_restriction,
          std::array<std::string, 7>{{"restriction",
                                      "barrier (waiting for late processes)",
                                      "wait for done signals",
                                      "cell loop",
                                      "post and wait for messages",
                                      "add received contributions",
                                      "compress coarse vector"}},
          n_restrictions);
    if(Utilities::MPI::this_mpi_process(comm) == 0)
      std::cout << std::endl;
  }
  std::vector<unsigned int>                                fine_dof_indices;
  std::vector<unsigned char>                               all_indices_unconstrained;
  AlignedVector<Number>                                    prolongation_matrix_data;
  std::shared_ptr<const Utilities::MPI::Partitioner>       partitioner_coarse_cells;
  std::vector<unsigned int>                                cell_indices_to_be_read;
  std::vector<std::uint32_t>                               first_contribution_lanes;
  std::vector<unsigned int>                                cell_indices_to_send;
  std::vector<dealii::ndarray<std::uint8_t, dim, n_lanes>> interpolation_index_of_cells;

  // Communication buffers, both with one section of whole cells per partner
  // process in the order of the partitioner's targets:
  // - send buffer: coarse cells owned here, packed for the partners in the
  //   prolongation, and the contributions to them received in the
  //   restriction;
  // - receive buffer: coarse cells of the partners received in the
  //   prolongation, and the contributions to them computed here in the
  //   restriction.
  // With shared memory, both buffers are allocated in shared windows, and the
  // partners on the same node read the sections for them in place instead
  // of receiving messages.
  Number *                                                                send_buffer = nullptr;
  Number *                                                                recv_buffer = nullptr;
  unsigned int                                                            send_buffer_size = 0;
  unsigned int                                                            recv_buffer_size = 0;
  std::vector<Number>                                                     send_storage;
  std::vector<Number>                                                     recv_storage;
  dealii::MemorySpace::MemorySpaceData<Number, dealii::MemorySpace::Host> send_storage_sm;
  dealii::MemorySpace::MemorySpaceData<Number, dealii::MemorySpace::Host> recv_storage_sm;

  // per ghost coarse cell: location of its values in the prolongation, in the
  // receive buffer or in the send buffer of the owner on the same node
  std::vector<const Number *> ghost_cell_values;

  // per imported coarse cell: location of the contributions to it in the
  // restriction, in the send buffer or in the receive buffer of the partner
  // on the same node
  std::vector<const Number *> import_cell_contributions;

  // per ghost resp. import target: whether the partner is on the same node
  std::vector<bool> ghost_target_shared;
  std::vector<bool> import_target_shared;

  // Signals that a partner on the same node has finished reading a buffer of
  // this process, posted after reading and waited for before the buffer is
  // overwritten in the next call. They are sent over a separate communicator
  // because they are still in flight when the transfer returns. The send
  // requests are kept per target to be completed before their slots are
  // reused.
  MPI_Comm                         comm_done = MPI_COMM_NULL;
  mutable std::vector<MPI_Request> prolongation_done_send;
  mutable std::vector<MPI_Request> prolongation_done_recv;
  mutable std::vector<MPI_Request> restriction_done_send;
  mutable std::vector<MPI_Request> restriction_done_recv;

  static constexpr int tag_offset_send       = 1;
  static constexpr int tag_offset_recv       = 4;
  static constexpr int tag_prolongation_done = 2;
  static constexpr int tag_restriction_done  = 3;

  /**
   * Allocate the communication buffers and set up the access to the data of
   * the partners. Shared memory is used unless the environment variable
   * EXADG_TRANSFER_NO_SHARED_MEMORY is set. Collective over the communicator
   * of the coarse cells.
   */
  void
  setup_communication(const unsigned int dofs_per_cell)
  {
    const Utilities::MPI::Partitioner & part = *partitioner_coarse_cells;
    const MPI_Comm                      comm = part.get_mpi_communicator();
    send_buffer_size                         = part.n_import_indices() * dofs_per_cell;
    recv_buffer_size                         = part.n_ghost_indices() * dofs_per_cell;
    ghost_target_shared.assign(part.ghost_targets().size(), false);
    import_target_shared.assign(part.import_targets().size(), false);
    ghost_cell_values.resize(part.n_ghost_indices());
    import_cell_contributions.resize(part.n_import_indices());

    MPI_Comm comm_sm = MPI_COMM_NULL;
    int      ierr;
    if(std::getenv("EXADG_TRANSFER_NO_SHARED_MEMORY") == nullptr)
    {
      ierr = MPI_Comm_split_type(comm,
                                 MPI_COMM_TYPE_SHARED,
                                 Utilities::MPI::this_mpi_process(comm),
                                 MPI_INFO_NULL,
                                 &comm_sm);
      AssertThrowMPI(ierr);

      // emulate a layout with several nodes on a single machine, for testing
      // the combination of shared memory and messages
      if(const char * env = std::getenv("EXADG_RANKS_PER_NODE"))
      {
        const unsigned int rank           = Utilities::MPI::this_mpi_process(comm);
        const unsigned int ranks_per_node = std::max(1, std::atoi(env));
        MPI_Comm           comm_node;
        ierr = MPI_Comm_split(comm_sm, rank / ranks_per_node, rank, &comm_node);
        AssertThrowMPI(ierr);
        ierr = MPI_Comm_free(&comm_sm);
        AssertThrowMPI(ierr);
        comm_sm = comm_node;
      }
    }

    if(comm_sm == MPI_COMM_NULL)
    {
      send_storage.resize(send_buffer_size);
      recv_storage.resize(recv_buffer_size);
      send_buffer = send_storage.data();
      recv_buffer = recv_storage.data();
      for(unsigned int c = 0; c < part.n_ghost_indices(); ++c)
        ghost_cell_values[c] = recv_buffer + c * dofs_per_cell;
      for(unsigned int c = 0; c < part.n_import_indices(); ++c)
        import_cell_contributions[c] = send_buffer + c * dofs_per_cell;
      return;
    }

    mpi_shared_memory::allocate_shared_recv_data(send_storage_sm, send_buffer_size, comm_sm);
    mpi_shared_memory::allocate_shared_recv_data(recv_storage_sm, recv_buffer_size, comm_sm);
    send_buffer = send_storage_sm.values.data();
    recv_buffer = recv_storage_sm.values.data();

    // rank of the partners in the shared-memory communicator, or
    // MPI_UNDEFINED for partners on other nodes
    const auto translate_ranks = [&](const auto & targets) {
      std::vector<int> ranks(targets.size()), ranks_sm(targets.size());
      for(unsigned int t = 0; t < targets.size(); ++t)
        ranks[t] = targets[t].first;
      MPI_Group group, group_sm;
      int       ierr = MPI_Comm_group(comm, &group);
      AssertThrowMPI(ierr);
      ierr = MPI_Comm_group(comm_sm, &group_sm);
      AssertThrowMPI(ierr);
      ierr =
        MPI_Group_translate_ranks(group, ranks.size(), ranks.data(), group_sm, ranks_sm.data());
      AssertThrowMPI(ierr);
      MPI_Group_free(&group);
      MPI_Group_free(&group_sm);
      return ranks_sm;
    };
    const std::vector<int> ghost_ranks_sm  = translate_ranks(part.ghost_targets());
    const std::vector<int> import_ranks_sm = translate_ranks(part.import_targets());

    ierr = MPI_Comm_dup(comm, &comm_done);
    AssertThrowMPI(ierr);

    // offsets of the sections of the targets in the two buffers
    const auto compute_offsets = [&](const auto & targets) {
      std::vector<unsigned int> offsets(targets.size());
      for(unsigned int t = 0, offset = 0; t < targets.size(); ++t)
      {
        offsets[t] = offset;
        offset += targets[t].second * dofs_per_cell;
      }
      return offsets;
    };
    std::vector<unsigned int> ghost_offsets  = compute_offsets(part.ghost_targets());
    std::vector<unsigned int> import_offsets = compute_offsets(part.import_targets());

    // Exchange the offsets of the sections with the partners on the same
    // node: the owner of the coarse cells sends the offset in its send
    // buffer, the other side the offset in its receive buffer.
    std::vector<unsigned int> remote_ghost_offsets(ghost_offsets.size());
    std::vector<unsigned int> remote_import_offsets(import_offsets.size());
    std::vector<MPI_Request>  requests;
    requests.reserve(2 * (ghost_offsets.size() + import_offsets.size()));
    const auto exchange_offset = [&](const unsigned int rank,
                                     unsigned int &     local_offset,
                                     unsigned int &     remote_offset,
                                     const int          tag_local,
                                     const int          tag_remote) {
      requests.emplace_back();
      int ierr =
        MPI_Irecv(&remote_offset, 1, MPI_UNSIGNED, rank, tag_remote, comm_done, &requests.back());
      AssertThrowMPI(ierr);
      requests.emplace_back();
      ierr =
        MPI_Isend(&local_offset, 1, MPI_UNSIGNED, rank, tag_local, comm_done, &requests.back());
      AssertThrowMPI(ierr);
    };
    // two partners may need cells of each other, so the offsets in the send
    // and receive buffers are distinguished by the tag
    for(unsigned int t = 0; t < ghost_offsets.size(); ++t)
      if(ghost_ranks_sm[t] != MPI_UNDEFINED)
      {
        ghost_target_shared[t] = true;
        exchange_offset(part.ghost_targets()[t].first,
                        ghost_offsets[t],
                        remote_ghost_offsets[t],
                        tag_offset_recv,
                        tag_offset_send);
      }
    for(unsigned int t = 0; t < import_offsets.size(); ++t)
      if(import_ranks_sm[t] != MPI_UNDEFINED)
      {
        import_target_shared[t] = true;
        exchange_offset(part.import_targets()[t].first,
                        import_offsets[t],
                        remote_import_offsets[t],
                        tag_offset_send,
                        tag_offset_recv);
      }
    ierr = MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    AssertThrowMPI(ierr);

    for(unsigned int t = 0, c = 0; t < ghost_offsets.size(); ++t)
    {
      const Number * base =
        ghost_target_shared[t] ?
          send_storage_sm.values_sm[ghost_ranks_sm[t]].data() + remote_ghost_offsets[t] :
          recv_buffer + ghost_offsets[t];
      for(unsigned int i = 0; i < part.ghost_targets()[t].second; ++i, ++c)
        ghost_cell_values[c] = base + i * dofs_per_cell;
    }
    for(unsigned int t = 0, c = 0; t < import_offsets.size(); ++t)
    {
      const Number * base =
        import_target_shared[t] ?
          recv_storage_sm.values_sm[import_ranks_sm[t]].data() + remote_import_offsets[t] :
          send_buffer + import_offsets[t];
      for(unsigned int i = 0; i < part.import_targets()[t].second; ++i, ++c)
        import_cell_contributions[c] = base + i * dofs_per_cell;
    }

    prolongation_done_send.assign(ghost_offsets.size(), MPI_REQUEST_NULL);
    restriction_done_recv.assign(ghost_offsets.size(), MPI_REQUEST_NULL);
    prolongation_done_recv.assign(import_offsets.size(), MPI_REQUEST_NULL);
    restriction_done_send.assign(import_offsets.size(), MPI_REQUEST_NULL);

    ierr = MPI_Comm_free(&comm_sm);
    AssertThrowMPI(ierr);
  }

  // Post empty messages over 'comm_done' to (send == true) or from
  // (send == false) the targets on the same node, one request per target.
  void
  post_signals(const std::vector<std::pair<unsigned int, unsigned int>> & targets,
               const std::vector<bool> &                                  target_shared,
               const int                                                  tag,
               const bool                                                 send,
               std::vector<MPI_Request> &                                 requests) const
  {
    if(requests.empty())
      return;
    if(send)
      std::atomic_thread_fence(std::memory_order_release);
    for(unsigned int t = 0; t < targets.size(); ++t)
      if(target_shared[t])
      {
        const int ierr =
          send ? MPI_Isend(nullptr, 0, MPI_BYTE, targets[t].first, tag, comm_done, &requests[t]) :
                 MPI_Irecv(nullptr, 0, MPI_BYTE, targets[t].first, tag, comm_done, &requests[t]);
        AssertThrowMPI(ierr);
      }
  }

  // Complete the given signal requests, a no-op for null requests
  void
  wait_for_signals(std::vector<MPI_Request> & requests) const
  {
    if(requests.empty())
      return;
    const int ierr = MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    AssertThrowMPI(ierr);
    std::atomic_thread_fence(std::memory_order_acquire);
  }

  template<int size_fine>
  void
  do_prolongate_and_add_on_cell(const unsigned int cell,
                                VectorType &       dst,
                                const VectorType & src) const
  {
    constexpr int size_coarse = size_fine;
    AssertDimension(prolongation_matrix_data.size(), size_coarse * size_fine * 3);
    AssertDimension(prolongation_matrix_data.size(), size_coarse * size_fine * 3);
    constexpr unsigned int  dofs_per_cell = Utilities::pow(size_fine, dim);
    VectorizedArray<Number> tmp[dofs_per_cell];
    Number                  tmp2[dofs_per_cell];
    VectorizedArray<Number> interpolation_matrix[size_coarse * size_fine];
    const unsigned int      locally_owned_size = partitioner_coarse_cells->locally_owned_size();

    for(unsigned int v = 0; v < n_lanes; ++v)
      if(cell_indices_to_be_read[cell * n_lanes + v] < locally_owned_size)
      {
        operator_coarse.template read_dof_values_compressed<size_coarse - 1>(
          src,
          cell_indices_to_be_read[cell * n_lanes + v] / n_lanes,
          cell_indices_to_be_read[cell * n_lanes + v] % n_lanes,
          tmp2);
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          tmp[i][v] = tmp2[i];
      }
      else if(cell_indices_to_be_read[cell * n_lanes + v] != numbers::invalid_unsigned_int)
      {
        const Number * ptr =
          ghost_cell_values[cell_indices_to_be_read[cell * n_lanes + v] - locally_owned_size];
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          tmp[i][v] = ptr[i];
      }
      else
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          tmp[i][v] = 0;

    std::array<unsigned int, n_lanes> indices_interpolation;
    dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_general,
                                             dim,
                                             size_coarse,
                                             size_fine,
                                             VectorizedArray<Number>>
      evaluator;
    for(unsigned int v = 0; v < n_lanes; ++v)
      indices_interpolation[v] = interpolation_index_of_cells[cell][0][v];
    vectorized_load_and_transpose(size_coarse * size_fine,
                                  prolongation_matrix_data.data(),
                                  indices_interpolation.data(),
                                  interpolation_matrix);
    evaluator.template apply<0, true, false>(interpolation_matrix, tmp, tmp);
    if constexpr(dim > 1)
    {
      for(unsigned int v = 0; v < n_lanes; ++v)
        indices_interpolation[v] = interpolation_index_of_cells[cell][1][v];
      vectorized_load_and_transpose(size_coarse * size_fine,
                                    prolongation_matrix_data.data(),
                                    indices_interpolation.data(),
                                    interpolation_matrix);
      evaluator.template apply<1, true, false>(interpolation_matrix, tmp, tmp);
    }
    if constexpr(dim > 2)
    {
      for(unsigned int v = 0; v < n_lanes; ++v)
        indices_interpolation[v] = interpolation_index_of_cells[cell][2][v];
      vectorized_load_and_transpose(size_coarse * size_fine,
                                    prolongation_matrix_data.data(),
                                    indices_interpolation.data(),
                                    interpolation_matrix);
      evaluator.template apply<2, true, false>(interpolation_matrix, tmp, tmp);
    }

    if constexpr(size_fine == 2)
    {
      // degree 1 stores the 2^dim vertex indices of the cell
      const unsigned int * indices = fine_dof_indices.data() + cell * dofs_per_cell * n_lanes;
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        for(unsigned int v = 0; v < n_lanes; ++v)
          if(indices[i * n_lanes + v] != numbers::invalid_unsigned_int)
            dst.local_element(indices[i * n_lanes + v]) += tmp[i][v];
    }
    else
      operator_fine.template distribute_local_to_global_compressed<size_fine - 1, size_fine, 1>(
        dst, fine_dof_indices, all_indices_unconstrained, cell, {}, true, tmp);
  }

  template<int size_fine, bool residual>
  void
  do_restrict_and_add_on_cell(const unsigned int cell,
                              VectorType &       dst,
                              const VectorType & src,
                              const VectorType * rhs) const
  {
    constexpr int size_coarse = size_fine;
    AssertDimension(prolongation_matrix_data.size(), size_coarse * size_fine * 3);
    AssertDimension(prolongation_matrix_data.size(), size_coarse * size_fine * 3);
    constexpr unsigned int  dofs_per_cell = Utilities::pow(size_fine, dim);
    VectorizedArray<Number> tmp[dofs_per_cell];
    Number                  tmp2[dofs_per_cell];
    VectorizedArray<Number> interpolation_matrix[size_coarse * size_fine];

    if constexpr(size_fine == 2)
    {
      // degree 1 stores the 2^dim vertex indices of the cell
      const unsigned int * indices = fine_dof_indices.data() + cell * dofs_per_cell * n_lanes;
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        for(unsigned int v = 0; v < n_lanes; ++v)
          tmp[i][v] = indices[i * n_lanes + v] != numbers::invalid_unsigned_int ?
                        (residual ? *rhs : src).local_element(indices[i * n_lanes + v]) :
                        Number(0);
    }
    else
      operator_fine.template read_dof_values_compressed<size_fine - 1, size_fine, 1>(
        residual ? *rhs : src, fine_dof_indices, all_indices_unconstrained, cell, {}, true, tmp);
    if constexpr(residual)
    {
      // The operator is applied without the transpose of the hanging-node
      // interpolation and the cell values are restricted at all nodes, see
      // MGTwoLevelTransferFromOperator
      VectorizedArray<Number> cell_result[dofs_per_cell];
      operator_fine.template compute_cell_operation_nodal<size_fine - 1>(cell, src, cell_result);
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        tmp[i] -= cell_result[i];
    }
    std::array<unsigned int, n_lanes> indices_interpolation;
    dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_general,
                                             dim,
                                             size_coarse,
                                             size_fine,
                                             VectorizedArray<Number>>
      evaluator;
    for(unsigned int v = 0; v < n_lanes; ++v)
      indices_interpolation[v] = interpolation_index_of_cells[cell][0][v];
    vectorized_load_and_transpose(size_coarse * size_fine,
                                  prolongation_matrix_data.data(),
                                  indices_interpolation.data(),
                                  interpolation_matrix);
    evaluator.template apply<0, false, false>(interpolation_matrix, tmp, tmp);
    if constexpr(dim > 1)
    {
      for(unsigned int v = 0; v < n_lanes; ++v)
        indices_interpolation[v] = interpolation_index_of_cells[cell][1][v];
      vectorized_load_and_transpose(size_coarse * size_fine,
                                    prolongation_matrix_data.data(),
                                    indices_interpolation.data(),
                                    interpolation_matrix);
      evaluator.template apply<1, false, false>(interpolation_matrix, tmp, tmp);
    }
    if constexpr(dim > 2)
    {
      for(unsigned int v = 0; v < n_lanes; ++v)
        indices_interpolation[v] = interpolation_index_of_cells[cell][2][v];
      vectorized_load_and_transpose(size_coarse * size_fine,
                                    prolongation_matrix_data.data(),
                                    indices_interpolation.data(),
                                    interpolation_matrix);
      evaluator.template apply<2, false, false>(interpolation_matrix, tmp, tmp);
    }

    const unsigned int locally_owned_size = partitioner_coarse_cells->locally_owned_size();
    for(unsigned int v = 0; v < n_lanes; ++v)
      if(cell_indices_to_be_read[cell * n_lanes + v] < locally_owned_size)
      {
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          tmp2[i] = tmp[i][v];
        operator_coarse.template distribute_local_to_global_compressed<size_coarse - 1>(
          dst,
          cell_indices_to_be_read[cell * n_lanes + v] / n_lanes,
          cell_indices_to_be_read[cell * n_lanes + v] % n_lanes,
          tmp2);
      }
      else if(cell_indices_to_be_read[cell * n_lanes + v] != numbers::invalid_unsigned_int)
      {
        Number * ptr =
          recv_buffer +
          (cell_indices_to_be_read[cell * n_lanes + v] - locally_owned_size) * dofs_per_cell;
        if(first_contribution_lanes[cell] & (1u << v))
          for(unsigned int i = 0; i < dofs_per_cell; ++i)
            ptr[i] = tmp[i][v];
        else
          for(unsigned int i = 0; i < dofs_per_cell; ++i)
            ptr[i] += tmp[i][v];
      }
  }
};



template<int dim, typename Number = float>
class PoissonPreconditionerMG
{
public:
  using VectorizedArrayType        = dealii::VectorizedArray<Number>;
  using MatrixType                 = LaplaceOperator::LaplaceOperatorFE<dim, Number>;
  using MatrixTypeDG               = LaplaceOperator::LaplaceOperatorDG<dim, Number>;
  using VectorType                 = dealii::LinearAlgebra::distributed::Vector<Number>;
  using SmootherPreconditionerType = dealii::DiagonalMatrix<VectorType>;
  using SmootherType               = ExaDG::PreconditionChebyshevScratch<MatrixType>;
  using SmootherTypeDG             = ExaDG::PreconditionChebyshevScratch<MatrixTypeDG>;
  using MGTransferType             = dealii::MGTransferMatrixFree<dim, Number>;

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
      is_test(is_test),
      factor_for_zero_mean_with_constraints_l0(1.0)
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

      reinit_level_constraints(dof_h,
                               level >= coarse_triangulations.size() ? cell_vectorization_category :
                                                                       std::vector<unsigned int>(),
                               level_constraints[level]);

      if(level < coarse_triangulations.size())
      {
        if(mapping_function)
        {
          dealii::MappingQCache<dim> mapping_coarse(1);
          mapping_coarse.initialize(dof_h.get_triangulation(), mapping_function);
          mg_matrices[level].reinit(mapping_coarse,
                                    dof_h,
                                    level_constraints[level],
                                    level + 1 < coarse_triangulations.size() ?
                                      std::vector<unsigned int>() :
                                      cell_vectorization_category,
                                    dealii::QGauss<1>(dof_h.get_fe().degree + 1));
        }
        else
        {
          dealii::MappingQ1<dim> mapping_coarse;
          mg_matrices[level].reinit(mapping_coarse,
                                    dof_h,
                                    level_constraints[level],
                                    level + 1 < coarse_triangulations.size() ?
                                      std::vector<unsigned int>() :
                                      cell_vectorization_category,
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
      if(level >= coarse_triangulations.size())
      {
        auto transfer =
          std::make_unique<MGTwoLevelTransferFromOperator<dim, Number>>(mg_matrices[level],
                                                                        mg_matrices[level - 1]);
        mg_transfers[level - 1] = std::move(transfer);
      }
      else if(level > 0)
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

    reinit_smoothers_and_dg(mapping_fine, dof_handler, ip_factor, cell_vectorization_category);
  }

  PoissonPreconditionerMG(
    const dealii::Mapping<dim> &                                   mapping_fine,
    const dealii::DoFHandler<dim> &                                dof_handler,
    const std::vector<unsigned int> &                              cell_vectorization_category,
    const std::vector<std::shared_ptr<const Triangulation<dim>>> & coarser_triangulations,
    const std::vector<std::shared_ptr<const Mapping<dim>>> &       coarser_mappings,
    const std::function<std::vector<dealii::Point<dim>>(
      typename dealii::Triangulation<dim>::cell_iterator const)> & mapping_function,
    const Number                                                   ip_factor,
    const bool                                                     is_test)
    : coarse_triangulations(
        dealii::
          MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(coarser_triangulations
                                                                                      .size() > 0 ?
                                                                                  *coarser_triangulations
                                                                                     .back() :
                                                                                  dof_handler
                                                                                    .get_triangulation() /*,
dealii::RepartitioningPolicyTools::MinimalGranularityPolicy<dim>(64)*/)),
      fe_hierarchy(create_fe_hierarchy(dof_handler.get_fe())),
      min_level(0),
      max_level(coarser_triangulations.size() + dof_handler.get_triangulation().n_global_levels() -
                1 + fe_hierarchy.max_level()),
      is_test(is_test),
      factor_for_zero_mean_with_constraints_l0(1.0)
  {
    dof_handler_hierarchy.resize(min_level, max_level);

    level_constraints.resize(min_level, max_level);
    mg_matrices.resize(min_level, max_level);
    mg_smoother.resize(min_level, max_level);
    mg_transfers.resize(min_level, max_level);
    rhs.resize(min_level, max_level);
    temp_vector.resize(min_level, max_level);
    solution_update.resize(min_level, max_level);

    unsigned int n_final_geometric_levels     = coarser_triangulations.size();
    int          index_coarser_triangulations = coarser_triangulations.size() - 1;
    while(index_coarser_triangulations > 0)
    {
      // the last index has already been added, so work on next level
      --index_coarser_triangulations;
      coarse_triangulations.push_back(coarser_triangulations[index_coarser_triangulations]);
    }
    if(n_final_geometric_levels > 0)
      coarse_triangulations.emplace_back(&dof_handler.get_triangulation(), [](auto *) {
        // empty deleter, since fine_triangulation_in is an external field
        // and its destructor is called somewhere else
      });

    // initialize levels
    for(unsigned int level = min_level; level <= max_level; level++)
    {
      dealii::AffineConstraints<Number> constraints;
      dealii::DoFHandler<dim> &         dof_h = dof_handler_hierarchy[level];
      dealii::MappingQ1<dim>            mapping_q1;
      dealii::MappingQCache<dim>        mapping_q1_cache(1);
      const dealii::Mapping<dim> *      mapping = nullptr;
      if(level < (coarser_triangulations.empty() ?
                    coarse_triangulations.size() :
                    coarser_triangulations.back()->n_global_levels()))
      {
        dof_h.reinit(*coarse_triangulations[level]);
        dof_h.distribute_dofs(*fe_hierarchy[0]);
        if(mapping_function)
        {
          mapping_q1_cache.initialize(dof_h.get_triangulation(), mapping_function);
          mapping = &mapping_q1_cache;
        }
        else
          mapping = &mapping_q1;
      }
      else if(level <= max_level - n_final_geometric_levels)
      {
        dof_h.reinit(
          *coarse_triangulations[coarse_triangulations.size() - 1 - n_final_geometric_levels]);
        dof_h.distribute_dofs(
          *fe_hierarchy[level - coarse_triangulations.size() + 1 + n_final_geometric_levels]);
        if(n_final_geometric_levels > 0)
          mapping = coarser_mappings.back().get();
        else
          mapping = &mapping_fine;
      }
      else
      {
        dof_h.reinit(*coarse_triangulations[level - fe_hierarchy.max_level()]);
        dof_h.distribute_dofs(*fe_hierarchy.back());
        if(level == max_level)
          mapping = &mapping_fine;
        else
          mapping = coarser_mappings[max_level - 1 - level].get();
      }

      std::vector<unsigned int> manual_vectorization_category(
        dof_h.get_triangulation().n_active_cells());
      if(level < max_level)
      {
        if(n_final_geometric_levels == 0 &&
           manual_vectorization_category.size() == cell_vectorization_category.size())
          manual_vectorization_category = cell_vectorization_category;
        else
        {
          unsigned int count = 0;
          for(const auto & cell : dof_h.active_cell_iterators())
            if(cell->is_locally_owned())
              manual_vectorization_category[cell->active_cell_index()] = count++;
        }
      }
      reinit_level_constraints(dof_h,
                               level < max_level ? manual_vectorization_category :
                                                   cell_vectorization_category,
                               level_constraints[level]);

      mg_matrices[level].reinit(*mapping,
                                dof_h,
                                level_constraints[level],
                                level < max_level ? manual_vectorization_category :
                                                    cell_vectorization_category,
                                dealii::QGauss<1>(dof_h.get_fe().degree + 1));

      // initialize transfer operator
      if(level > 0)
      {
        if(level <= max_level - n_final_geometric_levels - fe_hierarchy.max_level())
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
        else if(level <= max_level - n_final_geometric_levels)
        {
          auto transfer =
            std::make_unique<MGTwoLevelTransferFromOperator<dim, Number>>(mg_matrices[level],
                                                                          mg_matrices[level - 1]);
          mg_transfers[level - 1] = std::move(transfer);
        }
        else if(level > max_level - n_final_geometric_levels)
        {
          auto transfer = std::make_unique<MGTwoLevelTransferAnisotropicNested<dim, Number>>(
            mg_matrices[level], mg_matrices[level - 1], not is_test);
          // Compare the nested anisotropic transfer result with the output of
          // the (more general) non-nested transfer of deal.II to ensure
          // correctness of the result.
          bool constexpr perform_check = false;
          if constexpr(perform_check)
          {
            MGTwoLevelTransferNonNested<dim, VectorType> transfer_nonnested;
            MappingQ1<dim>                               mapping;
            transfer_nonnested.reinit(dof_handler_hierarchy[level],
                                      dof_handler_hierarchy[level - 1],
                                      mapping,
                                      mapping,
                                      level_constraints[level],
                                      level_constraints[level - 1]);
            VectorType v1, v2, v3, v4;
            mg_matrices[level - 1].initialize_dof_vector(v1);
            mg_matrices[level - 1].initialize_dof_vector(v4);
            mg_matrices[level].initialize_dof_vector(v2);
            mg_matrices[level].initialize_dof_vector(v3);
            Tensor<1, dim> tens;
            tens[0] = 1;
            tens[1] = 1;
            tens[2] = 1;
            VectorTools::interpolate(mapping,
                                     dof_handler_hierarchy[level - 1],
                                     Functions::Monomial<dim, Number>(tens, 1),
                                     v1);
            transfer->prolongate_and_add(v2, v1);
            transfer_nonnested.prolongate_and_add(v3, v1);
            std::cout << "Prolongate norms: " << v2.l2_norm() << " " << v3.l2_norm()
                      << " difference ";
            v2 -= v3;
            std::cout << v2.l2_norm() << std::endl;
            v1 = 0;
            v4 = 0;
            transfer->restrict_and_add(v1, v3);
            transfer_nonnested.restrict_and_add(v4, v3);
            std::cout << "Restrict norms: " << v1.l2_norm() << " " << v4.l2_norm()
                      << " difference ";
            v1 -= v4;
            std::cout << v1.l2_norm() << std::endl;
          }
          mg_transfers[level - 1] = std::move(transfer);
        }
      }
    }

    reinit_smoothers_and_dg(mapping_fine, dof_handler, ip_factor, cell_vectorization_category);
  }

  void
  reinit_level_constraints(DoFHandler<dim> &                 dof_h,
                           const std::vector<unsigned int> & cell_vectorization_category,
                           AffineConstraints<Number> &       level_constraints)
  {
    level_constraints.reinit(dof_h.locally_owned_dofs(),
                             dealii::DoFTools::extract_locally_relevant_dofs(dof_h));

    // hanging nodes need to be added before the periodicity constraints;
    // since the latter skip already constrained DoFs, hanging DoFs on
    // periodic faces stay hanging-node constrained on both sides
    const bool has_hanging_nodes = dof_h.get_triangulation().has_hanging_nodes();
    if(has_hanging_nodes)
      dealii::DoFTools::make_hanging_node_constraints(dof_h, level_constraints);

    dealii::ndarray<unsigned int, dim, 2> periodic_ids;
    for(unsigned int d = 0; d < dim; ++d)
      for(unsigned int e = 0; e < 2; ++e)
        periodic_ids[d][e] = numbers::invalid_unsigned_int;
    for(const auto & cell : dof_h.cell_iterators_on_level(0))
      for(unsigned int d = 0; d < dim; ++d)
        if(cell->at_boundary(2 * d) && cell->has_periodic_neighbor(2 * d))
        {
          periodic_ids[d][0] = cell->face(2 * d)->boundary_id();
          periodic_ids[d][1] = cell->periodic_neighbor(2 * d)
                                 ->face(cell->periodic_neighbor_face_no(2 * d))
                                 ->boundary_id();
        }
    std::vector<GridTools::PeriodicFacePair<typename DoFHandler<dim>::cell_iterator>>
      periodic_faces;
    for(unsigned int d = 0; d < dim; ++d)
      if(periodic_ids[d][0] != numbers::invalid_unsigned_int)
        GridTools::collect_periodic_faces(
          dof_h, periodic_ids[d][0], periodic_ids[d][1], d, periodic_faces);
    for(const auto & face_pair : periodic_faces)
      ExaDG::Periodicity::make_periodicity_constraints_recursively(
        face_pair.cell[0]->face(face_pair.face_idx[0]),
        face_pair.cell[1]->face(face_pair.face_idx[1]),
        level_constraints);
    level_constraints.close();

    typename dealii::MatrixFree<dim, Number>::AdditionalData mf_data;
    mf_data.cell_vectorization_category = cell_vectorization_category;

    // renumber Dofs to minimize the number of partitions in import
    // indices of partitioner
    dealii::DoFRenumbering::matrix_free_data_locality(dof_h, level_constraints, mf_data);
    level_constraints.reinit(dof_h.locally_owned_dofs(),
                             dealii::DoFTools::extract_locally_relevant_dofs(dof_h));
    if(has_hanging_nodes)
      dealii::DoFTools::make_hanging_node_constraints(dof_h, level_constraints);
    for(const auto & face_pair : periodic_faces)
      ExaDG::Periodicity::make_periodicity_constraints_recursively(
        face_pair.cell[0]->face(face_pair.face_idx[0]),
        face_pair.cell[1]->face(face_pair.face_idx[1]),
        level_constraints);

    level_constraints.close();
  }

  void
  reinit_smoothers_and_dg(const Mapping<dim> &              mapping_fine,
                          const DoFHandler<dim> &           dof_handler,
                          const Number                      ip_factor,
                          const std::vector<unsigned int> & cell_vectorization_category)
  {
    // initialize levels
    for(unsigned int level = min_level; level <= max_level; level++)
    {
      // ... initialize smoother
      typename SmootherType::AdditionalData smoother_data;
      smoother_data.preconditioner = std::make_shared<SmootherPreconditionerType>();
      mg_matrices[level].compute_inverse_diagonal(smoother_data.preconditioner->get_vector());

      // For 'float' numbers, round the diagonal to BF16 format to reduce
      // memory transfer in the smoother
      if constexpr(std::is_same_v<Number, float>)
        ExaDG::round_to_bf16(smoother_data.preconditioner->get_vector(),
                             smoother_data.preconditioner_bf16);

      smoother_data.smoothing_range = 20.;
      smoother_data.degree =
        mg_matrices[level].get_matrix_free().get_dof_handler().get_fe().degree > 1 ? 5 : 3;

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
      solver.solve(mg_matrices[level],
                   temp_vector[level],
                   rhs[level],
                   *smoother_data.preconditioner);

      if(eigenvalue_tracker.values.empty())
        smoother_data.max_eigenvalue = 1.0;
      else
        smoother_data.max_eigenvalue = 1.05 * eigenvalue_tracker.values.back();

      // For float, we use the bf16 representation
      if constexpr(std::is_same_v<Number, float>)
        smoother_data.preconditioner.reset();

      mg_smoother[level].initialize(mg_matrices[level], smoother_data);

      // Factor to rescale the mean value computed among all vector entries on
      // level 0 to the vector size without constraints. We precompute this
      // factor to avoid calling MPI::sum() in every V-cycle, where this
      // function is used.
      if(level == min_level)
      {
        const unsigned int n_unconstrained_dofs =
          rhs[level].locally_owned_size() -
          mg_matrices[level].get_matrix_free().get_constrained_dofs().size();
        factor_for_zero_mean_with_constraints_l0 =
          mg_matrices[level].m() /
          dealii::Utilities::MPI::sum(n_unconstrained_dofs, rhs[level].get_mpi_communicator());
      }
    }

    // create a different matrix on the finest level to enable the efficient
    // implementation of the DG discretization
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

      // For 'float' numbers, round the diagonal to BF16 format to reduce
      // memory transfer in the smoother
      if constexpr(std::is_same_v<Number, float>)
        ExaDG::round_to_bf16(smoother_data_dg.preconditioner->get_vector(),
                             smoother_data_dg.preconditioner_bf16);
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

      if(eigenvalue_tracker.values.empty())
        smoother_data_dg.max_eigenvalue = 1.0;
      else
        smoother_data_dg.max_eigenvalue = 1.05 * eigenvalue_tracker.values.back();

      // For float, we use the bf16 representation
      if constexpr(std::is_same_v<Number, float>)
        smoother_data_dg.preconditioner.reset();

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

    // record the size of each level for the statistics printed with the
    // timings, the last entry is the DG level
    const auto describe_level = [](const dealii::DoFHandler<dim> & dof_h) {
      const Triangulation<dim> &            tria = dof_h.get_triangulation();
      std::vector<types::global_cell_index> n_cells(tria.n_global_levels());
      for(const auto & cell : tria.active_cell_iterators())
        if(cell->is_locally_owned())
          ++n_cells[cell->level()];
      dealii::Utilities::MPI::sum(n_cells, tria.get_mpi_communicator(), n_cells);
      std::string cells;
      for(unsigned int l = 0; l < n_cells.size(); ++l)
        if(n_cells[l] > 0)
          cells +=
            (cells.empty() ? "L" : ", L") + std::to_string(l) + ": " + std::to_string(n_cells[l]);
      return LevelDescription{dof_h.get_fe().degree, dof_h.n_dofs(), cells};
    };
    level_descriptions.clear();
    for(unsigned int level = min_level; level <= max_level; ++level)
      level_descriptions.push_back(describe_level(dof_handler_hierarchy[level]));
    level_descriptions.push_back(describe_level(dof_handler));
  }

  ~PoissonPreconditionerMG()
  {
    if(not this->is_test)
    {
      // maximum over the processes of each entry, as the processes wait for
      // the slowest one in the communication of each level
      std::vector<double> local_times;
      for(const auto & array : timings)
        for(const auto time : array)
          local_times.push_back(time);
      std::vector<double> max_times(local_times.size());
      const int           ierr = MPI_Allreduce(local_times.data(),
                                     max_times.data(),
                                     local_times.size(),
                                     MPI_DOUBLE,
                                     MPI_MAX,
                                     MPI_COMM_WORLD);
      AssertThrowMPI(ierr);
      for(unsigned int i = 0, c = 0; i < timings.size(); ++i)
        for(unsigned int j = 0; j < timings[i].size(); ++j, ++c)
          timings[i][j] = max_times[c];
    }
    if(not this->is_test and Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      double all_time = 0;
      for(const auto & array : timings)
        for(const auto time : array)
          all_time += time;
      std::cout << "Collected multigrid timings in " << count_times
                << " evaluations [t_total=" << all_time << "s, max over processes]" << std::endl;
      std::size_t cells_width = 6;
      for(const auto & description : level_descriptions)
        cells_width = std::max(cells_width, description.cells.size() + 2);
      std::cout << "lvl  deg  dofs        " << std::left << std::setw(cells_width) << "cells"
                << "smooth      resid_restrict  prolongate   [share]" << std::endl;
      for(unsigned int i = 0; i < timings.size(); ++i)
      {
        const LevelDescription & description = level_descriptions[i];
        std::cout << std::left << std::setw(5) << i << std::setw(5)
                  << (std::to_string(description.degree) + (i + 1 < timings.size() ? "" : "DG"))
                  << std::setw(12) << description.n_dofs << std::setw(cells_width)
                  << description.cells << std::scientific << std::setw(12) << std::setprecision(3)
                  << timings[i][1] << std::setw(16) << timings[i][0] << std::setw(13)
                  << timings[i][3] << std::defaultfloat
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
      p_levels.push_back(get_coarser_fe_degree(p_levels.back()));
    dealii::MGLevelObject<std::unique_ptr<dealii::FE_Q<dim>>> fes(0, p_levels.size() - 1);
    for(unsigned int level = 0; level < p_levels.size(); ++level)
      fes[level] = std::make_unique<dealii::FE_Q<dim>>(p_levels[p_levels.size() - 1 - level]);
    return fes;
  }

  template<typename VectorTypeOuter>
  void
  vmult(VectorTypeOuter & dst, const VectorTypeOuter & src) const
  {
    if constexpr(std::is_same_v<VectorTypeOuter, VectorType>)
    {
      do_vcycle(true, src, dst);
    }
    else
    {
      Timer time;
      rhs_dg.copy_locally_owned_data_from(src);
      timings.back()[2] += time.wall_time();

      do_vcycle(true, rhs_dg, solution_update_dg);

      time.restart();
      dst.copy_locally_owned_data_from(solution_update_dg);
      timings.back()[2] += time.wall_time();
    }
  }

  template<typename VectorTypeOuter, typename MatrixTypeOuter>
  std::pair<unsigned int, double>
  solve(const MatrixTypeOuter & matrix,
        const VectorTypeOuter & rhs,
        VectorTypeOuter &       sol,
        const unsigned int      n_max_iterations,
        const double            target_residual) const
  {
    using Number2 = typename VectorTypeOuter::value_type;
    matrix.vmult_scratch(
      sol, [&](const unsigned int begin, const unsigned int end, const Number2 * matvec) {
        DEAL_II_OPENMP_SIMD_PRAGMA
        for(unsigned int i = begin; i < end; ++i)
        {
          rhs_dg.local_element(i)             = rhs.local_element(i) - matvec[i - begin];
          solution_update_dg.local_element(i) = 0.;
        }
      });

    dealii::IterationNumberControl control(n_max_iterations, target_residual);
    dealii::SolverCG<VectorType>   solver_cg(control);
    solver_cg.solve(dg_matrix, solution_update_dg, rhs_dg, *this);
    DEAL_II_OPENMP_SIMD_PRAGMA
    for(unsigned int i = 0; i < solution_update_dg.locally_owned_size(); ++i)
      sol.local_element(i) += solution_update_dg.local_element(i);

    return {control.last_step(), control.last_value()};
  }

  const MatrixTypeDG &
  get_dg_matrix() const
  {
    return dg_matrix;
  }

private:
  // Find the transfers that can compute the residual within the restriction,
  // unless disabled by the environment variable EXADG_NO_FUSED_RESIDUAL. If
  // EXADG_CHECK_FUSED_RESIDUAL is set, compare to the separate computation
  // on random vectors.
  void
  setup_residual_restriction() const
  {
    residual_restriction.assign(max_level + 1, nullptr);
    for(unsigned int level = min_level; level < max_level; ++level)
    {
      if(std::getenv("EXADG_NO_FUSED_RESIDUAL") != nullptr)
        continue;
      const auto * transfer =
        dynamic_cast<const MGTwoLevelTransferResidualRestriction<dim, Number> *>(
          mg_transfers[level].get());
      if(transfer == nullptr)
        continue;
      AssertThrow(&transfer->get_operator_fine() == &mg_matrices[level + 1],
                  ExcMessage("The fine operator of the transfer must be the level operator"));
      residual_restriction[level] = transfer;

      if(std::getenv("EXADG_CHECK_FUSED_RESIDUAL") != nullptr)
      {
        VectorType u, b, r, coarse_separate, coarse_fused;
        mg_matrices[level + 1].initialize_dof_vector(u);
        mg_matrices[level + 1].initialize_dof_vector(b);
        mg_matrices[level + 1].initialize_dof_vector(r);
        mg_matrices[level].initialize_dof_vector(coarse_separate);
        mg_matrices[level].initialize_dof_vector(coarse_fused);
        std::mt19937 gen(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));
        std::uniform_real_distribution<Number> dist(-1., 1.);
        for(unsigned int i = 0; i < u.locally_owned_size(); ++i)
        {
          u.local_element(i) = dist(gen);
          b.local_element(i) = dist(gen);
        }
        mg_matrices[level + 1].vmult(r, u);
        r.sadd(-1.0, 1.0, b);
        mg_transfers[level]->restrict_and_add(coarse_separate, r);
        transfer->restrict_residual(coarse_fused, b, u);
        const double norm = coarse_separate.l2_norm();
        coarse_fused -= coarse_separate;
        const double error = coarse_fused.l2_norm();
        const bool   has_hanging_nodes =
          Utilities::MPI::max(mg_matrices[level + 1].has_hanging_nodes() ? 1 : 0, MPI_COMM_WORLD) ==
          1;
        const types::global_dof_index n_constrained = Utilities::MPI::sum<types::global_dof_index>(
          mg_matrices[level + 1].get_matrix_free().get_constrained_dofs().size(), MPI_COMM_WORLD);
        if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          std::cout << "Fused residual restriction to level " << level << ": |separate| = " << norm
                    << ", |fused - separate| / |separate| = " << error / norm
                    << ", hanging nodes: " << (has_hanging_nodes ? "yes" : "no")
                    << ", constrained DoFs: " << n_constrained << std::endl;
      }
    }
  }

  void
  do_vcycle(const bool zero_out_solution, const VectorType & src_rhs, VectorType & solution) const
  {
    if(residual_restriction.empty())
      setup_residual_restriction();
    ++count_times;
    Timer time;
    if(zero_out_solution)
      mg_smoother_dg.vmult(solution, src_rhs);
    else
      mg_smoother_dg.step(solution, src_rhs);
    timings.back()[1] += time.wall_time();
    time.restart();

    dg_matrix.vmult_residual_and_restrict_to_fe(src_rhs,
                                                solution,
                                                mg_matrices[max_level],
                                                rhs[max_level]);
    timings.back()[0] += time.wall_time();
    time.restart();

    for(unsigned int level = max_level; level > min_level; --level)
    {
      mg_smoother[level].vmult(solution_update[level], rhs[level]);
      timings[level][1] += time.wall_time();
      time.restart();

      if(residual_restriction[level - 1] != nullptr)
      {
        // residual and restriction in one cell loop
        residual_restriction[level - 1]->restrict_residual(rhs[level - 1],
                                                           rhs[level],
                                                           solution_update[level]);
        timings[level][0] += time.wall_time();
        time.restart();
      }
      else
      {
        mg_matrices[level].vmult(temp_vector[level], solution_update[level]);
        temp_vector[level].sadd(-1.0, 1.0, rhs[level]);
        rhs[level - 1] = 0;
        mg_transfers[level - 1]->restrict_and_add(rhs[level - 1], temp_vector[level]);
        timings[level][0] += time.wall_time();
        time.restart();
      }
    }

    // coarse solver, taking into account zero mean
    make_zero_mean(factor_for_zero_mean_with_constraints_l0,
                   mg_matrices[min_level].get_matrix_free().get_constrained_dofs(),
                   rhs[min_level]);
    mg_smoother[min_level].vmult(solution_update[min_level], rhs[min_level]);
    make_zero_mean(factor_for_zero_mean_with_constraints_l0,
                   mg_matrices[min_level].get_matrix_free().get_constrained_dofs(),
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
    dg_matrix.prolongate_and_add(solution, solution_update[max_level], mg_matrices[max_level]);
    timings.back()[3] += time.wall_time();
    time.restart();

    mg_smoother_dg.step(solution, src_rhs);
    timings.back()[1] += time.wall_time();
  }

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
  Number                              factor_for_zero_mean_with_constraints_l0;

  dealii::MGLevelObject<std::unique_ptr<dealii::MGTwoLevelTransferBase<dim, VectorType>>>
    mg_transfers;

  // per level the transfer if it can compute the residual within the
  // restriction, else nullptr, set up in the first V-cycle
  mutable std::vector<const MGTwoLevelTransferResidualRestriction<dim, Number> *>
    residual_restriction;

  mutable dealii::MGLevelObject<VectorType> rhs;
  mutable dealii::MGLevelObject<VectorType> temp_vector;
  mutable dealii::MGLevelObject<VectorType> solution_update;

  mutable std::size_t count_times;
  // per level the time for residual and restriction [0], smoothing [1],
  // other operations such as vector copies on the DG level, only contained
  // in the total and the share [2], and prolongation [3]
  mutable std::vector<std::array<double, 4>> timings;

  // polynomial degree, number of DoFs, and number of active cells per
  // refinement level of the triangulation for each multigrid level
  struct LevelDescription
  {
    unsigned int                    degree;
    dealii::types::global_dof_index n_dofs;
    std::string                     cells;
  };
  std::vector<LevelDescription> level_descriptions;

  mutable VectorType rhs_dg;
  mutable VectorType solution_update_dg;
};

} // namespace LaplaceOperator
