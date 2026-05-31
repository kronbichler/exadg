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
#include <deal.II/fe/mapping_q_cache.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tensor_product_kernels.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
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



// Manually implement periodicity constraints, because the deal.II
// implementation places the constraints the 'wrong' way around, leading to a
// case with more unknowns on the processor with the lower rank and less on
// the higher ranks. This function constrains the unknown on the lower-left
// boundary by the one on the upper-right boundary. This is coherent with the
// strategy that DoFs on interfaces of two processes are assigned to the
// process with lower rank.
namespace Periodicity
{
template<typename FaceIterator, typename Number>
void
make_periodicity_constraints_recursively(const FaceIterator &                             face_1,
                                         const std_cxx20::type_identity_t<FaceIterator> & face_2,
                                         AffineConstraints<Number> & affine_constraints)
{
  if(face_1->has_children())
  {
    if(!face_2->has_children())
      return;
    for(unsigned int i = 0; i < face_1->n_children(); ++i)
      make_periodicity_constraints_recursively(face_1->child(i),
                                               face_2->child(i),
                                               affine_constraints);
  }
  else
  {
    const unsigned int                   dofs_per_face = face_1->get_fe(0).n_dofs_per_face(0);
    std::vector<types::global_dof_index> dofs_1(dofs_per_face);
    std::vector<types::global_dof_index> dofs_2(dofs_per_face);

    face_1->get_dof_indices(dofs_1, 0);
    face_2->get_dof_indices(dofs_2, 0);
    for(unsigned int i = 0; i < dofs_per_face; ++i)
      if(dofs_1[i] == numbers::invalid_dof_index || dofs_2[i] == numbers::invalid_dof_index)
        return;
    for(unsigned int i = 0; i < dofs_per_face; ++i)
    {
      if(!affine_constraints.is_constrained(dofs_1[i]))
        affine_constraints.add_constraint(dofs_1[i], {{dofs_2[i], 1.0}});
    }
  }
}

} // namespace Periodicity



template<int dim, typename Number>
class MGTwoLevelTransferFromOperator
  : public MGTwoLevelTransferBase<dim, LinearAlgebra::distributed::Vector<Number>>
{
public:
  using VectorType                      = LinearAlgebra::distributed::Vector<Number>;
  static constexpr unsigned int n_lanes = VectorizedArray<Number>::size();

  MGTwoLevelTransferFromOperator(const LaplaceOperatorFE<dim, Number> & operator_fine,
                                 const LaplaceOperatorFE<dim, Number> & operator_coarse)
    : operator_fine(operator_fine), operator_coarse(operator_coarse)
  {
    // find unique indices of prolongation to identify the actual read pattern
    fine_dof_indices = operator_fine.get_compressed_dof_indices();
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
            "Mismatch in cells " +
            operator_fine.get_matrix_free().get_cell_iterator(cell, v)->id().to_string() + " vs " +
            operator_coarse.get_matrix_free().get_cell_iterator(cell, v)->id().to_string() + " " +
            std::to_string(cell) + " " + std::to_string(v)));
      }
    }

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

    const unsigned int size_fine = prolongation_matrix.n();
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

    src.zero_out_ghost_values();
  }

  /**
   * @copydoc MGTwoLevelTransferBase::restrict_and_add
   */
  void
  restrict_and_add(VectorType & dst, const VectorType & src) const override
  {
    const unsigned int size_fine = prolongation_matrix.n();
    const unsigned int n_cell_batches =
      fine_dof_indices.size() / n_lanes /
      Utilities::pow(std::min<unsigned int>(3u, prolongation_matrix.n()), dim);
    for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
    {
      if(size_fine == 2)
        do_restrict_and_add_on_cell<2>(cell, dst, src);
      else if(size_fine == 3)
        do_restrict_and_add_on_cell<3>(cell, dst, src);
      else if(size_fine == 4)
        do_restrict_and_add_on_cell<4>(cell, dst, src);
      else if(size_fine == 5)
        do_restrict_and_add_on_cell<5>(cell, dst, src);
      else if(size_fine == 6)
        do_restrict_and_add_on_cell<6>(cell, dst, src);
      else if(size_fine == 7)
        do_restrict_and_add_on_cell<7>(cell, dst, src);
      else if(size_fine == 8)
        do_restrict_and_add_on_cell<8>(cell, dst, src);
      else if(size_fine == 9)
        do_restrict_and_add_on_cell<9>(cell, dst, src);
      else
        AssertThrow(false,
                    ExcMessage("Fine degree " + std::to_string(size_fine - 1) +
                               " not instantiated"));
    }

    dst.compress(VectorOperation::add);
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

  template<int size_fine>
  void
  do_restrict_and_add_on_cell(const unsigned int cell,
                              VectorType &       dst,
                              const VectorType & src) const
  {
    constexpr int size_coarse = get_coarser_fe_degree(size_fine - 1) + 1;
    AssertDimension(prolongation_matrix.m(), size_coarse);
    VectorizedArray<Number> tmp[Utilities::pow(size_fine, dim)];

    operator_fine.template read_dof_values_compressed<size_fine - 1, size_fine, 1>(
      src, fine_dof_indices, all_indices_unconstrained, cell, {}, true, tmp);
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



template<int dim, typename Number>
class MGTwoLevelTransferAnisotropicNested
  : public MGTwoLevelTransferBase<dim, LinearAlgebra::distributed::Vector<Number>>
{
public:
  using VectorType                      = LinearAlgebra::distributed::Vector<Number>;
  static constexpr unsigned int n_lanes = VectorizedArray<Number>::size();

  MGTwoLevelTransferAnisotropicNested(
    const LaplaceOperator::LaplaceOperatorFE<dim, Number> & operator_fine,
    const LaplaceOperator::LaplaceOperatorFE<dim, Number> & operator_coarse)
    : operator_fine(operator_fine), operator_coarse(operator_coarse)
  {
    // find unique indices of prolongation to identify the actual read pattern
    fine_dof_indices = operator_fine.get_compressed_dof_indices();
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
    data_sent.resize(partitioner_coarse_cells->n_import_indices() * dofs_per_cell);
    data_received.resize(partitioner_coarse_cells->n_ghost_indices() * dofs_per_cell);

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

    count = 0;
    interpolation_index_of_cells.resize(mf_fine.n_cell_batches());
    const unsigned int size_prol = (dof_handler_coarse.get_fe().degree + 1) * (degree_fine + 1);
    for(unsigned int c = 0; c < mf_fine.n_cell_batches(); ++c)
      for(unsigned int v = 0; v < mf_fine.n_active_entries_per_cell_batch(c); ++v, ++count)
      {
        const auto & item        = indices_of_needed_cells[count];
        unsigned int local_index = partitioner_coarse_cells->global_to_local(item.first);
        if(local_index < partitioner_coarse_cells->locally_owned_size())
          cell_indices_to_be_read[c * n_lanes + v] = active_cell_index_to_index[local_index];
        else
          cell_indices_to_be_read[c * n_lanes + v] = local_index;
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

  /**
   * @copydoc MGTwoLevelTransferBase::prolongate_and_add
   */
  void
  prolongate_and_add(VectorType & dst, const VectorType & src) const override
  {
    // Retrieve solution from coarse space into a separate field that gets
    // communicated (cells not present in the local range of the solution
    // vector) or by reading directly on the locally owned cells
    src.update_ghost_values();
    const unsigned int dofs_per_cell =
      operator_coarse.get_matrix_free().get_dof_handler().get_fe().dofs_per_cell;

    std::vector<MPI_Request> requests(partitioner_coarse_cells->import_targets().size() +
                                      partitioner_coarse_cells->ghost_targets().size());
    unsigned int             offset = 0, count_request = 0;
    for(const auto & target : partitioner_coarse_cells->ghost_targets())
    {
      const auto ierr = MPI_Irecv(data_received.data() + offset,
                                  target.second * dofs_per_cell * sizeof(Number),
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
    for(const auto & target : partitioner_coarse_cells->import_targets())
    {
      for(unsigned int i = 0; i < target.second; ++i, ++count_cell)
        operator_coarse.read_dof_values_compressed(src,
                                                   cell_indices_to_send[count_cell] / n_lanes,
                                                   cell_indices_to_send[count_cell] % n_lanes,
                                                   data_sent.data() + offset + i * dofs_per_cell);
      const auto ierr = MPI_Isend(data_sent.data() + offset,
                                  target.second * dofs_per_cell * sizeof(Number),
                                  MPI_BYTE,
                                  target.first,
                                  15,
                                  src.get_mpi_communicator(),
                                  &requests[count_request]);
      AssertThrowMPI(ierr);
      offset += target.second * dofs_per_cell;
      ++count_request;
    }
    AssertDimension(offset, data_sent.size());
    AssertDimension(count_request, requests.size());

    if(requests.size() > 0)
    {
      const int ierr = MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
      AssertThrowMPI(ierr);
    }

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

    src.zero_out_ghost_values();
  }

  /**
   * @copydoc MGTwoLevelTransferBase::restrict_and_add
   */
  void
  restrict_and_add(VectorType & dst, const VectorType & src) const override
  {
    const unsigned int size_fine =
      operator_fine.get_matrix_free().get_dof_handler().get_fe().degree + 1;
    const unsigned int n_cell_batches = fine_dof_indices.size() / n_lanes /
                                        Utilities::pow(std::min<unsigned int>(3u, size_fine), dim);
    std::fill(data_received.begin(), data_received.end(), Number(0));
    for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
    {
      if(size_fine == 2)
        do_restrict_and_add_on_cell<2>(cell, dst, src);
      else if(size_fine == 3)
        do_restrict_and_add_on_cell<3>(cell, dst, src);
      else if(size_fine == 4)
        do_restrict_and_add_on_cell<4>(cell, dst, src);
      else if(size_fine == 5)
        do_restrict_and_add_on_cell<5>(cell, dst, src);
      else if(size_fine == 6)
        do_restrict_and_add_on_cell<6>(cell, dst, src);
      else if(size_fine == 7)
        do_restrict_and_add_on_cell<7>(cell, dst, src);
      else if(size_fine == 8)
        do_restrict_and_add_on_cell<8>(cell, dst, src);
      else if(size_fine == 9)
        do_restrict_and_add_on_cell<9>(cell, dst, src);
      else
        AssertThrow(false,
                    ExcMessage("Fine degree " + std::to_string(size_fine - 1) +
                               " not instantiated"));
    }

    const unsigned int dofs_per_cell =
      operator_coarse.get_matrix_free().get_dof_handler().get_fe().dofs_per_cell;

    std::vector<MPI_Request> requests(partitioner_coarse_cells->import_targets().size() +
                                      partitioner_coarse_cells->ghost_targets().size());
    unsigned int             offset = 0, count_request = 0;
    for(const auto & target : partitioner_coarse_cells->import_targets())
    {
      const auto ierr = MPI_Irecv(data_sent.data() + offset,
                                  target.second * dofs_per_cell * sizeof(Number),
                                  MPI_BYTE,
                                  target.first,
                                  15,
                                  dst.get_mpi_communicator(),
                                  &requests[count_request]);
      AssertThrowMPI(ierr);
      offset += target.second * dofs_per_cell;
      ++count_request;
    }

    offset = 0;
    for(const auto & target : partitioner_coarse_cells->ghost_targets())
    {
      const auto ierr = MPI_Isend(data_received.data() + offset,
                                  target.second * dofs_per_cell * sizeof(Number),
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

    if(requests.size() > 0)
    {
      const int ierr = MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
      AssertThrowMPI(ierr);
    }
    for(unsigned int i = 0; i < partitioner_coarse_cells->n_import_indices(); ++i)
      operator_coarse.distribute_local_to_global_compressed(dst,
                                                            cell_indices_to_send[i] / n_lanes,
                                                            cell_indices_to_send[i] % n_lanes,
                                                            data_sent.data() + i * dofs_per_cell);

    dst.compress(VectorOperation::add);
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
  const LaplaceOperator::LaplaceOperatorFE<dim, Number> &  operator_fine;
  const LaplaceOperator::LaplaceOperatorFE<dim, Number> &  operator_coarse;
  std::vector<unsigned int>                                fine_dof_indices;
  std::vector<unsigned char>                               all_indices_unconstrained;
  AlignedVector<Number>                                    prolongation_matrix_data;
  std::shared_ptr<const Utilities::MPI::Partitioner>       partitioner_coarse_cells;
  std::vector<unsigned int>                                cell_indices_to_be_read;
  std::vector<unsigned int>                                cell_indices_to_send;
  std::vector<dealii::ndarray<std::uint8_t, dim, n_lanes>> interpolation_index_of_cells;
  mutable std::vector<Number>                              data_sent;
  mutable std::vector<Number>                              data_received;

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
          data_received.data() +
          (cell_indices_to_be_read[cell * n_lanes + v] - locally_owned_size) * dofs_per_cell;
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

    operator_fine.template distribute_local_to_global_compressed<size_fine - 1, size_fine, 1>(
      dst, fine_dof_indices, all_indices_unconstrained, cell, {}, true, tmp);
  }

  template<int size_fine>
  void
  do_restrict_and_add_on_cell(const unsigned int cell,
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

    operator_fine.template read_dof_values_compressed<size_fine - 1, size_fine, 1>(
      src, fine_dof_indices, all_indices_unconstrained, cell, {}, true, tmp);
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
          data_received.data() +
          (cell_indices_to_be_read[cell * n_lanes + v] - locally_owned_size) * dofs_per_cell;
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

      reinit_level_constraints(dof_h,
                               level + 1 >= coarse_triangulations.size() ?
                                 cell_vectorization_category :
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
      else if (level > 0)
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
      if(level < coarser_triangulations.back()->n_global_levels())
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

      reinit_level_constraints(dof_h,
                               level < max_level ? std::vector<unsigned int>() :
                                                   cell_vectorization_category,
                               level_constraints[level]);

      mg_matrices[level].reinit(*mapping,
                                dof_h,
                                level_constraints[level],
                                level < max_level ? std::vector<unsigned int>() :
                                                    cell_vectorization_category,
                                dealii::QGauss<1>(dof_h.get_fe().degree + 1));

      // initialize transfer operator
      if(level > 0 && level <= max_level - n_final_geometric_levels)
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
      else if(level > max_level - n_final_geometric_levels)
      {
        auto transfer = std::make_unique<MGTwoLevelTransferAnisotropicNested<dim, Number>>(
          mg_matrices[level], mg_matrices[level - 1]);
        /*
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
        std::cout << "Prolongate norms: " << v2.l2_norm() << " " << v3.l2_norm() << " difference ";
        v2 -= v3;
        std::cout << v2.l2_norm() << std::endl;
        v1 = 0;
        v4 = 0;
        transfer->restrict_and_add(v1, v3);
        transfer_nonnested.restrict_and_add(v4, v3);
        std::cout << "Restrict norms: " << v1.l2_norm() << " " << v4.l2_norm() << " difference ";
        v1 -= v4;
        std::cout << v1.l2_norm() << std::endl;
        */
        mg_transfers[level - 1] = std::move(transfer);
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
      Periodicity::make_periodicity_constraints_recursively(
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
    for(const auto & face_pair : periodic_faces)
      Periodicity::make_periodicity_constraints_recursively(
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
