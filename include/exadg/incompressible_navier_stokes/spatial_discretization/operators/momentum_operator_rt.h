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
#include <deal.II/base/memory_space_data.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tensor_product_kernels.h>

#include <cstdint>
#include <map>

#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/extruded_mapping_info.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/momentum_operator_rt_single_exchange.h>
#include <exadg/solvers_and_preconditioners/preconditioners/scratch_matrix_vector_product.h>
#include <exadg/utilities/mpi_shared_memory.h>

namespace RTOperator
{
template<int dim, int degree, typename Number>
void
distribute_local_to_global_rt_compressed(
  const dealii::VectorizedArray<Number> * local_array,
  const unsigned int                      n_filled_lanes,
  const dealii::ndarray<unsigned int, 2 * dim + 1, dealii::VectorizedArray<Number>::size()> &
           dof_indices,
  Number * dst_vector)
{
  constexpr unsigned int n_lanes       = dealii::VectorizedArray<Number>::size();
  constexpr unsigned int dofs_per_face = dealii::Utilities::pow(degree, dim - 1);
  constexpr unsigned int dofs_per_comp = dofs_per_face * (degree + 1);

  dealii::VectorizedArray<Number> data[dofs_per_face];
  for(unsigned int f = 0; f < 2; ++f)
  {
    for(unsigned int i = 0; i < dofs_per_face; ++i)
      data[i] = local_array[i * (degree + 1) + f * degree];

    // check if indices unconstrained
    bool all_indices_unconstrained = n_filled_lanes == n_lanes;
    for(const unsigned int i : dof_indices[f])
      if(i == dealii::numbers::invalid_unsigned_int)
      {
        all_indices_unconstrained = false;
        break;
      }
    if(all_indices_unconstrained)
      vectorized_transpose_and_store(true, dofs_per_face, data, dof_indices[f].data(), dst_vector);
    else
    {
      for(unsigned int v = 0; v < n_filled_lanes; ++v)
        if(dof_indices[f][v] != dealii::numbers::invalid_unsigned_int)
        {
          Number * dst_ptr = dst_vector + dof_indices[f][v];
          for(unsigned int i = 0; i < dofs_per_face; ++i)
            dst_ptr[i] += data[i][v];
        }
    }
  }
  if(dim == 3)
  {
    for(unsigned int f = 0; f < 2; ++f)
    {
      for(unsigned int i = 0; i < degree; ++i)
        for(unsigned int j = 0; j < degree; ++j)
          data[i * degree + j] =
            local_array[dofs_per_comp + i * (degree + 1) * degree + j + f * degree * degree];

      // check if indices unconstrained
      bool all_indices_unconstrained = n_filled_lanes == n_lanes;
      for(const unsigned int i : dof_indices[2 + f])
        if(i == dealii::numbers::invalid_unsigned_int)
        {
          all_indices_unconstrained = false;
          break;
        }
      if(all_indices_unconstrained)
        vectorized_transpose_and_store(
          true, dofs_per_face, data, dof_indices[2 + f].data(), dst_vector);
      else
      {
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          if(dof_indices[2 + f][v] != dealii::numbers::invalid_unsigned_int)
          {
            Number * dst_ptr = dst_vector + dof_indices[2 + f][v];
            for(unsigned int i = 0; i < dofs_per_face; ++i)
              dst_ptr[i] += data[i][v];
          }
      }
    }
  }

  for(unsigned int f = 0; f < 2; ++f)
  {
    const dealii::VectorizedArray<Number> * data =
      local_array + (dim - 1) * dofs_per_comp + f * degree * dofs_per_face;

    // check if indices unconstrained
    bool all_indices_unconstrained = n_filled_lanes == n_lanes;
    for(const unsigned int i : dof_indices[2 * dim - 2 + f])
      if(i == dealii::numbers::invalid_unsigned_int)
      {
        all_indices_unconstrained = false;
        break;
      }
    if(all_indices_unconstrained)
      vectorized_transpose_and_store(
        true, dofs_per_face, data, dof_indices[2 * dim - 2 + f].data(), dst_vector);
    else
    {
      for(unsigned int v = 0; v < n_filled_lanes; ++v)
        if(dof_indices[2 * dim - 2 + f][v] != dealii::numbers::invalid_unsigned_int)
        {
          Number * dst_ptr = dst_vector + dof_indices[2 * dim - 2 + f][v];
          for(unsigned int i = 0; i < dofs_per_face; ++i)
            dst_ptr[i] += data[i][v];
        }
    }
  }

  dealii::VectorizedArray<Number> data_2[dofs_per_comp - 2 * dofs_per_face];
  for(unsigned int i = 0; i < dofs_per_face; ++i)
    for(unsigned int j = 1; j < degree; ++j)
      data_2[i * (degree - 1) + j - 1] = local_array[i * (degree + 1) + j];

  if(n_filled_lanes == n_lanes)
    vectorized_transpose_and_store(
      true, dofs_per_comp - 2 * dofs_per_face, data_2, dof_indices[2 * dim].data(), dst_vector);
  else
    for(unsigned int v = 0; v < n_filled_lanes; ++v)
    {
      Number * dst_ptr = dst_vector + dof_indices[2 * dim][v];
      for(unsigned int i = 0; i < dofs_per_comp - 2 * dofs_per_face; ++i)
        dst_ptr[i] += data_2[i][v];
    }
  if(dim == 3)
    for(unsigned int i = 0; i < degree; ++i)
    {
      const dealii::VectorizedArray<Number> * data =
        local_array + dofs_per_comp + i * degree * (degree + 1) + degree;
      Number * dst_ptr = dst_vector + dofs_per_comp - 2 * dofs_per_face + i * (degree - 1) * degree;
      if(n_filled_lanes == n_lanes)
        vectorized_transpose_and_store(
          true, degree * (degree - 1), data, dof_indices[2 * dim].data(), dst_ptr);
      else
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
        {
          for(unsigned int i = 0; i < degree * (degree - 1); ++i)
            dst_ptr[dof_indices[2 * dim][v] + i] += data[i][v];
        }
    }

  const dealii::VectorizedArray<Number> * my_data =
    local_array + (dim - 1) * dofs_per_comp + dofs_per_face;
  Number * dst_ptr = dst_vector + (dim - 1) * dofs_per_comp - (2 * dim - 2) * dofs_per_face;

  if(n_filled_lanes == n_lanes)
    vectorized_transpose_and_store(
      true, dofs_per_comp - 2 * dofs_per_face, my_data, dof_indices[2 * dim].data(), dst_ptr);
  else
    for(unsigned int v = 0; v < n_filled_lanes; ++v)
    {
      for(unsigned int i = 0; i < dofs_per_comp - 2 * dofs_per_face; ++i)
        dst_ptr[dof_indices[2 * dim][v] + i] += my_data[i][v];
    }
}


using namespace dealii;



template<int dim, typename Number = double>
class RaviartThomasOperator : public EnableObserverPointer
{
public:
  static constexpr unsigned int n_lanes = VectorizedArray<Number>::size();
  using VectorType                      = LinearAlgebra::distributed::Vector<Number>;

  RaviartThomasOperator()
    : is_test(false), timings{}, factor_mass(0), factor_laplace(0), pressure_dof_indices(nullptr)
  {
  }

  template<typename OtherNumber>
  void
  reinit(const Mapping<dim> &                   mapping,
         const DoFHandler<dim> &                dof_handler,
         const AffineConstraints<OtherNumber> & constraints,
         const std::vector<unsigned int> &      cell_vectorization_category,
         const Quadrature<1> &                  quadrature,
         const bool                             is_test,
         const MPI_Comm                         communicator_shared = MPI_COMM_SELF)
  {
    this->is_test                 = is_test;
    this->mapping                 = &mapping;
    this->dof_handler             = &dof_handler;
    const FiniteElement<dim> & fe = dof_handler.get_fe();

    // per cell batch, face and lane: index of the face among the faces
    // whose data is received from remote processes, used to set up
    // single_exchange
    std::vector<dealii::ndarray<unsigned int, 2 * dim, n_lanes>> mpi_exchange_data_on_faces;

    AssertThrow(dynamic_cast<const FE_RaviartThomasNodal<dim> *>(&fe),
                ExcMessage("This class only works for Raviart-Thomas elements. Set "
                           "spatial discretization to HDIV"));
    MatrixFree<dim, Number>                          matrix_free;
    typename MatrixFree<dim, Number>::AdditionalData mf_data;
    mf_data.cell_vectorization_category          = cell_vectorization_category;
    mf_data.cell_vectorization_categories_strict = false;
    mf_data.overlap_communication_computation    = false;
    mf_data.initialize_mapping                   = false;
    matrix_free.reinit(MappingQ1<dim>(), dof_handler, constraints, quadrature, mf_data);

    {
      const MPI_Comm                comm               = dof_handler.get_mpi_communicator();
      IndexSet                      locally_owned_dofs = dof_handler.locally_owned_dofs();
      const types::global_dof_index n_unconstrained_owned_dofs =
        locally_owned_dofs.n_elements() - matrix_free.get_constrained_dofs().size();
      const types::global_dof_index first_constrained_dof =
        matrix_free.get_constrained_dofs().empty() ? n_unconstrained_owned_dofs :
                                                     matrix_free.get_constrained_dofs()[0];

      AssertThrow(first_constrained_dof == n_unconstrained_owned_dofs,
                  ExcMessage("Expected all constrained DoFs to be sorted to "
                             "the end of locally owned range, but found " +
                             std::to_string(first_constrained_dof) + " vs " +
                             std::to_string(n_unconstrained_owned_dofs)));

      // Assign DoF numbers for the unconstrained degrees of freedom only
      std::pair<types::global_dof_index, types::global_dof_index> positions =
        Utilities::MPI::partial_and_total_sum(n_unconstrained_owned_dofs, comm);
      std::vector<types::global_dof_index> dof_numbers(locally_owned_dofs.n_elements(),
                                                       numbers::invalid_dof_index);
      types::global_dof_index              counter = positions.first;
      for(unsigned int i = 0; i < n_unconstrained_owned_dofs; ++i)
        dof_numbers[i] = counter++;

      // Extract ghost entries for which we need to query the numbers from
      // remote processes - we do not know the start indices of the respective
      // processes even though we could query the DoF index owner through the
      // triangulation, so we need to perform a lookup anyway and do that by a
      // ghost exchange of all data. While there, also extract a compressed
      // representation, taking the first number on each entity (face, cell)
      // in global index space, which we later translate to local numbers

      std::vector<types::global_dof_index> local_dof_indices(fe.dofs_per_cell);
      std::vector<dealii::ndarray<types::global_dof_index, 2 * dim + 1, n_lanes>>
                                           dof_indices_per_entity(matrix_free.n_cell_batches());
      std::vector<types::global_dof_index> ghost_indices;
      for(unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
        for(unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(cell); ++v)
        {
          matrix_free.get_cell_iterator(cell, v)->get_dof_indices(local_dof_indices);

          // Adjust dof indices due to periodicity
          for(types::global_dof_index & a : local_dof_indices)
          {
            const auto line = constraints.get_constraint_entries(a);
            if(line != nullptr && line->size() == 1 && (*line)[0].second == Number(1.0))
              a = (*line)[0].first;
            else if(line != nullptr && line->size() > 0)
              AssertThrow(false,
                          ExcMessage("Unsupported constraint of length " +
                                     std::to_string(line->size()) +
                                     " with non-unit constraint found"));
          }
          for(types::global_dof_index a : local_dof_indices)
            if(!locally_owned_dofs.is_element(a) && !constraints.is_constrained(a))
              ghost_indices.push_back(a);

          const unsigned int dofs_per_face = fe.dofs_per_face;
          for(unsigned int f = 0; f < 2 * dim; ++f)
          {
            for(unsigned int i = 0; i < dofs_per_face; ++i)
              AssertThrow(local_dof_indices[f * dofs_per_face + i] ==
                            local_dof_indices[f * dofs_per_face] + i,
                          ExcInternalError());

            dof_indices_per_entity[cell][f][v] = local_dof_indices[f * dofs_per_face];
          }
          const unsigned int start_cell_dofs = 2 * dim * dofs_per_face;
          for(unsigned int i = 0; i < fe.dofs_per_cell - start_cell_dofs; ++i)
            AssertThrow(local_dof_indices[start_cell_dofs + i] ==
                          local_dof_indices[start_cell_dofs] + i,
                        ExcInternalError());
          dof_indices_per_entity[cell][2 * dim][v] = local_dof_indices[start_cell_dofs];
        }

      IndexSet ghost_index_set(locally_owned_dofs.size());
      ghost_index_set.add_indices(ghost_indices.begin(), ghost_indices.end());
      ghost_index_set.compress();
      Utilities::MPI::Partitioner partitioner_dofs(locally_owned_dofs, ghost_index_set, comm);

      std::vector<types::global_dof_index> tmp_array(partitioner_dofs.n_import_indices());
      std::vector<types::global_dof_index> numbers_ghosts(partitioner_dofs.n_ghost_indices());
      std::vector<MPI_Request>             requests;
      partitioner_dofs.export_to_ghosted_array_start(3,
                                                     make_const_array_view(dof_numbers),
                                                     make_array_view(tmp_array),
                                                     make_array_view(numbers_ghosts),
                                                     requests);
      partitioner_dofs.export_to_ghosted_array_finish(make_array_view(numbers_ghosts), requests);

      IndexSet owned_dofs(positions.second);
      owned_dofs.add_range(positions.first, positions.first + n_unconstrained_owned_dofs);
      std::vector<types::global_dof_index> compressed_ghost;
      compressed_ghost.reserve(numbers_ghosts.size());
      for(const types::global_dof_index a : numbers_ghosts)
        if(a != numbers::invalid_dof_index)
          compressed_ghost.push_back(a);
      IndexSet ghost_dofs(positions.second);
      ghost_dofs.add_indices(compressed_ghost.begin(), compressed_ghost.end());
      ghost_dofs.compress();

      partitioner = std::make_shared<Utilities::MPI::Partitioner>(owned_dofs, ghost_dofs, comm);

      {
        ndarray<unsigned int, 2 * dim, n_lanes> default_argument;
        for(unsigned int i = 0; i < 2 * dim; ++i)
          for(unsigned int j = 0; j < n_lanes; ++j)
            default_argument[i][j] = numbers::invalid_unsigned_int;
        neighbor_cells.assign(matrix_free.n_cell_batches(), default_argument);
        mpi_exchange_data_on_faces.assign(matrix_free.n_cell_batches(), default_argument);
      }
      {
        ndarray<unsigned int, 2 * dim + 1, n_lanes> default_argument;
        for(unsigned int i = 0; i < 2 * dim + 1; ++i)
          for(unsigned int j = 0; j < n_lanes; ++j)
            default_argument[i][j] = numbers::invalid_unsigned_int;
        dof_indices.assign(matrix_free.n_cell_batches(), default_argument);
      }

      std::array<std::vector<unsigned int>, 2 * dim> cells_at_dirichlet_boundary_by_face;
      std::vector<unsigned int> cell_indices(dof_handler.get_triangulation().n_active_cells(),
                                             numbers::invalid_unsigned_int);
      for(unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
        for(unsigned int v = 0; v < n_lanes; ++v)
        {
          if(v >= matrix_free.n_active_entries_per_cell_batch(cell))
          {
            // fill in valid data to simplify work with vector access
            for(unsigned int f = 0; f < 2 * dim + 1; ++f)
              dof_indices[cell][f][v] = dof_indices[cell][f][0];
            continue;
          }

          for(unsigned int f = 0; f < 2 * dim + 1; ++f)
          {
            const types::global_dof_index index    = dof_indices_per_entity[cell][f][v];
            const unsigned int            my_index = partitioner_dofs.global_to_local(index);
            types::global_dof_index       number_compressed;
            if(my_index < locally_owned_dofs.n_elements())
              number_compressed = dof_numbers[my_index];
            else
              number_compressed = numbers_ghosts[my_index - dof_numbers.size()];
            Assert(number_compressed != numbers::invalid_dof_index ||
                     constraints.is_constrained(index),
                   ExcInternalError());
            if(number_compressed != numbers::invalid_dof_index)
              dof_indices[cell][f][v] = partitioner->global_to_local(number_compressed);
            else
              dof_indices[cell][f][v] = numbers::invalid_unsigned_int;
          }

          cell_indices[matrix_free.get_cell_iterator(cell, v)->active_cell_index()] =
            cell * VectorizedArray<Number>::size() + v;

          for(unsigned int f = 0; f < 2 * dim; ++f)
            if(dof_indices[cell][f][v] == numbers::invalid_unsigned_int)
              cells_at_dirichlet_boundary_by_face[f].push_back(cell * n_lanes + v);
        }

      // put cells at Dirichlet boundary together in one data structure
      unsigned int n_batches_boundary = 0;
      for(unsigned int f = 0; f < 2 * dim; ++f)
        n_batches_boundary +=
          (cells_at_dirichlet_boundary_by_face[f].size() + n_lanes - 1) / n_lanes;
      cells_at_dirichlet_boundary.clear();
      cells_at_dirichlet_boundary.reserve(n_batches_boundary);
      for(unsigned int f = 0; f < 2 * dim; ++f)
      {
        const unsigned int n_cells = cells_at_dirichlet_boundary_by_face[f].size();
        for(unsigned int i = 0; i < n_cells; i += n_lanes)
        {
          std::pair<unsigned int, std::array<unsigned int, n_lanes>> entry;
          entry.first = f;
          for(unsigned int v = 0; v < n_lanes; ++v)
            if(i + v < n_cells)
              entry.second[v] = cells_at_dirichlet_boundary_by_face[f][i + v];
            else
              entry.second[v] = numbers::invalid_unsigned_int;
          cells_at_dirichlet_boundary.push_back(entry);
        }
      }

      std::map<unsigned int, std::vector<std::array<types::global_dof_index, 5>>> proc_neighbors;
      for(unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
        for(unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(cell); ++v)
        {
          const auto dcell = matrix_free.get_cell_iterator(cell, v);
          for(unsigned int f = 0; f < 2 * dim; ++f)
          {
            const bool at_boundary           = dcell->at_boundary(f);
            const bool has_periodic_neighbor = at_boundary && dcell->has_periodic_neighbor(f);
            if(at_boundary == false || has_periodic_neighbor)
            {
              const auto neighbor =
                has_periodic_neighbor ? dcell->periodic_neighbor(f) : dcell->neighbor(f);
              if(neighbor->is_locally_owned())
              {
                AssertIndexRange(neighbor->active_cell_index(), cell_indices.size());
                neighbor_cells[cell][f][v] = cell_indices[neighbor->active_cell_index()];
              }
              else
              {
                std::array<types::global_dof_index, 5> neighbor_data;
                neighbor_data[0] = cell * n_lanes + v;
                neighbor_data[1] = has_periodic_neighbor ? dcell->periodic_neighbor_face_no(f) :
                                                           dcell->neighbor_face_no(f);
                neighbor_data[2] = dcell->global_active_cell_index();
                neighbor_data[3] = neighbor->global_active_cell_index();
                neighbor_data[4] = f;
                proc_neighbors[neighbor->subdomain_id()].push_back(neighbor_data);
                // set dummy
                neighbor_cells[cell][f][v] = cell_indices[dcell->active_cell_index()];
              }
            }
            else
              AssertThrow(
                f < 4, ExcMessage("Non-periodic boundaries in z direction currently unsupported"));
          }
        }

      // find out how the neighbor wants us to send the data -> we sort by the
      // global cell index. We use a reverse order of processes as the data
      // access might favor nearby access to cell data.
      send_data_process.clear();
      send_data_cell_index.clear();
      send_data_face_index.clear();
      {
        std::size_t sizes = 0;
        for(const auto & it : proc_neighbors)
          sizes += it.second.size();
        send_data_cell_index.reserve(sizes);
        send_data_face_index.reserve(sizes);
      }
      unsigned int face_index = 0;
      for(auto it = proc_neighbors.rbegin(); it != proc_neighbors.rend(); ++it)
      {
        std::sort(it->second.begin(),
                  it->second.end(),
                  [](const std::array<types::global_dof_index, 5> & a,
                     const std::array<types::global_dof_index, 5> & b) {
                    if(a[4] < b[4])
                      return true;
                    else if(a[4] == b[4] && a[3] < b[3])
                      return true;
                    else
                      return false;
                  });
        send_data_process.emplace_back(it->first, it->second.size());
        for(unsigned int i = 0; i < it->second.size(); ++i)
        {
          send_data_cell_index.emplace_back(it->second[i][0]);
          send_data_face_index.emplace_back(it->second[i][4]);
        }

        // finally figure out where to read the imported face data from,
        // which is done by sorting in the way the neighboring process would
        // have run the sorting we just did a few lines up
        std::sort(it->second.begin(),
                  it->second.end(),
                  [](const std::array<types::global_dof_index, 5> & a,
                     const std::array<types::global_dof_index, 5> & b) {
                    if(a[1] < b[1])
                      return true;
                    else if(a[1] == b[1] && a[2] < b[2])
                      return true;
                    else
                      return false;
                  });

        for(unsigned int i = 0; i < it->second.size(); ++i)
        {
          const unsigned int cell                               = it->second[i][0] / n_lanes;
          const unsigned int v                                  = it->second[i][0] % n_lanes;
          mpi_exchange_data_on_faces[cell][it->second[i][4]][v] = face_index;
          ++face_index;
        }
      }
    }

    cell_level_index.resize(matrix_free.n_cell_batches());
    AssertDimension(cell_level_index.size(), dof_indices.size());
    for(unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
      for(unsigned int lane = 0; lane < n_lanes; ++lane)
        if(lane >= matrix_free.n_active_entries_per_cell_batch(cell))
          cell_level_index[cell][lane] = {
            {numbers::invalid_unsigned_int, numbers::invalid_unsigned_int}};
        else
        {
          const auto iter                 = matrix_free.get_cell_iterator(cell, lane);
          cell_level_index[cell][lane][0] = iter->level();
          cell_level_index[cell][lane][1] = iter->index();
        }

    mapping_info.reinit(mapping, quadrature, dof_handler.get_triangulation(), cell_level_index);

    compute_vector_access_pattern();

    setup_scratch();

    {
      std::vector<Polynomials::Polynomial<double>> basis =
        Polynomials::generate_complete_Lagrange_basis(quadrature.get_points());
      interpolate_quad_to_boundary[0].resize(basis.size());
      interpolate_quad_to_boundary[1].resize(basis.size());
      std::vector<double> val_and_der(2);
      for(unsigned int i = 0; i < basis.size(); ++i)
      {
        basis[i].value(0., val_and_der);
        interpolate_quad_to_boundary[0][i][0] = val_and_der[0];
        interpolate_quad_to_boundary[0][i][1] = val_and_der[1];
        basis[i].value(1., val_and_der);
        interpolate_quad_to_boundary[1][i][0] = val_and_der[0];
        interpolate_quad_to_boundary[1][i][1] = val_and_der[1];
      }
    }

    shape_info = matrix_free.get_shape_info();

    // The face integrals take the value of the normal component on the
    // neighbor from the current cell, which requires shape functions that
    // are nodal on the faces in normal direction
    for(unsigned int side = 0; side < 2; ++side)
      for(unsigned int i = 0; i <= fe.degree; ++i)
        AssertThrow(std::abs(shape_info.data[0].shape_data_on_face[side][i] -
                             (i == side * fe.degree ? 1. : 0.)) < 1e-12,
                    ExcMessage("Expected nodal shape functions in normal direction"));

    {
      QGauss<1> quad(fe.degree + (fe.degree + 1) / 2);

      convective_mapping_info.reinit(mapping,
                                     quad,
                                     dof_handler.get_triangulation(),
                                     cell_level_index);

      convective_shape_info.reinit(quad, fe);

      std::vector<Polynomials::Polynomial<double>> basis =
        Polynomials::generate_complete_Lagrange_basis(quad.get_points());
      convective_interpolate_quad_to_boundary[0].resize(basis.size());
      convective_interpolate_quad_to_boundary[1].resize(basis.size());
      std::vector<double> val_and_der(2);
      for(unsigned int i = 0; i < basis.size(); ++i)
      {
        basis[i].value(0., val_and_der);
        convective_interpolate_quad_to_boundary[0][i][0] = val_and_der[0];
        convective_interpolate_quad_to_boundary[0][i][1] = val_and_der[1];
        basis[i].value(1., val_and_der);
        convective_interpolate_quad_to_boundary[1][i][0] = val_and_der[0];
        convective_interpolate_quad_to_boundary[1][i][1] = val_and_der[1];
      }
    }

    detect_dependencies_of_face_integrals();

    single_exchange.reinit(dof_handler,
                           shape_info,
                           *partitioner,
                           dof_indices,
                           cell_level_index,
                           mpi_exchange_data_on_faces,
                           send_data_process,
                           send_data_cell_index,
                           send_data_face_index);

    mpi_recv_buffer.clear();
    mpi_send_buffer.clear();
    {
      mpi_recv_buffer.resize_fast(single_exchange.recv_buffer_size);
      mpi_requests.resize(2 * single_exchange.partners.size());

      // With a shared-memory communicator, the send buffer is allocated in
      // shared memory, and the partners in the same shared-memory domain
      // read their messages directly from it.
      single_exchange.setup_shared_memory(dof_handler.get_mpi_communicator(), communicator_shared);
      free_communicator_sm();
      if(communicator_shared != MPI_COMM_SELF)
      {
        // keep a duplicate of the shared-memory communicator for the
        // vectors created by initialize_dof_vector(), which store it
        const int ierr = MPI_Comm_dup(communicator_shared, &communicator_sm);
        AssertThrowMPI(ierr);
      }
      shared_send_buffer.values = Kokkos::View<Number *, Kokkos::HostSpace>();
      shared_send_buffer.values_sm_ptr.reset();
      shared_send_buffer.values_sm.clear();
      if(communicator_shared != MPI_COMM_SELF)
      {
        mpi_shared_memory::allocate_shared_recv_data(shared_send_buffer,
                                                     single_exchange.send_buffer_size,
                                                     communicator_shared);
        single_exchange.set_buffers(mpi_recv_buffer.data(), shared_send_buffer.values_sm);
      }
      else
      {
        mpi_send_buffer.resize_fast(single_exchange.send_buffer_size);
        single_exchange.set_buffers(mpi_recv_buffer.data(), {});
      }
    }

    set_parameters(0.0, 1.0);

    for(double & t : timings)
      t = 0.0;
    product_timings            = {};
    time_wait_readers          = 0;
    time_wait_readers_deferred = 0;
  }

  ~RaviartThomasOperator()
  {
    free_communicator_sm();

    // Nothing to report if reinit() was never called, otherwise print
    // the timing information.
    if(dof_handler == nullptr)
      return;
    const MPI_Comm comm = dof_handler->get_mpi_communicator();
    if(not this->is_test and timings[0] > 0)
    {
      std::cout << std::defaultfloat << std::setprecision(3);
      const double total_time =
        Utilities::MPI::sum(timings[9], comm) / timings[0] / Utilities::MPI::n_mpi_processes(comm);
      if(Utilities::MPI::this_mpi_process(comm) == 0)
        std::cout << "Collected timings for RT Laplace operator <"
                  << (std::is_same_v<Number, double> ? "double" : "float") << "> in "
                  << static_cast<unsigned long>(timings[0])
                  << " evaluations [t_total=" << total_time * timings[0] << "s]" << std::endl;
      Helper::print_time(timings[1] / timings[0], "Pack all ghosts and send", comm, total_time);
      if(Utilities::MPI::n_mpi_processes(comm) > 1)
      {
        Helper::print_time(timings[3] / timings[0], "Wait and unpack ghosts", comm, total_time);
        Helper::print_time(timings[4] / timings[0], "Pre-loop before ghosts", comm, total_time);
      }

      Helper::print_time(timings[7] / timings[0], "Matrix-free loop", comm, total_time);
      Helper::print_time_per_node(timings[7] / timings[0],
                                  "Matrix-free loop, node avg",
                                  comm,
                                  total_time);
      Helper::print_time((timings[8] - time_wait_readers) / timings[0],
                         "Compress cell ghost values",
                         comm,
                         total_time);
      if(Utilities::MPI::sum(single_exchange.n_shared_memory_partners(), comm) > 0)
        Helper::print_time((time_wait_readers + time_wait_readers_deferred) / timings[0],
                           "Wait for readers (shm)",
                           comm,
                           total_time);

      // time of the whole product per call, for each variant
      const std::array<std::string, 3> names = {{"vmult with functions", "vmult", "vmult_scratch"}};
      for(unsigned int k = 0; k < 3; ++k)
        if(product_timings[k][0] > 0)
          Helper::print_time(product_timings[k][1] / product_timings[k][0],
                             names[k] + " (" +
                               std::to_string(static_cast<unsigned long>(product_timings[k][0])) +
                               "x)",
                             comm);
      if(Utilities::MPI::this_mpi_process(comm) == 0)
        std::cout << std::endl;
    }
    if(not this->is_test and timings[10] > 0)
    {
      const double total_time = Utilities::MPI::sum(timings[14], comm) / timings[10] /
                                Utilities::MPI::n_mpi_processes(comm);
      if(Utilities::MPI::this_mpi_process(comm) == 0)
        std::cout << "Collected timings for mass operator <"
                  << (std::is_same_v<Number, double> ? "double" : "float") << "> in "
                  << static_cast<unsigned long>(timings[10])
                  << " evaluations [t_total=" << total_time * timings[10] << "s]" << std::endl;
      Helper::print_time(timings[11] / timings[10], "Update cell ghost values", comm, total_time);

      Helper::print_time(timings[12] / timings[10], "Matrix-free loop", comm, total_time);
      Helper::print_time(timings[13] / timings[10], "Compress cell ghost values", comm, total_time);
      if(Utilities::MPI::this_mpi_process(comm) == 0)
        std::cout << std::endl;
    }
  }

  unsigned int
  n_cells_at_dirichlet_boundary() const
  {
    return cells_at_dirichlet_boundary.size();
  }

  void
  initialize_dof_vector(VectorType & vec) const
  {
    // With a shared-memory communicator, the vectors are allocated in
    // shared memory, so that the processes of the same shared-memory domain
    // can read each other's locally owned entries
    vec.reinit(partitioner, communicator_sm);
  }

  template<typename VectorType1, typename VectorType2>
  void
  copy_mf_to_this_vector(const VectorType1 & src, VectorType2 & dst) const
  {
    Assert(partitioner->is_compatible(*dst.get_partitioner()), ExcInternalError());
    AssertIndexRange(dst.locally_owned_size(), src.locally_owned_size() + 1);
    std::copy(src.begin(), src.begin() + dst.locally_owned_size(), dst.begin());
  }

  template<typename VectorType1, typename VectorType2>
  void
  copy_this_to_mf_vector(const VectorType1 & src, VectorType2 & dst) const
  {
    Assert(partitioner->is_compatible(*src.get_partitioner()), ExcInternalError());
    AssertIndexRange(src.locally_owned_size(), dst.locally_owned_size() + 1);
    std::copy(src.begin(), src.begin() + src.locally_owned_size(), dst.begin());
  }

  void
  set_parameters(const double factor_mass, const double factor_laplace)
  {
    this->factor_mass    = factor_mass;
    this->factor_laplace = factor_laplace;
  }

  void
  vmult(VectorType & dst, const VectorType & src) const
  {
    vmult_before_on_dst(
      dst,
      src,
      [&dst](const unsigned int start, const unsigned int end) {
        std::memset(dst.begin() + start,
                    0,
                    (end - start) * sizeof(typename VectorType::value_type));
      },
      [](const unsigned int, const unsigned int) {});
  }

  void
  vmult_add(VectorType & dst, const VectorType & src) const
  {
    vmult_before_on_dst(
      dst,
      src,
      [](const unsigned int, const unsigned int) {},
      [](const unsigned int, const unsigned int) {});
  }

  void
  compute_diagonal(VectorType & dst) const
  {
    initialize_dof_vector(dst);
    const unsigned int n_cell_batches = dof_indices.size();
    for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
      diagonal_operation(cell, dst);
    dst.compress(VectorOperation::add);
  }

  // The parameter `factor_convective` can be set to (1.0) for the convective
  // form of the convective term, 0.5 for the skew-symmetric form and to 0.0
  // for the divergence form.
  void
  evaluate_convective_term(const VectorType & src,
                           const Number       factor_convective,
                           VectorType &       dst) const
  {
    const unsigned int n_cell_batches = dof_indices.size();

    exchange_ghosts_and_face_data(src);

    // since we just zero out the dst vector, can use the mass list
    for(unsigned int range = cell_loop_pre_list_index[n_cell_batches];
        range < cell_loop_pre_list_index[n_cell_batches + 1];
        ++range)
      std::fill(dst.begin() + cell_loop_pre_list[range].first,
                dst.begin() + cell_loop_pre_list[range].second,
                Number());

    for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
    {
      for(unsigned int range = cell_loop_mass_pre_list_index[cell];
          range < cell_loop_mass_pre_list_index[cell + 1];
          ++range)
        std::fill(dst.begin() + cell_loop_mass_pre_list[range].first,
                  dst.begin() + cell_loop_mass_pre_list[range].second,
                  Number());

      const unsigned int degree = shape_info.data[0].fe_degree;
      if(degree == 2)
        do_convective_on_cell<2>(cell, src, factor_convective, dst);
      else if(degree == 3)
        do_convective_on_cell<3>(cell, src, factor_convective, dst);
      else if(degree == 4)
        do_convective_on_cell<4>(cell, src, factor_convective, dst);
#ifndef DEBUG
      else if(degree == 5)
        do_convective_on_cell<5>(cell, src, factor_convective, dst);
      else if(degree == 6)
        do_convective_on_cell<6>(cell, src, factor_convective, dst);
      else if(degree == 7)
        do_convective_on_cell<7>(cell, src, factor_convective, dst);
      else if(degree == 8)
        do_convective_on_cell<8>(cell, src, factor_convective, dst);
      else if(degree == 9)
        do_convective_on_cell<9>(cell, src, factor_convective, dst);
#endif
      else
        AssertThrow(false, ExcMessage("Degree " + std::to_string(degree) + " not instantiated"));
    }

    single_exchange.signal_done_reading();
    dst.compress_start(0, VectorOperation::add);
    src.zero_out_ghost_values();
    dst.compress_finish(VectorOperation::add);
    single_exchange.wait_until_source_vector_free();
  }

  Number
  evaluate_convective_and_divergence_term(const VectorType & src,
                                          VectorType &       velocity_dst,
                                          VectorType &       convective_divergence,
                                          VectorType &       divergence) const
  {
    const unsigned int n_cell_batches = dof_indices.size();

    exchange_ghosts_and_face_data(src);

    // since we just zero out the dst vector, can use the mass list
    for(unsigned int range = cell_loop_pre_list_index[n_cell_batches];
        range < cell_loop_pre_list_index[n_cell_batches + 1];
        ++range)
      std::fill(velocity_dst.begin() + cell_loop_pre_list[range].first,
                velocity_dst.begin() + cell_loop_pre_list[range].second,
                Number());

    Number cfl_factor = 0.;
    for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
    {
      for(unsigned int range = cell_loop_mass_pre_list_index[cell];
          range < cell_loop_mass_pre_list_index[cell + 1];
          ++range)
        std::fill(velocity_dst.begin() + cell_loop_mass_pre_list[range].first,
                  velocity_dst.begin() + cell_loop_mass_pre_list[range].second,
                  Number());

      VectorizedArray<Number> convective_cfl = 0.;
      const unsigned int      degree         = shape_info.data[0].fe_degree;
      if(degree == 2)
        convective_cfl = do_convective_and_divergence_on_cell<2>(
          cell, src, velocity_dst, convective_divergence, divergence);
      else if(degree == 3)
        convective_cfl = do_convective_and_divergence_on_cell<3>(
          cell, src, velocity_dst, convective_divergence, divergence);
      else if(degree == 4)
        convective_cfl = do_convective_and_divergence_on_cell<4>(
          cell, src, velocity_dst, convective_divergence, divergence);
#ifndef DEBUG
      else if(degree == 5)
        convective_cfl = do_convective_and_divergence_on_cell<5>(
          cell, src, velocity_dst, convective_divergence, divergence);
      else if(degree == 6)
        convective_cfl = do_convective_and_divergence_on_cell<6>(
          cell, src, velocity_dst, convective_divergence, divergence);
      else if(degree == 7)
        convective_cfl = do_convective_and_divergence_on_cell<7>(
          cell, src, velocity_dst, convective_divergence, divergence);
      else if(degree == 8)
        convective_cfl = do_convective_and_divergence_on_cell<8>(
          cell, src, velocity_dst, convective_divergence, divergence);
      else if(degree == 9)
        convective_cfl = do_convective_and_divergence_on_cell<9>(
          cell, src, velocity_dst, convective_divergence, divergence);
#endif
      else
        AssertThrow(false, ExcMessage("Degree " + std::to_string(degree) + " not instantiated"));

      for(unsigned int v = 0; v < n_active_entries_per_cell_batch(cell); ++v)
        cfl_factor = std::max(cfl_factor, convective_cfl[v]);
    }

    single_exchange.signal_done_reading();
    velocity_dst.compress_start(0, VectorOperation::add);
    src.zero_out_ghost_values();
    velocity_dst.compress_finish(VectorOperation::add);
    single_exchange.wait_until_source_vector_free();

    return Utilities::MPI::max(cfl_factor, divergence.get_mpi_communicator());
  }

  // Interface for benchmarking the cell kernels of vmult() and
  // evaluate_convective_and_divergence_term() in isolation, used by
  // prototypes/rt_kernel_benchmark.cpp. benchmark_prepare() exchanges the
  // ghost values and face data of src like the regular loops. Then,
  // benchmark_cell_range() may be called any number of times, followed by
  // benchmark_finish(). benchmark_cell_range() runs a kernel on the cell
  // batches [begin, end), without zeroing or compressing the result vectors.
  // The kernel stops after the phase 'last_stage':
  //   0: interpolation of the cell values to the quadrature points
  //   1: + cell integrals (momentum) or face integrals (convective)
  //   2: + face integrals (momentum) or cell integrals (convective, which
  //      include applying the face terms, the integration of the
  //      divergence and the gradient part of the convective divergence)
  //   3: + integration and scatter into the result vectors (full kernel)
  // The kernel 'convective' (the convective term alone, as in
  // evaluate_convective_term() with factor 1) only runs in full, i.e., with
  // last_stage = 3.
  enum class BenchmarkKernel
  {
    momentum,
    convective,
    convective_and_divergence
  };

  // Switch the flux buffer of the face integrals of the convective term
  // off, so that each cell computes the face integrals of all its faces
  // itself, for comparisons
  void
  set_use_convective_face_flux_buffer(const bool use_buffer)
  {
    use_convective_face_flux_buffer = use_buffer;
  }

  void
  benchmark_prepare(const VectorType & src) const
  {
    exchange_ghosts_and_face_data(src);
  }

  void
  benchmark_finish(const VectorType & src) const
  {
    single_exchange.signal_done_reading();
    src.zero_out_ghost_values();
    single_exchange.wait_until_source_vector_free();
  }

  void
  benchmark_cell_range(const BenchmarkKernel kernel,
                       const unsigned int    last_stage,
                       const unsigned int    begin,
                       const unsigned int    end,
                       const VectorType &    src,
                       VectorType &          dst,
                       VectorType &          convective_divergence,
                       VectorType &          divergence) const
  {
    AssertThrow(last_stage <= 3, ExcIndexRange(last_stage, 0, 4));
    AssertThrow(begin <= end && end <= dof_indices.size(), ExcInternalError());
    AssertThrow(kernel != BenchmarkKernel::convective_and_divergence ||
                  pressure_dof_indices != nullptr,
                ExcMessage("Call initialize_coupling_pressure() first"));
    AssertThrow(kernel != BenchmarkKernel::convective || last_stage == 3,
                ExcMessage("The convective kernel only runs in full (last_stage = 3)"));
    const unsigned int degree = shape_info.data[0].fe_degree;
    if(degree == 2)
      benchmark_cell_range<2>(
        kernel, last_stage, begin, end, src, dst, convective_divergence, divergence);
    else if(degree == 3)
      benchmark_cell_range<3>(
        kernel, last_stage, begin, end, src, dst, convective_divergence, divergence);
    else if(degree == 4)
      benchmark_cell_range<4>(
        kernel, last_stage, begin, end, src, dst, convective_divergence, divergence);
#ifndef DEBUG
    else if(degree == 5)
      benchmark_cell_range<5>(
        kernel, last_stage, begin, end, src, dst, convective_divergence, divergence);
    else if(degree == 6)
      benchmark_cell_range<6>(
        kernel, last_stage, begin, end, src, dst, convective_divergence, divergence);
    else if(degree == 7)
      benchmark_cell_range<7>(
        kernel, last_stage, begin, end, src, dst, convective_divergence, divergence);
    else if(degree == 8)
      benchmark_cell_range<8>(
        kernel, last_stage, begin, end, src, dst, convective_divergence, divergence);
    else if(degree == 9)
      benchmark_cell_range<9>(
        kernel, last_stage, begin, end, src, dst, convective_divergence, divergence);
#endif
    else
      AssertThrow(false, ExcMessage("Degree " + std::to_string(degree) + " not instantiated"));
  }

  void
  evaluate_add_body_force(const double                  time,
                          const dealii::Function<dim> & rhs_function,
                          VectorType &                  dst) const
  {
    const unsigned int n_cell_batches = dof_indices.size();
    const_cast<dealii::Function<dim> &>(rhs_function).set_time(time);

    for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
    {
      const unsigned int degree = shape_info.data[0].fe_degree;
      if(degree == 2)
        do_body_force_on_cell<2>(cell, rhs_function, dst);
      else if(degree == 3)
        do_body_force_on_cell<3>(cell, rhs_function, dst);
      else if(degree == 4)
        do_body_force_on_cell<4>(cell, rhs_function, dst);
#ifndef DEBUG
      else if(degree == 5)
        do_body_force_on_cell<5>(cell, rhs_function, dst);
      else if(degree == 6)
        do_body_force_on_cell<6>(cell, rhs_function, dst);
      else if(degree == 7)
        do_body_force_on_cell<7>(cell, rhs_function, dst);
      else if(degree == 8)
        do_body_force_on_cell<8>(cell, rhs_function, dst);
      else if(degree == 9)
        do_body_force_on_cell<9>(cell, rhs_function, dst);
#endif
      else
        AssertThrow(false, ExcMessage("Degree " + std::to_string(degree) + " not instantiated"));
    }

    dst.compress(VectorOperation::add);
  }

  void
  set_penalty_parameters(const Number additional_factor)
  {
    // Same as in ExaDG::IP::get_penalty_factor<dim> but split off to test
    // this function independently of ExaDG
    const Number factor =
      additional_factor * Utilities::fixed_power<2>(dof_handler->get_fe().degree + 1.0);
    this->penalty_parameters.resize(dof_indices.size());
    const unsigned int n_q_points_1d = shape_info.data[0].n_q_points_1d;
    for(unsigned int cell = 0; cell < dof_indices.size(); ++cell)
      for(unsigned int v = 0; v < n_lanes; ++v)
      {
        for(unsigned int face = 0; face < 4; ++face)
        {
          const unsigned int idx =
            mapping_info.face_mapping_data_index[cell][face][0][v] / n_q_points_1d / 4;
          AssertThrow(idx < mapping_info.ip_penalty_factors.size(),
                      ExcIndexRange(idx, 0, mapping_info.ip_penalty_factors.size()));
          const unsigned int neigh_idx =
            mapping_info.face_mapping_data_index[cell][face][1][v] / n_q_points_1d / 4;
          AssertThrow(neigh_idx < mapping_info.ip_penalty_factors.size(),
                      ExcIndexRange(neigh_idx, 0, mapping_info.ip_penalty_factors.size()));
          this->penalty_parameters[cell][face][v] =
            std::max(mapping_info.ip_penalty_factors[idx],
                     mapping_info.ip_penalty_factors[neigh_idx]) *
            factor;
        }
        for(unsigned int face = 4; face < 2 * dim; ++face)
          this->penalty_parameters[cell][face][v] =
            mapping_info.ip_penalty_factors[mapping_info.mapping_data_index[cell][v] /
                                            Utilities::pow(n_q_points_1d, 2)] *
            factor;
      }
  }

  types::global_dof_index
  m() const
  {
    return partitioner->size();
  }

  Number
  el(const types::global_dof_index, const types::global_dof_index) const
  {
    AssertThrow(false, ExcNotImplemented());
    return 0;
  }

  void
  vmult(VectorType &                                                        dst,
        const VectorType &                                                  src,
        const std::function<void(const unsigned int, const unsigned int)> & before_loop,
        const std::function<void(const unsigned int, const unsigned int)> & after_loop) const
  {
    Timer total_timer;
    if(factor_laplace != 0.)
      timings[0] += 1;
    else
      timings[10] += 1;
    Timer time;

    const unsigned int n_cell_batches = dof_indices.size();

    for(unsigned int range = cell_loop_pre_list_index[n_cell_batches];
        range < cell_loop_pre_list_index[n_cell_batches + 1];
        ++range)
      before_loop(cell_loop_pre_list[range].first, cell_loop_pre_list[range].second);

    if(factor_laplace != 0.)
    {
      // Single communication algorithm: the messages only contain data
      // computed from locally owned entries, so we can pack them as soon as
      // the entries touched by the packing have passed through before_loop,
      // without previously calling 'src.update_ghost_values'.
      const unsigned int n_partners = single_exchange.partners.size();
      check_source_vector_is_shared(src);
      single_exchange.start_receives(mpi_recv_buffer.data(),
                                     src.get_mpi_communicator(),
                                     mpi_requests.data());

      // ghost entities must not contribute to the packed face data
      Assert(src.has_ghost_elements() == false, ExcInternalError());

      for(unsigned int range = cell_loop_pre_list_index[n_cell_batches + 1];
          range < cell_loop_pre_list_index[n_cell_batches + 2];
          ++range)
        before_loop(cell_loop_pre_list[range].first, cell_loop_pre_list[range].second);

      timings[4] += time.wall_time();
      time.restart();

      dispatch_single_exchange_pack_and_send(src);
      timings[1] += time.wall_time();
      time.restart();

      if(n_partners > 0)
      {
        const int ierr = MPI_Waitall(n_partners * 2, mpi_requests.data(), MPI_STATUSES_IGNORE);
        AssertThrowMPI(ierr);
      }
      single_exchange.unpack_and_complete(mpi_recv_buffer.data(),
                                          const_cast<Number *>(src.begin()));
      timings[3] += time.wall_time();
    }
    else
    {
      src.update_ghost_values();
      timings[11] += time.wall_time();
    }

    time.restart();

    for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
    {
      if(factor_laplace != 0.)
        for(unsigned int range = cell_loop_pre_list_index[cell];
            range < cell_loop_pre_list_index[cell + 1];
            ++range)
          before_loop(cell_loop_pre_list[range].first, cell_loop_pre_list[range].second);
      else
        for(unsigned int range = cell_loop_mass_pre_list_index[cell];
            range < cell_loop_mass_pre_list_index[cell + 1];
            ++range)
          before_loop(cell_loop_mass_pre_list[range].first, cell_loop_mass_pre_list[range].second);

      dispatch_cell_operation(cell, src, dst);

      for(unsigned int range = cell_loop_post_list_index[cell];
          range < cell_loop_post_list_index[cell + 1];
          ++range)
        after_loop(cell_loop_post_list[range].first, cell_loop_post_list[range].second);
    }

    if(factor_laplace != 0)
      timings[7] += time.wall_time();
    else
      timings[12] += time.wall_time();

    time.restart();
    single_exchange.signal_done_reading();
    dst.compress_start(0, VectorOperation::add);
    src.zero_out_ghost_values();
    dst.compress_finish(VectorOperation::add);
    wait_for_readers();

    for(unsigned int range = cell_loop_post_list_index[n_cell_batches];
        range < cell_loop_post_list_index[n_cell_batches + 1];
        ++range)
      after_loop(cell_loop_post_list[range].first, cell_loop_post_list[range].second);

    if(factor_laplace != 0)
    {
      timings[8] += time.wall_time();
      timings[9] += total_timer.wall_time();
      product_timings[0][0] += 1;
      product_timings[0][1] += total_timer.wall_time();
    }
    else
    {
      timings[13] += time.wall_time();
      timings[14] += total_timer.wall_time();
    }
  }

  // Variant of the vmult() function with 'before_loop' and 'after_loop' for
  // the case that 'before_loop' only touches entries of dst, but not of
  // src. Then, src is final upon entry, so the ghost exchange can start
  // right away, and 'before_loop' needs to run only before the first write
  // into dst by the cell integrals (as given by the mass pre list), rather
  // than before the first read of src by the face integrals of the
  // neighbors. This keeps the range of entries between 'before_loop' and
  // 'after_loop' much smaller.
  void
  vmult_before_on_dst(
    VectorType &                                                        dst,
    const VectorType &                                                  src,
    const std::function<void(const unsigned int, const unsigned int)> & before_loop,
    const std::function<void(const unsigned int, const unsigned int)> & after_loop) const
  {
    Timer total_timer;
    if(factor_laplace != 0.)
      timings[0] += 1;
    else
      timings[10] += 1;
    Timer time;

    const unsigned int n_cell_batches = dof_indices.size();

    // entries written by other processes in the compress step, which must be
    // processed before the cell loop
    const auto before_loop_mpi_entries = [&]() {
      for(unsigned int range = cell_loop_mass_pre_list_index[n_cell_batches];
          range < cell_loop_mass_pre_list_index[n_cell_batches + 1];
          ++range)
        before_loop(cell_loop_mass_pre_list[range].first, cell_loop_mass_pre_list[range].second);
    };

    exchange_ghosts_overlapped(src, before_loop_mpi_entries);

    time.restart();

    for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
    {
      for(unsigned int range = cell_loop_mass_pre_list_index[cell];
          range < cell_loop_mass_pre_list_index[cell + 1];
          ++range)
        before_loop(cell_loop_mass_pre_list[range].first, cell_loop_mass_pre_list[range].second);

      dispatch_cell_operation(cell, src, dst);

      for(unsigned int range = cell_loop_post_list_index[cell];
          range < cell_loop_post_list_index[cell + 1];
          ++range)
        after_loop(cell_loop_post_list[range].first, cell_loop_post_list[range].second);
    }

    if(factor_laplace != 0)
      timings[7] += time.wall_time();
    else
      timings[12] += time.wall_time();

    time.restart();
    single_exchange.signal_done_reading();
    dst.compress_start(0, VectorOperation::add);
    src.zero_out_ghost_values();
    dst.compress_finish(VectorOperation::add);
    wait_for_readers();

    for(unsigned int range = cell_loop_post_list_index[n_cell_batches];
        range < cell_loop_post_list_index[n_cell_batches + 1];
        ++range)
      after_loop(cell_loop_post_list[range].first, cell_loop_post_list[range].second);

    if(factor_laplace != 0)
    {
      timings[8] += time.wall_time();
      timings[9] += total_timer.wall_time();
      product_timings[1][0] += 1;
      product_timings[1][1] += total_timer.wall_time();
    }
    else
    {
      timings[13] += time.wall_time();
      timings[14] += total_timer.wall_time();
    }
  }

  // Variant of the matrix-vector product that accumulates the result into
  // a small scratch array instead of a global vector, see
  // ExaDG::ScratchMatrixVectorProduct for the format of the ranges passed
  // to 'after_loop'. Like for vmult_before_on_dst(), the vector src must be
  // final upon entry.
  //
  // With shared memory, other processes may still read src after this
  // function returns, so src must not be modified before the next call of
  // vmult_scratch(), which waits for them before its cell loop. This avoids
  // waiting for the slowest reader at the end of each product and suits the
  // Chebyshev iteration in PreconditionChebyshevScratch, which writes the
  // source vector of one product only in the 'after_loop' of the next one.
  template<typename AfterLoop>
  void
  vmult_scratch(const VectorType & src, const AfterLoop & after_loop) const
  {
    vmult_scratch(src, nullptr, after_loop);
  }

  // Variant of vmult_scratch() that also runs 'before_loop(begin, end)' on
  // the locally owned entries of src, like vmult(), before their first read
  // by this process (in the cell integrals or the packing of the data sent
  // to other processes) or by other processes in shared memory. Each entry
  // passes through 'before_loop' exactly once, in the ranges of the pre list
  // in units of 'vector_access_chunk_size'. As the other processes may read
  // src during the call, 'before_loop' may only change entries of src
  // (together with other vectors not read by the product).
  template<typename BeforeLoop, typename AfterLoop>
  void
  vmult_scratch(const VectorType & src,
                const BeforeLoop & before_loop,
                const AfterLoop &  after_loop) const
  {
    Timer total_timer;
    if(factor_laplace != 0.)
      timings[0] += 1;
    else
      timings[10] += 1;

    constexpr bool     has_before_loop = !std::is_same_v<BeforeLoop, std::nullptr_t>;
    const unsigned int n_cell_batches  = dof_indices.size();
    const auto &       pre_list_index =
      factor_laplace != 0. ? cell_loop_pre_list_index : cell_loop_mass_pre_list_index;
    const auto & pre_list = factor_laplace != 0. ? cell_loop_pre_list : cell_loop_mass_pre_list;
    [[maybe_unused]] const auto run_pre_list = [&](const unsigned int index) {
      if constexpr(has_before_loop)
        for(unsigned int range = pre_list_index[index]; range < pre_list_index[index + 1]; ++range)
          before_loop(pre_list[range].first, pre_list[range].second);
    };

    // entries sent to other processes as ghost values (both lists) and the
    // entries of the cells whose data is sent or read in shared memory by
    // other processes (Laplace list only)
    if constexpr(has_before_loop)
    {
      Timer time;
      run_pre_list(n_cell_batches);
      if(factor_laplace != 0.)
        run_pre_list(n_cell_batches + 1);
      timings[4] += time.wall_time();
    }

    exchange_ghosts_overlapped(src, [&]() { scratch.prepare_loop(); });

    // the processes reading the source vector of the previous product, which
    // may be written in 'after_loop', have finished by now in most cases
    wait_for_readers(false);

    Timer time;
    scratch.cell_loop(
      [&](const unsigned int cell, Number * scratch_array) {
        if constexpr(has_before_loop)
          run_pre_list(cell);
        dispatch_cell_operation(cell, src, scratch_indices[cell], scratch_array);
      },
      after_loop);

    if(factor_laplace != 0)
      timings[7] += time.wall_time();
    else
      timings[12] += time.wall_time();

    time.restart();
    single_exchange.signal_done_reading();
    scratch.finish_loop(after_loop, [&]() { src.zero_out_ghost_values(); });

    if(factor_laplace != 0)
    {
      timings[8] += time.wall_time();
      timings[9] += total_timer.wall_time();
      product_timings[2][0] += 1;
      product_timings[2][1] += total_timer.wall_time();
    }
    else
    {
      timings[13] += time.wall_time();
      timings[14] += total_timer.wall_time();
    }
  }

  // Print the setup of the data exchange: the number of neighboring
  // processes in the same and in other shared-memory domains, the share of
  // the faces to cells on other processes whose cell is read directly in
  // shared memory, and the size of the messages sent per product to the
  // other shared-memory domains
  void
  print_exchange_statistics() const
  {
    const MPI_Comm comm         = dof_handler->get_mpi_communicator();
    unsigned int   n_on_node    = 0;
    double         bytes_remote = 0;
    for(const auto & partner : single_exchange.partners)
      if(partner.sm_rank != numbers::invalid_unsigned_int)
        ++n_on_node;
      else
        bytes_remote += partner.send_size * sizeof(Number);
    const auto [n_faces, n_faces_shared] = n_remote_faces();

    if(Utilities::MPI::this_mpi_process(comm) == 0)
      std::cout << "Data exchange of RT operator <"
                << (std::is_same_v<Number, double> ? "double" : "float")
                << "> per process:" << std::endl;
    Helper::print_time(n_on_node, "Neighbors on same node", comm);
    Helper::print_time(single_exchange.partners.size() - n_on_node,
                       "Neighbors on other nodes",
                       comm);
    Helper::print_time(n_faces, "Faces to remote cells", comm);
    Helper::print_time(n_faces > 0 ? double(n_faces_shared) / n_faces : 1.,
                       "Share read in shared memory",
                       comm);
    Helper::print_time(bytes_remote / 1024., "Sent to other nodes [KiB]", comm);
    if(Utilities::MPI::this_mpi_process(comm) == 0)
      std::cout << std::endl;
  }

  // Number of faces to cells on other processes, and the number of those
  // whose cell can be read directly in shared memory
  std::pair<unsigned int, unsigned int>
  n_remote_faces() const
  {
    return {single_exchange.recv_face_offset.size(), single_exchange.n_slots_in_shared_memory()};
  }

  // Check the locations of the entities of remote cells in shared memory
  // with a vector created by initialize_dof_vector() whose locally owned
  // entries hold their global index, returning the number of mismatches
  unsigned int
  count_wrong_remote_cell_entities(const VectorType & global_indices) const
  {
    return single_exchange.count_wrong_remote_cell_entities(global_indices);
  }

  // Wait until the other processes have finished reading the source vector
  // of the last call of vmult_scratch(), see there
  void
  wait_until_source_vector_free() const
  {
    single_exchange.wait_until_source_vector_free();
  }

  // Number of neighboring processes whose face data is read directly from
  // shared memory, see reinit()
  unsigned int
  n_shared_memory_partners() const
  {
    return single_exchange.n_shared_memory_partners();
  }

  void
  verify_other_cell_level_index(
    const std::vector<dealii::ndarray<unsigned int, n_lanes, 2>> & other_cell_level_index) const
  {
    AssertThrow(cell_level_index.size() == other_cell_level_index.size(),
                ExcDimensionMismatch(cell_level_index.size(), other_cell_level_index.size()));
    for(unsigned int i = 0; i < cell_level_index.size(); ++i)
      for(unsigned int v = 0; v < n_lanes; ++v)
        for(unsigned int d = 0; d < 2; ++d)
          AssertThrow(cell_level_index[i][v][d] == other_cell_level_index[i][v][d],
                      ExcMessage("Found invalid cell/level index of cells in two operators "
                                 "for batch index " +
                                 std::to_string(i) + " and lane " + std::to_string(v) + ": " +
                                 std::to_string(cell_level_index[i][v][0]) + "," +
                                 std::to_string(cell_level_index[i][v][1]) + " vs " +
                                 std::to_string(other_cell_level_index[i][v][0]) + "," +
                                 std::to_string(other_cell_level_index[i][v][1])));
  }

  const std::vector<dealii::ndarray<unsigned int, VectorizedArray<Number>::size(), 2>> &
  get_cell_level_index() const
  {
    return cell_level_index;
  }

  void
  initialize_coupling_pressure(
    const FiniteElement<dim> &                             fe_pressure,
    const std::vector<std::array<unsigned int, n_lanes>> & pressure_dof_indices)
  {
    const FiniteElement<dim> & fe = dof_handler->get_fe();
    QGauss<1>                  quad(fe.degree + 1);
    pressure_shape_info.reinit(quad, fe_pressure);
    this->pressure_dof_indices = &pressure_dof_indices;
  }

  std::size_t
  memory_consumption() const
  {
    return MemoryConsumption::memory_consumption(dof_indices) +
           MemoryConsumption::memory_consumption(neighbor_cells) +
           MemoryConsumption::memory_consumption(single_exchange.remote_face_data) +
           MemoryConsumption::memory_consumption(mpi_recv_buffer) +
           MemoryConsumption::memory_consumption(mpi_send_buffer) +
           MemoryConsumption::memory_consumption(all_owned_faces) +
           mapping_info.memory_consumption() +
           MemoryConsumption::memory_consumption(interpolate_quad_to_boundary) +
           convective_mapping_info.memory_consumption() +
           MemoryConsumption::memory_consumption(send_data_process) +
           MemoryConsumption::memory_consumption(send_data_cell_index) +
           MemoryConsumption::memory_consumption(send_data_face_index) +
           MemoryConsumption::memory_consumption(cell_loop_pre_list_index) +
           MemoryConsumption::memory_consumption(cell_loop_pre_list) +
           MemoryConsumption::memory_consumption(cell_loop_mass_pre_list_index) +
           MemoryConsumption::memory_consumption(cell_loop_mass_pre_list) +
           MemoryConsumption::memory_consumption(cell_loop_post_list_index) +
           MemoryConsumption::memory_consumption(cell_loop_post_list) +
           MemoryConsumption::memory_consumption(face_flux_buffer) +
           MemoryConsumption::memory_consumption(face_flux_buffer_index) +
           MemoryConsumption::memory_consumption(all_left_face_fluxes_from_buffer) +
           MemoryConsumption::memory_consumption(convective_face_flux_buffer) +
           MemoryConsumption::memory_consumption(convective_face_flux_buffer_index);
  }

  void
  evaluate_momentum_rhs(const VectorType &                           pressure,
                        const VectorType &                           velocity_old,
                        const Number                                 velocity_scale,
                        VectorType &                                 rhs,
                        const std::shared_ptr<dealii::Function<dim>> source_function = {}) const
  {
    evaluate_momentum_rhs(
      pressure,
      velocity_old,
      velocity_scale,
      rhs,
      [&rhs](const unsigned int begin, const unsigned int end) {
        std::fill(rhs.begin() + begin, rhs.begin() + end, Number(0.));
      },
      [](const unsigned int, const unsigned int) {},
      source_function);
  }

  void
  evaluate_momentum_rhs(
    const VectorType &                                                  pressure,
    const VectorType &                                                  velocity_old,
    const Number                                                        velocity_scale,
    VectorType &                                                        rhs,
    const std::function<void(const unsigned int, const unsigned int)> & before_loop,
    const std::function<void(const unsigned int, const unsigned int)> & after_loop,
    const std::shared_ptr<dealii::Function<dim>>                        source_function = {}) const
  {
    AssertDimension(pressure_shape_info.data[0].n_q_points_1d, shape_info.data[0].n_q_points_1d);
    AssertDimension(pressure_shape_info.data[0].fe_degree, shape_info.data[1].fe_degree);

    // Run the functions on dependencies as determined by the 'mass' operator,
    // because there are no face integrals in this function.
    const unsigned int n_cell_batches = dof_indices.size();
    for(unsigned int range = cell_loop_pre_list_index[n_cell_batches];
        range < cell_loop_pre_list_index[n_cell_batches + 1];
        ++range)
      before_loop(cell_loop_pre_list[range].first, cell_loop_pre_list[range].second);

    velocity_old.update_ghost_values();
    for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
    {
      for(unsigned int range = cell_loop_mass_pre_list_index[cell];
          range < cell_loop_mass_pre_list_index[cell + 1];
          ++range)
        before_loop(cell_loop_mass_pre_list[range].first, cell_loop_mass_pre_list[range].second);

      const unsigned int degree = shape_info.data[0].fe_degree;
      if(degree == 2)
        do_momentum_rhs_on_cell<2>(
          cell, pressure, velocity_old, velocity_scale, source_function, rhs);
      else if(degree == 3)
        do_momentum_rhs_on_cell<3>(
          cell, pressure, velocity_old, velocity_scale, source_function, rhs);
      else if(degree == 4)
        do_momentum_rhs_on_cell<4>(
          cell, pressure, velocity_old, velocity_scale, source_function, rhs);
#ifndef DEBUG
      else if(degree == 5)
        do_momentum_rhs_on_cell<5>(
          cell, pressure, velocity_old, velocity_scale, source_function, rhs);
      else if(degree == 6)
        do_momentum_rhs_on_cell<6>(
          cell, pressure, velocity_old, velocity_scale, source_function, rhs);
      else if(degree == 7)
        do_momentum_rhs_on_cell<7>(
          cell, pressure, velocity_old, velocity_scale, source_function, rhs);
      else if(degree == 8)
        do_momentum_rhs_on_cell<8>(
          cell, pressure, velocity_old, velocity_scale, source_function, rhs);
      else if(degree == 9)
        do_momentum_rhs_on_cell<9>(
          cell, pressure, velocity_old, velocity_scale, source_function, rhs);
#endif
      else
        AssertThrow(false, ExcMessage("Degree " + std::to_string(degree) + " not instantiated"));

      for(unsigned int range = cell_loop_post_list_index[cell];
          range < cell_loop_post_list_index[cell + 1];
          ++range)
        after_loop(cell_loop_post_list[range].first, cell_loop_post_list[range].second);
    }

    rhs.compress_start(0, VectorOperation::add);
    velocity_old.zero_out_ghost_values();
    rhs.compress_finish(VectorOperation::add);

    for(unsigned int range = cell_loop_post_list_index[n_cell_batches];
        range < cell_loop_post_list_index[n_cell_batches + 1];
        ++range)
      after_loop(cell_loop_post_list[range].first, cell_loop_post_list[range].second);
  }

  void
  evaluate_pressure_neumann_from_velocity(const VectorType & velocity,
                                          const bool         convective,
                                          const Number       viscosity,
                                          VectorType &       pressure_boundary_vector) const
  {
    velocity.update_ghost_values();

    for(unsigned int cell = 0; cell < cells_at_dirichlet_boundary.size(); ++cell)
    {
      const unsigned int degree = shape_info.data[0].fe_degree;
      if(degree == 2)
        do_pressure_neumann_on_cell<2>(
          cell, velocity, convective, viscosity, pressure_boundary_vector);
      else if(degree == 3)
        do_pressure_neumann_on_cell<3>(
          cell, velocity, convective, viscosity, pressure_boundary_vector);
      else if(degree == 4)
        do_pressure_neumann_on_cell<4>(
          cell, velocity, convective, viscosity, pressure_boundary_vector);
#ifndef DEBUG
      else if(degree == 5)
        do_pressure_neumann_on_cell<5>(
          cell, velocity, convective, viscosity, pressure_boundary_vector);
      else if(degree == 6)
        do_pressure_neumann_on_cell<6>(
          cell, velocity, convective, viscosity, pressure_boundary_vector);
      else if(degree == 7)
        do_pressure_neumann_on_cell<7>(
          cell, velocity, convective, viscosity, pressure_boundary_vector);
      else if(degree == 8)
        do_pressure_neumann_on_cell<8>(
          cell, velocity, convective, viscosity, pressure_boundary_vector);
      else if(degree == 9)
        do_pressure_neumann_on_cell<9>(
          cell, velocity, convective, viscosity, pressure_boundary_vector);
#endif
      else
        AssertThrow(false, ExcMessage("Degree " + std::to_string(degree) + " not instantiated"));
    }
    velocity.zero_out_ghost_values();
  }

  void
  add_pressure_neumann_boundary_to_vector(const VectorType & pressure_boundary_vector,
                                          VectorType &       pressure_vector) const
  {
    const unsigned int mm = shape_info.data[0].fe_degree;
    AssertDimension(mm, pressure_shape_info.data[0].fe_degree + 1);
    const unsigned int data_per_face = Utilities::pow(mm, dim - 1);
    AssertDimension(pressure_boundary_vector.size(),
                    cells_at_dirichlet_boundary.size() * n_lanes * data_per_face);
    for(unsigned int cell = 0; cell < cells_at_dirichlet_boundary.size(); ++cell)
      for(unsigned int v = 0; v < n_lanes && cells_at_dirichlet_boundary[cell].second[v] !=
                                               numbers::invalid_unsigned_int;
          ++v)
      {
        const unsigned int face           = cells_at_dirichlet_boundary[cell].first;
        const unsigned int face_direction = face / 2;
        const unsigned int cell_index     = cells_at_dirichlet_boundary[cell].second[v];
        const unsigned int base           = (cell * n_lanes + v) * data_per_face;
        const unsigned int offset = (face % 2) * Utilities::pow(mm, face_direction) * (mm - 1);
        const unsigned int idx =
          (*pressure_dof_indices)[cell_index / n_lanes][cell_index % n_lanes] + offset;
        if(face_direction == 0)
          for(unsigned int i = 0; i < data_per_face; ++i)
            pressure_vector.local_element(idx + i * mm) +=
              pressure_boundary_vector.local_element(base + i);
        else if(face_direction == 1)
          for(unsigned int i1 = 0; i1 < (dim == 3 ? mm : 1); ++i1)
            for(unsigned int i0 = 0; i0 < mm; ++i0)
              pressure_vector.local_element(idx + i1 * mm * mm + i0) +=
                pressure_boundary_vector.local_element(base + i1 * mm + i0);
        else
          for(unsigned int i = 0; i < data_per_face; ++i)
            pressure_vector.local_element(idx + i) +=
              pressure_boundary_vector.local_element(base + i);
      }
  }

  void
  initialize_vector_pressure_neumann_boundary(VectorType & pressure_boundary_vector) const
  {
    const unsigned int mm            = shape_info.data[0].fe_degree;
    const unsigned int data_per_face = Utilities::pow(mm, dim - 1);

    // create vector without parallel partitioning
    pressure_boundary_vector.reinit(cells_at_dirichlet_boundary.size() * n_lanes * data_per_face);
  }

  void
  evaluate_add_pressure_neumann_from_body_force(const dealii::Function<dim> & rhs_function,
                                                VectorType &                  pressure_rhs) const
  {
    const unsigned int degree   = shape_info.data[0].fe_degree;
    const unsigned int nn       = degree + 1;
    const unsigned int n_points = Utilities::pow(nn, dim - 1);
    AssertDimension(degree, pressure_shape_info.data[0].fe_degree + 1);
    const unsigned int mm = degree;

    AlignedVector<VectorizedArray<Number>> pressure_values(n_points);

    for(unsigned int cell = 0; cell < cells_at_dirichlet_boundary.size(); ++cell)
    {
      const unsigned int face           = cells_at_dirichlet_boundary[cell].first;
      const unsigned int face_direction = face / 2;

      const auto & shape_data_pres = pressure_shape_info.data[0].shape_values_eo;

      if(face_direction < 2)
      {
        std::array<unsigned int, n_lanes> shifted_data_indices, data_indices;
        Assert(cells_at_dirichlet_boundary[cell].second[0] != numbers::invalid_unsigned_int,
               ExcInternalError("Expect at least the first lane to be populated"));
        for(unsigned int v = 0; v < n_lanes; ++v)
        {
          const unsigned int cell_index = cells_at_dirichlet_boundary[cell].second[v];
          const unsigned int idx =
            cell_index != numbers::invalid_unsigned_int ?
              mapping_info
                .face_mapping_data_index[cell_index / n_lanes][face][0][cell_index % n_lanes] :
              data_indices[0];
          shifted_data_indices[v] = idx * 4;
          data_indices[v]         = idx;
        }

        for(unsigned int q1 = 0; q1 < nn; ++q1)
        {
          Tensor<2, 2, VectorizedArray<Number>> jac_xy;
          vectorized_load_and_transpose(4,
                                        &mapping_info.face_jacobians_xy[q1][0][0],
                                        shifted_data_indices.data(),
                                        &jac_xy[0][0]);

          const VectorizedArray<Number> area_element_xy =
            std::sqrt(jac_xy[0][1 - face_direction] * jac_xy[0][1 - face_direction] +
                      jac_xy[1][1 - face_direction] * jac_xy[1][1 - face_direction]);
          const Number h_z = mapping_info.h_z;

          VectorizedArray<Number> val;
          for(unsigned int v = 0; v < n_lanes; ++v)
          {
            Point<dim> point;
            for(unsigned int d = 0; d < 2; ++d)
              point[d] = mapping_info.face_quadrature_points[data_indices[v] + q1][d];
            Point<dim> check = point;
            if constexpr(dim == 3)
              check[2] = point[2] + 0.001;
            val[v] = 0;
            for(unsigned int e = 0; e < 2; ++e)
            {
              Assert(rhs_function.value(point, e) == rhs_function.value(check, e),
                     ExcMessage("Currently, only forces constant in z direction "
                                "are supported"));
              val[v] += rhs_function.value(point, e) *
                        mapping_info.face_normal_vector_xy[data_indices[v] + q1][e];
            }
            if constexpr(dim == 3)
              Assert(rhs_function.value(point, 2) == 0,
                     ExcMessage("Currently, only forces pointing in xy space "
                                "are supported"));
          }

          for(unsigned int qz = 0, q = q1; qz < nn; ++qz, q += nn)
            pressure_values[q] = (mapping_info.quad_weights_z[q1] * area_element_xy * h_z) *
                                 mapping_info.quad_weights_z[qz] * val;
        }
      }
      else
        AssertThrow(false,
                    ExcNotImplemented(
                      "Face operations in extruded direction currently not supported"));

      // perform interpolation within face, utilize the fact that the pressure
      // space is the same as the tangential space in RT
      internal::FEEvaluationImplBasisChange<dealii::internal::evaluate_evenodd,
                                            dealii::internal::EvaluatorQuantity::value,
                                            dim - 1,
                                            0,
                                            0>::do_backward(1,
                                                            shape_data_pres,
                                                            false,
                                                            pressure_values.data(),
                                                            pressure_values.data(),
                                                            mm,
                                                            nn);
      for(unsigned int v = 0; v < n_lanes; ++v)
      {
        const unsigned int cell_index = cells_at_dirichlet_boundary[cell].second[v];
        if(cell_index == numbers::invalid_unsigned_int)
          continue;

        const unsigned int offset = (face % 2) * Utilities::pow(mm, face_direction) * (mm - 1);
        const unsigned int idx =
          (*pressure_dof_indices)[cell_index / n_lanes][cell_index % n_lanes] + offset;
        if(face_direction == 0)
          for(unsigned int i = 0; i < Utilities::pow(mm, dim - 1); ++i)
            pressure_rhs.local_element(idx + i * mm) += pressure_values[i][v];
        else if(face_direction == 1)
          for(unsigned int i1 = 0; i1 < (dim == 3 ? mm : 1); ++i1)
            for(unsigned int i0 = 0; i0 < mm; ++i0)
              pressure_rhs.local_element(idx + i1 * mm * mm + i0) +=
                pressure_values[i1 * mm + i0][v];
        else
          for(unsigned int i = 0; i < Utilities::pow(mm, dim - 1); ++i)
            pressure_rhs.local_element(idx + i) += pressure_values[i][v];
      }
    }
  }

  void
  evaluate_add_velocity_divergence(const VectorType & velocity,
                                   const Number       velocity_scale,
                                   VectorType &       pressure_rhs) const
  {
    AssertDimension(pressure_shape_info.data[0].n_q_points_1d, shape_info.data[0].n_q_points_1d);
    AssertDimension(pressure_shape_info.data[0].fe_degree, shape_info.data[1].fe_degree);

    const unsigned int n_cell_batches = dof_indices.size();

    velocity.update_ghost_values();
    for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
    {
      const unsigned int degree = shape_info.data[0].fe_degree;
      if(degree == 2)
        do_velocity_divergence_on_cell<2>(cell, velocity, velocity_scale, pressure_rhs);
      else if(degree == 3)
        do_velocity_divergence_on_cell<3>(cell, velocity, velocity_scale, pressure_rhs);
      else if(degree == 4)
        do_velocity_divergence_on_cell<4>(cell, velocity, velocity_scale, pressure_rhs);
#ifndef DEBUG
      else if(degree == 5)
        do_velocity_divergence_on_cell<5>(cell, velocity, velocity_scale, pressure_rhs);
      else if(degree == 6)
        do_velocity_divergence_on_cell<6>(cell, velocity, velocity_scale, pressure_rhs);
      else if(degree == 7)
        do_velocity_divergence_on_cell<7>(cell, velocity, velocity_scale, pressure_rhs);
      else if(degree == 8)
        do_velocity_divergence_on_cell<8>(cell, velocity, velocity_scale, pressure_rhs);
      else if(degree == 9)
        do_velocity_divergence_on_cell<9>(cell, velocity, velocity_scale, pressure_rhs);
#endif
      else
        AssertThrow(false, ExcMessage("Degree " + std::to_string(degree) + " not instantiated"));
    }
    velocity.zero_out_ghost_values();
  }

  void
  evaluate_field(const VectorType &                                       velocity_vector,
                 const std::array<unsigned int, n_lanes> &                cell_indices,
                 AlignedVector<Tensor<1, dim, VectorizedArray<Number>>> & velocities,
                 const bool multiply_by_JxW = false) const
  {
    const unsigned int degree = shape_info.data[0].fe_degree;
    if(degree == 2)
      do_evaluate_field_on_cells<2>(velocity_vector, cell_indices, velocities, multiply_by_JxW);
    else if(degree == 3)
      do_evaluate_field_on_cells<3>(velocity_vector, cell_indices, velocities, multiply_by_JxW);
    else if(degree == 4)
      do_evaluate_field_on_cells<4>(velocity_vector, cell_indices, velocities, multiply_by_JxW);
#ifndef DEBUG
    else if(degree == 5)
      do_evaluate_field_on_cells<5>(velocity_vector, cell_indices, velocities, multiply_by_JxW);
    else if(degree == 6)
      do_evaluate_field_on_cells<6>(velocity_vector, cell_indices, velocities, multiply_by_JxW);
    else if(degree == 7)
      do_evaluate_field_on_cells<7>(velocity_vector, cell_indices, velocities, multiply_by_JxW);
    else if(degree == 8)
      do_evaluate_field_on_cells<8>(velocity_vector, cell_indices, velocities, multiply_by_JxW);
    else if(degree == 9)
      do_evaluate_field_on_cells<9>(velocity_vector, cell_indices, velocities, multiply_by_JxW);
#endif
    else
      AssertThrow(false, ExcMessage("Degree " + std::to_string(degree) + " not instantiated"));
  }

private:
  enum class CellOperation
  {
    helmholtz,
    convective,
    convective_and_divergence
  };

  bool                                                             is_test;
  ObserverPointer<const Mapping<dim>>                              mapping;
  ObserverPointer<const DoFHandler<dim>>                           dof_handler;
  std::vector<dealii::ndarray<unsigned int, 2 * dim + 1, n_lanes>> dof_indices;
  std::vector<dealii::ndarray<unsigned int, 2 * dim, n_lanes>>     neighbor_cells;
  mutable std::vector<MPI_Request>                                 mpi_requests;
  mutable AlignedVector<Number>                                    mpi_recv_buffer;
  mutable AlignedVector<Number>                                    mpi_send_buffer;

  // duplicate of the shared-memory communicator passed to reinit(), or
  // MPI_COMM_SELF, used to allocate the vectors in shared memory
  MPI_Comm communicator_sm = MPI_COMM_SELF;

  void
  free_communicator_sm()
  {
    if(communicator_sm != MPI_COMM_SELF)
    {
      const int ierr = MPI_Comm_free(&communicator_sm);
      AssertThrowMPI(ierr);
      communicator_sm = MPI_COMM_SELF;
    }
  }

  // Check that the source vector of a product with face integrals is
  // allocated in shared memory if a shared-memory communicator was given,
  // i.e., created by initialize_dof_vector() or with reinit() from such a
  // vector
  void
  check_source_vector_is_shared(const VectorType & src) const
  {
    if(communicator_sm != MPI_COMM_SELF)
      AssertThrow(src.shared_vector_data().size() ==
                    Utilities::MPI::n_mpi_processes(communicator_sm),
                  ExcMessage("The source vector is not allocated in shared memory; create it "
                             "with initialize_dof_vector() of this operator"));
  }

  // send buffer of the single exchange in shared memory, see reinit()
  MemorySpace::MemorySpaceData<Number, MemorySpace::Host> shared_send_buffer;

  Number *
  single_exchange_send_buffer() const
  {
    return shared_send_buffer.values.size() > 0 ?
             const_cast<Number *>(shared_send_buffer.values.data()) :
             mpi_send_buffer.data();
  }
  Table<2, unsigned char> all_owned_faces;

  std::vector<std::pair<unsigned int, std::array<unsigned int, n_lanes>>>
    cells_at_dirichlet_boundary;

  std::shared_ptr<const Utilities::MPI::Partitioner> partitioner;

  mutable std::array<double, 15> timings;

  // number of calls and accumulated time of the products with the Laplace
  // part for the variants vmult() with functions, vmult_before_on_dst() and
  // vmult_scratch(), and the time waiting for the processes reading the
  // source vector in shared memory, at the end of the products (part of the
  // compress timer) and before the cell loop of vmult_scratch()
  mutable std::array<std::array<double, 2>, 3> product_timings            = {};
  mutable double                               time_wait_readers          = 0;
  mutable double                               time_wait_readers_deferred = 0;

  void
  wait_for_readers(const bool at_end_of_product = true) const
  {
    Timer time;
    single_exchange.wait_until_source_vector_free();
    if(factor_laplace != 0.)
      (at_end_of_product ? time_wait_readers : time_wait_readers_deferred) += time.wall_time();
  }

  Number factor_mass;
  Number factor_laplace;

  internal::MatrixFreeFunctions::ShapeInfo<Number>  shape_info;
  Extruded::MappingInfo<dim, Number>                mapping_info;
  std::array<std::vector<std::array<Number, 2>>, 2> interpolate_quad_to_boundary;

  internal::MatrixFreeFunctions::ShapeInfo<Number>       pressure_shape_info;
  const std::vector<std::array<unsigned int, n_lanes>> * pressure_dof_indices;

  internal::MatrixFreeFunctions::ShapeInfo<Number>  convective_shape_info;
  Extruded::MappingInfo<dim, Number>                convective_mapping_info;
  std::array<std::vector<std::array<Number, 2>>, 2> convective_interpolate_quad_to_boundary;

  AlignedVector<std::array<VectorizedArray<Number>, 2 * dim>> penalty_parameters;

  std::vector<std::pair<unsigned int, unsigned int>> send_data_process;
  std::vector<unsigned int>                          send_data_cell_index;
  std::vector<unsigned char>                         send_data_face_index;

  // Granularity of the pre and post lists. This number is a tradeoff of
  // creating sufficiently big chunks of data (that allows the CPU to do more
  // hardware prefetching and thus amortize loads) and the increase in cache
  // pressure. On the periodic hill (12 processes, AMD Zen 5), 64 made the
  // products with 'before_loop' and 'after_loop' on global vectors (conjugate
  // gradient method, residual) 4-5% faster than 512, as the entries stay in
  // cache between the two functions, without a measurable cost elsewhere.
  static constexpr unsigned int vector_access_chunk_size = 64;

  // scratch array of vmult_scratch() and the offsets of the entities of
  // each cell batch into it, in the same format as dof_indices
  ExaDG::ScratchMatrixVectorProduct<Number>                        scratch;
  std::vector<dealii::ndarray<unsigned int, 2 * dim + 1, n_lanes>> scratch_indices;

  std::vector<unsigned int>                          cell_loop_pre_list_index;
  std::vector<std::pair<unsigned int, unsigned int>> cell_loop_pre_list;
  std::vector<unsigned int>                          cell_loop_mass_pre_list_index;
  std::vector<std::pair<unsigned int, unsigned int>> cell_loop_mass_pre_list;
  std::vector<unsigned int>                          cell_loop_post_list_index;
  std::vector<std::pair<unsigned int, unsigned int>> cell_loop_post_list;

  std::vector<dealii::ndarray<unsigned int, n_lanes, 2>> cell_level_index;

  mutable AlignedVector<Number>                        face_flux_buffer;
  std::vector<ndarray<unsigned int, 2 * dim, n_lanes>> face_flux_buffer_index;
  std::vector<std::array<bool, dim>>                   all_left_face_fluxes_from_buffer;

  // Flux buffer of the face integrals of the convective term at the
  // convective quadrature points, with the same positions of the faces as
  // face_flux_buffer
  mutable AlignedVector<Number>                        convective_face_flux_buffer;
  std::vector<ndarray<unsigned int, 2 * dim, n_lanes>> convective_face_flux_buffer_index;
  bool                                                 use_convective_face_flux_buffer = true;

  SingleExchange::Data<dim, Number> single_exchange;

  // Start and complete the ghost exchange of src for the loops in which src
  // is final upon entry, running 'work' while the messages are in flight
  void
  exchange_ghosts_overlapped(const VectorType & src, const std::function<void()> & work) const
  {
    Timer time;
    if(factor_laplace != 0.)
    {
      const unsigned int n_partners = single_exchange.partners.size();
      check_source_vector_is_shared(src);
      single_exchange.start_receives(mpi_recv_buffer.data(),
                                     src.get_mpi_communicator(),
                                     mpi_requests.data());

      // ghost entities must not contribute to the packed face data
      Assert(src.has_ghost_elements() == false, ExcInternalError());

      dispatch_single_exchange_pack_and_send(src);
      timings[1] += time.wall_time();
      time.restart();

      work();
      timings[4] += time.wall_time();
      time.restart();

      if(n_partners > 0)
      {
        const int ierr = MPI_Waitall(n_partners * 2, mpi_requests.data(), MPI_STATUSES_IGNORE);
        AssertThrowMPI(ierr);
      }
      single_exchange.unpack_and_complete(mpi_recv_buffer.data(),
                                          const_cast<Number *>(src.begin()));
      timings[3] += time.wall_time();
    }
    else
    {
      src.update_ghost_values_start();
      work();
      src.update_ghost_values_finish();
      timings[11] += time.wall_time();
    }
  }

  // Exchange the ghost values of src and the face data of the cells next to
  // remote processes with the single exchange, for the loops in which src
  // is final upon entry
  void
  exchange_ghosts_and_face_data(const VectorType & src) const
  {
    const unsigned int n_partners = single_exchange.partners.size();
    check_source_vector_is_shared(src);
    single_exchange.start_receives(mpi_recv_buffer.data(),
                                   src.get_mpi_communicator(),
                                   mpi_requests.data());

    // ghost entities must not contribute to the packed face data
    Assert(src.has_ghost_elements() == false, ExcInternalError());

    dispatch_single_exchange_pack_and_send(src);

    if(n_partners > 0)
    {
      const int ierr = MPI_Waitall(n_partners * 2, mpi_requests.data(), MPI_STATUSES_IGNORE);
      AssertThrowMPI(ierr);
    }
    single_exchange.unpack_and_complete(mpi_recv_buffer.data(), const_cast<Number *>(src.begin()));

    // the ghost range now holds the ghost values, reset by the caller with
    // zero_out_ghost_values()
    src.set_ghost_state(true);
  }

  void
  dispatch_single_exchange_pack_and_send(const VectorType & src) const
  {
    const unsigned int n_t = shape_info.data[0].fe_degree;
    if(n_t == 2)
      single_exchange_pack_and_send<2>(src);
    else if(n_t == 3)
      single_exchange_pack_and_send<3>(src);
    else if(n_t == 4)
      single_exchange_pack_and_send<4>(src);
#ifndef DEBUG
    else if(n_t == 5)
      single_exchange_pack_and_send<5>(src);
    else if(n_t == 6)
      single_exchange_pack_and_send<6>(src);
    else if(n_t == 7)
      single_exchange_pack_and_send<7>(src);
    else if(n_t == 8)
      single_exchange_pack_and_send<8>(src);
    else if(n_t == 9)
      single_exchange_pack_and_send<9>(src);
#endif
    else
      AssertThrow(false, ExcNotImplemented());
  }

  void
  dispatch_cell_operation(const unsigned int cell, const VectorType & src, VectorType & dst) const
  {
    dispatch_cell_operation(cell, src, dof_indices[cell], dst.begin());
  }

  void
  dispatch_cell_operation(const unsigned int                                          cell,
                          const VectorType &                                          src,
                          const dealii::ndarray<unsigned int, 2 * dim + 1, n_lanes> & dst_indices,
                          Number *                                                    dst) const
  {
    const unsigned int degree = shape_info.data[0].fe_degree;
    if(degree == 2)
      do_cell_operation<2, true>(cell, src, dst_indices, dst);
    else if(degree == 3)
      do_cell_operation<3, true>(cell, src, dst_indices, dst);
    else if(degree == 4)
      do_cell_operation<4, true>(cell, src, dst_indices, dst);
#ifndef DEBUG
    else if(degree == 5)
      do_cell_operation<5, true>(cell, src, dst_indices, dst);
    else if(degree == 6)
      do_cell_operation<6, true>(cell, src, dst_indices, dst);
    else if(degree == 7)
      do_cell_operation<7, true>(cell, src, dst_indices, dst);
    else if(degree == 8)
      do_cell_operation<8, true>(cell, src, dst_indices, dst);
    else if(degree == 9)
      do_cell_operation<9, true>(cell, src, dst_indices, dst);
#endif
    else
      AssertThrow(false, ExcMessage("Degree " + std::to_string(degree) + " not instantiated"));
  }

  // Pack the messages of the single exchange from locally owned entries of
  // src and send them; the face data is evaluated with the regular indices
  // while the ghost range of src is zero, in the layout described in
  // SingleExchange::Data
  template<int n_t>
  void
  single_exchange_pack_and_send(const VectorType & src) const
  {
    single_exchange.wait_until_send_buffer_free();
    const auto & partners    = single_exchange.partners;
    Number *     send_buffer = single_exchange_send_buffer();
    for(unsigned int p = 0; p < partners.size(); ++p)
    {
      const auto & partner = partners[p];
      single_exchange.pack_ghost_section(partner, src.begin(), send_buffer);
      if(!partner.faces_read_directly)
        pack_face_data<n_t>(src.begin(),
                            partner.first_face,
                            partner.first_face + partner.n_faces,
                            single_exchange.face_section(partner, send_buffer));
      single_exchange.pack_supplement_section(partner, src.begin(), send_buffer);
      single_exchange.start_send(p,
                                 send_buffer,
                                 src.get_mpi_communicator(),
                                 mpi_requests.data() + partners.size());
    }
  }

  unsigned int
  n_active_entries_per_cell_batch(const unsigned int cell) const
  {
    AssertIndexRange(cell, dof_indices.size());
    unsigned int index = n_lanes - 1;
    while(cell_level_index[cell][index][0] == numbers::invalid_unsigned_int)
    {
      Assert(index > 0, ExcInternalError());
      --index;
    }
    return index + 1;
  }

  typename DoFHandler<dim>::cell_iterator
  get_cell_iterator(const unsigned int cell, const unsigned int lane) const
  {
    return typename DoFHandler<dim>::cell_iterator(&dof_handler->get_triangulation(),
                                                   cell_level_index[cell][lane][0],
                                                   cell_level_index[cell][lane][1],
                                                   dof_handler.get());
  }

  // For each chunk of 'chunk_size' locally owned vector entries, compute the
  // cell batch index in the loop where the chunk is first read
  // (touched_first_by, including the neighbors accessed by the face
  // integrals), first written (touched_mass_first_by) and last written
  // (touched_last_by). The values n_cell_batches and n_cell_batches + 1
  // indicate entries that need to be processed before the cell loop as they
  // are subject to MPI communication (pre lists) or after the cell loop once
  // the compress step has finished (post list).
  void
  compute_touched_chunks(const unsigned int          chunk_size,
                         std::vector<unsigned int> & touched_first_by,
                         std::vector<unsigned int> & touched_mass_first_by,
                         std::vector<unsigned int> & touched_last_by) const
  {
    const unsigned int n_dofs = partitioner->locally_owned_size();
    touched_first_by.assign((n_dofs + chunk_size - 1) / chunk_size, numbers::invalid_unsigned_int);
    touched_mass_first_by.assign((n_dofs + chunk_size - 1) / chunk_size,
                                 numbers::invalid_unsigned_int);
    touched_last_by.assign((n_dofs + chunk_size - 1) / chunk_size, numbers::invalid_unsigned_int);
    const unsigned int n_cell_batches = dof_indices.size();
    const unsigned int dofs_on_quad   = dof_handler->get_fe().dofs_per_face;
    const unsigned int dofs_on_hex = dof_handler->get_fe().dofs_per_cell - 2 * dim * dofs_on_quad;
    for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
    {
      for(unsigned int face = 0; face < 2 * dim; ++face)
        for(unsigned int v = 0; v < n_lanes; ++v)
        {
          const unsigned int c = neighbor_cells[cell][face][v];
          if(c != numbers::invalid_unsigned_int)
          {
            for(unsigned int i = 0; i < 2 * dim; ++i)
            {
              const unsigned int idx = dof_indices[c / n_lanes][i][c % n_lanes];
              if(idx < n_dofs)
              {
                const unsigned int first = idx / chunk_size;
                const unsigned int last  = (idx + dofs_on_quad - 1) / chunk_size;
                for(unsigned int j = first; j <= last; ++j)
                  if(touched_first_by[j] == numbers::invalid_unsigned_int)
                    touched_first_by[j] = cell;
              }
            }
            const unsigned int idx = dof_indices[c / n_lanes][2 * dim][c % n_lanes];
            if(idx < n_dofs)
            {
              const unsigned int first = idx / chunk_size;
              const unsigned int last  = (idx + dofs_on_hex - 1) / chunk_size;
              for(unsigned int j = first; j <= last; ++j)
                if(touched_first_by[j] == numbers::invalid_unsigned_int)
                  touched_first_by[j] = cell;
            }
          }
        }
      for(unsigned int v = 0; v < n_lanes; ++v)
      {
        for(unsigned int i = 0; i < 2 * dim; ++i)
        {
          const unsigned int idx = dof_indices[cell][i][v];
          if(idx < n_dofs)
          {
            const unsigned int first = idx / chunk_size;
            const unsigned int last  = (idx + dofs_on_quad - 1) / chunk_size;
            for(unsigned int j = first; j <= last; ++j)
            {
              if(touched_first_by[j] == numbers::invalid_unsigned_int)
                touched_first_by[j] = cell;
              if(touched_mass_first_by[j] == numbers::invalid_unsigned_int)
                touched_mass_first_by[j] = cell;
              touched_last_by[j] = cell;
            }
          }
        }
        const unsigned int idx = dof_indices[cell][2 * dim][v];
        if(idx < n_dofs)
        {
          const unsigned int first = idx / chunk_size;
          const unsigned int last  = (idx + dofs_on_hex - 1) / chunk_size;
          for(unsigned int j = first; j <= last; ++j)
          {
            if(touched_first_by[j] == numbers::invalid_unsigned_int)
              touched_first_by[j] = cell;
            if(touched_mass_first_by[j] == numbers::invalid_unsigned_int)
              touched_mass_first_by[j] = cell;
            touched_last_by[j] = cell;
          }
        }
      }
    }

    // ensure that all indices are touched at least during the last round
    for(const auto & index : touched_first_by)
      AssertThrow(index != numbers::invalid_unsigned_int, ExcInternalError());
    for(const auto & index : touched_mass_first_by)
      AssertThrow(index != numbers::invalid_unsigned_int, ExcInternalError());
    for(const auto & index : touched_last_by)
      AssertThrow(index != numbers::invalid_unsigned_int, ExcInternalError());

    // set the import indices in the data to be exchanged via the partitioner
    // to one index higher to indicate that we want to process them first and
    // to two indices higher for the face integrals.
    for(auto it : partitioner->import_indices())
      for(unsigned int i = it.first; i < it.second; ++i)
        touched_first_by[i / chunk_size] = n_cell_batches;
    for(auto it : partitioner->import_indices())
      for(unsigned int i = it.first; i < it.second; ++i)
        touched_mass_first_by[i / chunk_size] = n_cell_batches;

    for(const unsigned int cell_lane : send_data_cell_index)
    {
      const unsigned int cell = cell_lane / n_lanes;
      const unsigned int lane = cell_lane % n_lanes;
      for(unsigned int i = 0; i < 2 * dim; ++i)
      {
        const unsigned int idx = dof_indices[cell][i][lane];
        if(idx < n_dofs)
        {
          const unsigned int first = idx / chunk_size;
          const unsigned int last  = (idx + dofs_on_quad - 1) / chunk_size;
          for(unsigned int j = first; j <= last; ++j)
            if(touched_first_by[j] != n_cell_batches)
              touched_first_by[j] = n_cell_batches + 1;
        }
      }
      const unsigned int idx = dof_indices[cell][2 * dim][lane];
      if(idx < n_dofs)
      {
        const unsigned int first = idx / chunk_size;
        const unsigned int last  = (idx + dofs_on_hex - 1) / chunk_size;
        for(unsigned int j = first; j <= last; ++j)
          if(touched_first_by[j] != n_cell_batches)
            touched_first_by[j] = n_cell_batches + 1;
      }
    }

    // since the write is only on the cell indices, we only need to pick up
    // those indices with access to the vector partitioner in the post list
    for(auto it : partitioner->import_indices())
      for(unsigned int i = it.first; i < it.second; ++i)
        touched_last_by[i / chunk_size] = n_cell_batches;
  }

  void
  compute_vector_access_pattern()
  {
    const unsigned int n_dofs         = partitioner->locally_owned_size();
    const unsigned int n_cell_batches = dof_indices.size();

    const unsigned int        chunk_size = vector_access_chunk_size;
    std::vector<unsigned int> touched_first_by, touched_mass_first_by, touched_last_by;
    compute_touched_chunks(chunk_size, touched_first_by, touched_mass_first_by, touched_last_by);

#if 0
    std::cout << "data locality quality: ";
    for (unsigned int i = 0; i < touched_first_by.size(); ++i)
      std::cout << touched_last_by[i] - touched_first_by[i] << "  ";
    std::cout << std::endl;
#endif

    // debug output to assess sanity of implementation
    if(false)
    {
      unsigned int n_batches_half = 0, n_batches_10 = 0;
      for(unsigned int i = 0; i < touched_first_by.size(); ++i)
        if(touched_last_by[i] - touched_first_by[i] > n_cell_batches / 2)
          ++n_batches_half;
        else if(touched_last_by[i] - touched_first_by[i] > 10)
          ++n_batches_10;

      Helper::print_time(static_cast<double>(n_batches_half) / touched_first_by.size(),
                         "Pre-/post distance > 1/2 size",
                         MPI_COMM_WORLD);
      Helper::print_time(static_cast<double>(n_batches_half + n_batches_10) /
                           touched_first_by.size(),
                         "Pre-/post distance > 10",
                         MPI_COMM_WORLD);
    }

    std::map<unsigned int, std::vector<unsigned int>> chunk_must_do_pre;
    for(unsigned int i = 0; i < touched_first_by.size(); ++i)
      chunk_must_do_pre[touched_first_by[i]].push_back(i);
    Helper::convert_map_to_range_list(n_cell_batches + 2,
                                      chunk_size,
                                      chunk_must_do_pre,
                                      cell_loop_pre_list_index,
                                      cell_loop_pre_list,
                                      n_dofs);

    std::map<unsigned int, std::vector<unsigned int>> chunk_must_do_mass_pre;
    for(unsigned int i = 0; i < touched_mass_first_by.size(); ++i)
      chunk_must_do_mass_pre[touched_mass_first_by[i]].push_back(i);
    Helper::convert_map_to_range_list(n_cell_batches + 1,
                                      chunk_size,
                                      chunk_must_do_mass_pre,
                                      cell_loop_mass_pre_list_index,
                                      cell_loop_mass_pre_list,
                                      n_dofs);

    std::map<unsigned int, std::vector<unsigned int>> chunk_must_do_post;
    for(unsigned int i = 0; i < touched_last_by.size(); ++i)
      chunk_must_do_post[touched_last_by[i]].push_back(i);
    Helper::convert_map_to_range_list(n_cell_batches + 1,
                                      chunk_size,
                                      chunk_must_do_post,
                                      cell_loop_post_list_index,
                                      cell_loop_post_list,
                                      n_dofs);

#if 0
    for (unsigned int i = 0; i < n_cell_batches + 2; ++i)
      {
        std::cout << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) << " pre ";
        for (unsigned int j = cell_loop_pre_list_index[i];
             j < cell_loop_pre_list_index[i + 1];
             ++j)
          std::cout << "[" << cell_loop_pre_list[j].first << ","
                    << cell_loop_pre_list[j].second << ") ";
        if (i < n_cell_batches + 1)
          {
        std::cout << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) << " pre mass ";
        for (unsigned int j = cell_loop_mass_pre_list_index[i];
             j < cell_loop_mass_pre_list_index[i + 1];
             ++j)
          std::cout << "[" << cell_loop_mass_pre_list[j].first << ","
                    << cell_loop_mass_pre_list[j].second << ") ";
        std::cout << "post ";
        for (unsigned int j = cell_loop_post_list_index[i];
             j < cell_loop_post_list_index[i + 1];
             ++j)
          std::cout << "[" << cell_loop_post_list[j].first << ","
                    << cell_loop_post_list[j].second << ")";
        std::cout << std::endl;
      }
      }
#endif
  }

  // Set up the scratch array of vmult_scratch() with the faces and cell
  // interiors of the Raviart-Thomas element as entities
  void
  setup_scratch()
  {
    const unsigned int n_cell_batches = dof_indices.size();
    const unsigned int dofs_on_quad   = dof_handler->get_fe().dofs_per_face;
    const unsigned int dofs_on_hex = dof_handler->get_fe().dofs_per_cell - 2 * dim * dofs_on_quad;
    scratch.reinit(
      partitioner,
      n_cell_batches,
      [&](const unsigned int cell, const auto & add) {
        for(unsigned int e = 0; e < 2 * dim + 1; ++e)
          for(unsigned int v = 0; v < n_active_entries_per_cell_batch(cell); ++v)
            add(dof_indices[cell][e][v], e < 2 * dim ? dofs_on_quad : dofs_on_hex);
      },
      true);

    scratch_indices = dof_indices;
    for(auto & cell_indices : scratch_indices)
      for(auto & entity_indices : cell_indices)
        for(unsigned int & idx : entity_indices)
          idx = scratch.scratch_index(idx);
    scratch.release_setup_data();
  }

  template<int n_t, int n_points_1d>
  void
  read_cell_values(const Number *                                              src_vector,
                   const dealii::ndarray<unsigned int, 2 * dim + 1, n_lanes> & dof_indices,
                   VectorizedArray<Number> *                                   out,
                   VectorizedArray<Number> * dof_values = nullptr) const
  {
    // If 'dof_values' is given, also store the dofs of the cell, transposed
    // into vectorized form, per component in lexicographic order with n_n
    // dofs in the direction of the component and n_t dofs in the other
    // directions, see interpolate_cell_dofs_to_face()
    static_assert(n_t > 1, "Degree 0 not supported");
    constexpr unsigned int n_n                = n_t + 1;
    constexpr unsigned int dofs_per_face      = dealii::Utilities::pow(n_t, dim - 1);
    constexpr unsigned int dofs_per_plane     = n_t * (n_t - 1);
    constexpr unsigned int cell_dofs_per_comp = dofs_per_plane * (dim > 2 ? n_t : 1);

    VectorizedArray<Number> data_f[2][dofs_per_face];
    VectorizedArray<Number> data_cell[dofs_per_plane];
    VectorizedArray<Number> data_points[n_points_1d * n_points_1d];
    VectorizedArray<Number> data_interpolate[4 * n_n];

    constexpr bool do_convective = n_points_1d > (n_t + 1);
    const std::vector<internal::MatrixFreeFunctions::UnivariateShapeData<Number>> & sh_data =
      do_convective ? convective_shape_info.data : shape_info.data;
    AssertDimension(n_points_1d, sh_data[0].n_q_points_1d);
    AssertDimension(n_points_1d, sh_data[1].n_q_points_1d);
    const Number * DEAL_II_RESTRICT shape_data_n = sh_data[0].shape_values_eo.data();
    const Number * DEAL_II_RESTRICT shape_data_t = sh_data[1].shape_values_eo.data();
    for(unsigned int f = 0; f < 2; ++f)
    {
      // check if indices unconstrained
      bool all_indices_unconstrained = true;
      for(const unsigned int i : dof_indices[f])
        if(i == numbers::invalid_unsigned_int)
        {
          all_indices_unconstrained = false;
          break;
        }
      if(all_indices_unconstrained)
        vectorized_load_and_transpose(dofs_per_face, src_vector, dof_indices[f].data(), data_f[f]);
      else
      {
        for(unsigned int i = 0; i < dofs_per_face; ++i)
          data_f[f][i] = 0;
        for(unsigned int v = 0; v < n_lanes; ++v)
        {
          const unsigned int idx = dof_indices[f][v];
          if(idx != numbers::invalid_unsigned_int)
            for(unsigned int i = 0; i < dofs_per_face; ++i)
              data_f[f][i][v] = src_vector[idx + i];
        }
      }
    }
    for(unsigned int i_z = 0; i_z < (dim > 2 ? n_t : 1); ++i_z)
    {
      vectorized_load_and_transpose(dofs_per_plane,
                                    src_vector + i_z * dofs_per_plane,
                                    dof_indices[2 * dim].data(),
                                    data_cell);

      // perform interpolation in x direction
      for(unsigned int i_y = 0; i_y < n_t; ++i_y)
      {
        data_interpolate[0]   = data_f[0][i_z * n_t + i_y];
        data_interpolate[n_t] = data_f[1][i_z * n_t + i_y];
        for(unsigned int i = 0; i < n_t - 1; ++i)
          data_interpolate[1 + i] = data_cell[i_y * (n_t - 1) + i];
        if(dof_values != nullptr)
          for(unsigned int i = 0; i < n_n; ++i)
            dof_values[(i_z * n_t + i_y) * n_n + i] = data_interpolate[i];
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::value,
                                              n_n,
                                              n_points_1d,
                                              1,
                                              1,
                                              true,
                                              false>(shape_data_n,
                                                     data_interpolate,
                                                     data_points + i_y * n_points_1d);
      }

      // perform interpolation in y direction
      VectorizedArray<Number> * out_z = out + i_z * dim * n_points_1d * n_points_1d;
      for(unsigned int i_x = 0; i_x < n_points_1d; ++i_x)
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::value,
                                              n_t,
                                              n_points_1d,
                                              n_points_1d,
                                              dim * n_points_1d,
                                              true,
                                              false>(shape_data_t,
                                                     data_points + i_x,
                                                     out_z + i_x * dim);
    }

    // y component
    for(unsigned int f = 0; f < 2; ++f)
    {
      // check if indices unconstrained
      bool all_indices_unconstrained = true;
      for(const unsigned int i : dof_indices[2 + f])
        if(i == numbers::invalid_unsigned_int)
        {
          all_indices_unconstrained = false;
          break;
        }
      if(all_indices_unconstrained)
        vectorized_load_and_transpose(dofs_per_face,
                                      src_vector,
                                      dof_indices[2 + f].data(),
                                      data_f[f]);
      else
      {
        for(unsigned int i = 0; i < dofs_per_face; ++i)
          data_f[f][i] = 0;
        for(unsigned int v = 0; v < n_lanes; ++v)
        {
          const unsigned int idx = dof_indices[2 + f][v];
          if(idx != numbers::invalid_unsigned_int)
            for(unsigned int i = 0; i < dofs_per_face; ++i)
              data_f[f][i][v] = src_vector[idx + i];
        }
      }
    }
    for(unsigned int i_z = 0; i_z < (dim > 2 ? n_t : 1); ++i_z)
    {
      vectorized_load_and_transpose(dofs_per_plane,
                                    src_vector + cell_dofs_per_comp + i_z * dofs_per_plane,
                                    dof_indices[2 * dim].data(),
                                    data_cell);

      // perform interpolation in y direction
      for(unsigned int i_x = 0; i_x < n_t; ++i_x)
      {
        data_interpolate[0]   = data_f[0][i_z * n_t + i_x];
        data_interpolate[n_t] = data_f[1][i_z * n_t + i_x];
        for(unsigned int i = 0; i < n_t - 1; ++i)
          data_interpolate[1 + i] = data_cell[i * n_t + i_x];
        if(dof_values != nullptr)
          for(unsigned int i = 0; i < n_n; ++i)
            dof_values[n_n * dofs_per_face + (i_z * n_n + i) * n_t + i_x] = data_interpolate[i];
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::value,
                                              n_n,
                                              n_points_1d,
                                              1,
                                              n_t,
                                              true,
                                              false>(shape_data_n,
                                                     data_interpolate,
                                                     data_points + i_x);
      }

      // perform interpolation in x direction
      VectorizedArray<Number> * out_z = out + i_z * dim * n_points_1d * n_points_1d + 1;
      for(unsigned int i_y = 0; i_y < n_points_1d; ++i_y)
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::value,
                                              n_t,
                                              n_points_1d,
                                              1,
                                              dim,
                                              true,
                                              false>(shape_data_t,
                                                     data_points + i_y * n_t,
                                                     out_z + i_y * n_points_1d * dim);
    }
    if constexpr(dim == 3)
    {
      for(unsigned int i = 0; i < n_points_1d * n_points_1d; ++i)
        for(unsigned int j = 0; j < 2; ++j)
        {
          internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                internal::EvaluatorQuantity::value,
                                                n_t,
                                                n_points_1d,
                                                n_points_1d * n_points_1d * dim,
                                                n_points_1d * n_points_1d * dim,
                                                true,
                                                false>(shape_data_t,
                                                       out + i * 3 + j,
                                                       out + i * 3 + j);
        }
      for(unsigned int f = 0; f < 2; ++f)
      {
        bool all_indices_unconstrained = true;
        for(const unsigned int i : dof_indices[4 + f])
          if(i == numbers::invalid_unsigned_int)
          {
            all_indices_unconstrained = false;
            break;
          }
        if(all_indices_unconstrained)
          vectorized_load_and_transpose(dofs_per_face,
                                        src_vector,
                                        dof_indices[4 + f].data(),
                                        data_f[f]);
        else
        {
          for(unsigned int i = 0; i < dofs_per_face; ++i)
            data_f[f][i] = 0;
          for(unsigned int v = 0; v < n_lanes; ++v)
          {
            const unsigned int idx = dof_indices[4 + f][v];
            if(idx != numbers::invalid_unsigned_int)
              for(unsigned int i = 0; i < dofs_per_face; ++i)
                data_f[f][i][v] = src_vector[idx + i];
          }
        }
      }

      AssertThrow(dofs_per_face >= 4, ExcNotImplemented());
      for(unsigned int i2 = 0; i2 < dofs_per_face; i2 += 4)
      {
        const unsigned int i = std::min(i2, dofs_per_face - 4);

        for(unsigned int i1 = 0; i1 < n_t - 1; ++i1)
        {
          vectorized_load_and_transpose(4,
                                        src_vector + 2 * cell_dofs_per_comp + i +
                                          i1 * dofs_per_face,
                                        dof_indices[2 * dim].data(),
                                        data_interpolate + (i1 + 1) * 4);
        }
        for(unsigned int j = 0; j < 4; ++j)
        {
          data_interpolate[j]           = data_f[0][i + j];
          data_interpolate[j + 4 * n_t] = data_f[1][i + j];
          if(dof_values != nullptr)
            for(unsigned int l = 0; l < n_n; ++l)
              dof_values[2 * n_n * dofs_per_face + l * dofs_per_face + i + j] =
                data_interpolate[j + 4 * l];
          internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                internal::EvaluatorQuantity::value,
                                                n_n,
                                                n_points_1d,
                                                4,
                                                n_points_1d * n_points_1d * dim,
                                                true,
                                                false>(shape_data_n,
                                                       data_interpolate + j,
                                                       out + (i + j) * dim + 2);
        }
      }
      for(unsigned int i2 = 0; i2 < n_points_1d; ++i2)
      {
        VectorizedArray<Number> * out_i2 = out + i2 * n_points_1d * n_points_1d * dim + 2;
        for(int i1 = n_t - 1; i1 >= 0; --i1)
          internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                internal::EvaluatorQuantity::value,
                                                n_t,
                                                n_points_1d,
                                                dim,
                                                dim,
                                                true,
                                                false>(shape_data_t,
                                                       out_i2 + i1 * n_t * dim,
                                                       out_i2 + i1 * n_points_1d * dim);
        for(unsigned int i1 = 0; i1 < n_points_1d; ++i1)
          internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                internal::EvaluatorQuantity::value,
                                                n_t,
                                                n_points_1d,
                                                dim * n_points_1d,
                                                dim * n_points_1d,
                                                true,
                                                false>(shape_data_t,
                                                       out_i2 + i1 * dim,
                                                       out_i2 + i1 * dim);
      }
    }
  }

  template<int n_t, int n_points_1d>
  void
  integrate_cell_scatter(const dealii::ndarray<unsigned int, 2 * dim + 1, n_lanes> & dof_indices,
                         const unsigned int                                          n_filled_lanes,
                         VectorizedArray<Number> *                                   quad_values,
                         Number * dst_vector) const
  {
    static_assert(n_t > 1, "Degree 0 not supported");
    constexpr unsigned int n_n                = n_t + 1;
    constexpr unsigned int dofs_per_face      = dealii::Utilities::pow(n_t, dim - 1);
    constexpr unsigned int dofs_per_plane     = n_t * (n_t - 1);
    constexpr unsigned int cell_dofs_per_comp = dofs_per_plane * (dim > 2 ? n_t : 1);

    VectorizedArray<Number> data_f[2][dofs_per_face];
    VectorizedArray<Number> data_cell[dofs_per_plane];
    VectorizedArray<Number> data_points[n_points_1d * n_points_1d];
    VectorizedArray<Number> data_interpolate[n_n];

    constexpr bool do_convective = n_points_1d > (n_t + 1);
    const std::vector<internal::MatrixFreeFunctions::UnivariateShapeData<Number>> & sh_data =
      do_convective ? convective_shape_info.data : shape_info.data;
    AssertDimension(n_points_1d, sh_data[0].n_q_points_1d);
    AssertDimension(n_points_1d, sh_data[1].n_q_points_1d);
    const Number * DEAL_II_RESTRICT shape_data_n = sh_data[0].shape_values_eo.data();
    const Number * DEAL_II_RESTRICT shape_data_t = sh_data[1].shape_values_eo.data();

    if constexpr(dim == 3)
      for(unsigned int i = 0; i < n_points_1d * n_points_1d; ++i)
      {
        for(unsigned int j = 0; j < 2; ++j)
        {
          internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                internal::EvaluatorQuantity::value,
                                                n_t,
                                                n_points_1d,
                                                n_points_1d * n_points_1d * dim,
                                                n_points_1d * n_points_1d * dim,
                                                false,
                                                false>(shape_data_t,
                                                       quad_values + i * 3 + j,
                                                       quad_values + i * 3 + j);
        }
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::value,
                                              n_n,
                                              n_points_1d,
                                              n_points_1d * n_points_1d * dim,
                                              n_points_1d * n_points_1d * dim,
                                              false,
                                              false>(shape_data_n,
                                                     quad_values + i * 3 + 2,
                                                     quad_values + i * 3 + 2);
      }

    // x component
    for(unsigned int i_z = 0; i_z < (dim > 2 ? n_t : 1); ++i_z)
    {
      // perform interpolation in y direction
      VectorizedArray<Number> * in_z = quad_values + i_z * dim * n_points_1d * n_points_1d;
      for(unsigned int i_x = 0; i_x < n_points_1d; ++i_x)
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::value,
                                              n_t,
                                              n_points_1d,
                                              dim * n_points_1d,
                                              n_points_1d,
                                              false,
                                              false>(shape_data_t,
                                                     in_z + i_x * dim,
                                                     data_points + i_x);

      // perform interpolation in x direction
      for(unsigned int i_y = 0; i_y < n_t; ++i_y)
      {
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::value,
                                              n_n,
                                              n_points_1d,
                                              1,
                                              1,
                                              false,
                                              false>(shape_data_n,
                                                     data_points + i_y * n_points_1d,
                                                     data_interpolate);
        data_f[0][i_z * n_t + i_y] = data_interpolate[0];
        data_f[1][i_z * n_t + i_y] = data_interpolate[n_t];
        for(unsigned int i = 0; i < n_t - 1; ++i)
          data_cell[i_y * (n_t - 1) + i] = data_interpolate[1 + i];
      }

      if(n_filled_lanes == n_lanes)
        vectorized_transpose_and_store(true,
                                       dofs_per_plane,
                                       data_cell,
                                       dof_indices[2 * dim].data(),
                                       dst_vector + i_z * dofs_per_plane);
      else
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
        {
          Number * dst_ptr = dst_vector + dof_indices[2 * dim][v] + i_z * dofs_per_plane;
          for(unsigned int i = 0; i < dofs_per_plane; ++i)
            dst_ptr[i] += data_cell[i][v];
        }
    }
    for(unsigned int f = 0; f < 2; ++f)
    {
      // check if indices unconstrained
      bool all_indices_unconstrained = n_filled_lanes == n_lanes;
      for(const unsigned int i : dof_indices[f])
        if(i == numbers::invalid_unsigned_int)
        {
          all_indices_unconstrained = false;
          break;
        }
      if(all_indices_unconstrained)
        vectorized_transpose_and_store(
          true, dofs_per_face, data_f[f], dof_indices[f].data(), dst_vector);
      else
      {
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          if(dof_indices[f][v] != numbers::invalid_unsigned_int)
          {
            Number * dst_ptr = dst_vector + dof_indices[f][v];
            for(unsigned int i = 0; i < dofs_per_face; ++i)
              dst_ptr[i] += data_f[f][i][v];
          }
      }
    }

    // y component
    for(unsigned int i_z = 0; i_z < (dim > 2 ? n_t : 1); ++i_z)
    {
      // perform interpolation in x direction
      VectorizedArray<Number> * in_z = quad_values + i_z * dim * n_points_1d * n_points_1d + 1;
      for(unsigned int i_y = 0; i_y < n_points_1d; ++i_y)
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::value,
                                              n_t,
                                              n_points_1d,
                                              dim,
                                              1,
                                              false,
                                              false>(shape_data_t,
                                                     in_z + i_y * n_points_1d * dim,
                                                     data_points + i_y * n_t);

      // perform interpolation in y direction
      for(unsigned int i_x = 0; i_x < n_t; ++i_x)
      {
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::value,
                                              n_n,
                                              n_points_1d,
                                              n_t,
                                              1,
                                              false,
                                              false>(shape_data_n,
                                                     data_points + i_x,
                                                     data_interpolate);
        data_f[0][i_z * n_t + i_x] = data_interpolate[0];
        data_f[1][i_z * n_t + i_x] = data_interpolate[n_t];
        for(unsigned int i = 0; i < n_t - 1; ++i)
          data_cell[i * n_t + i_x] = data_interpolate[1 + i];
      }

      if(n_filled_lanes == n_lanes)
        vectorized_transpose_and_store(true,
                                       dofs_per_plane,
                                       data_cell,
                                       dof_indices[2 * dim].data(),
                                       dst_vector + cell_dofs_per_comp + i_z * dofs_per_plane);
      else
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
        {
          Number * dst_ptr =
            dst_vector + dof_indices[2 * dim][v] + i_z * dofs_per_plane + cell_dofs_per_comp;
          for(unsigned int i = 0; i < dofs_per_plane; ++i)
            dst_ptr[i] += data_cell[i][v];
        }
    }
    for(unsigned int f = 0; f < 2; ++f)
    {
      // check if indices unconstrained
      bool all_indices_unconstrained = n_filled_lanes == n_lanes;
      for(const unsigned int i : dof_indices[2 + f])
        if(i == numbers::invalid_unsigned_int)
        {
          all_indices_unconstrained = false;
          break;
        }
      if(all_indices_unconstrained)
        vectorized_transpose_and_store(
          true, dofs_per_face, data_f[f], dof_indices[2 + f].data(), dst_vector);
      else
      {
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
          if(dof_indices[2 + f][v] != numbers::invalid_unsigned_int)
          {
            Number * dst_ptr = dst_vector + dof_indices[2 + f][v];
            for(unsigned int i = 0; i < dofs_per_face; ++i)
              dst_ptr[i] += data_f[f][i][v];
          }
      }
    }

    if constexpr(dim == 3)
      for(unsigned int i2 = 0; i2 < n_n; ++i2)
      {
        VectorizedArray<Number> * in_i2 = quad_values + i2 * n_points_1d * n_points_1d * dim + 2;
        for(unsigned int i1 = 0; i1 < n_points_1d; ++i1)
          internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                internal::EvaluatorQuantity::value,
                                                n_t,
                                                n_points_1d,
                                                dim * n_points_1d,
                                                n_points_1d,
                                                false,
                                                false>(shape_data_t,
                                                       in_i2 + i1 * dim,
                                                       data_points + i1);
        for(unsigned int i1 = 0; i1 < n_t; ++i1)
          internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                internal::EvaluatorQuantity::value,
                                                n_t,
                                                n_points_1d,
                                                1,
                                                1,
                                                false,
                                                false>(shape_data_t,
                                                       data_points + i1 * n_points_1d,
                                                       data_points + i1 * n_t);

        if(i2 == 0 || i2 == n_t)
        {
          const unsigned int f = i2 == 0 ? 0 : 1;

          // check if indices unconstrained
          bool all_indices_unconstrained = n_filled_lanes == n_lanes;
          for(const unsigned int i : dof_indices[4 + f])
            if(i == numbers::invalid_unsigned_int)
            {
              all_indices_unconstrained = false;
              break;
            }
          if(all_indices_unconstrained)
            vectorized_transpose_and_store(
              true, dofs_per_face, data_points, dof_indices[4 + f].data(), dst_vector);
          else
          {
            for(unsigned int v = 0; v < n_filled_lanes; ++v)
              if(dof_indices[4 + f][v] != numbers::invalid_unsigned_int)
              {
                Number * dst_ptr = dst_vector + dof_indices[4 + f][v];
                for(unsigned int i = 0; i < dofs_per_face; ++i)
                  dst_ptr[i] += data_points[i][v];
              }
          }
        }
        else
        {
          if(n_filled_lanes == n_lanes)
            vectorized_transpose_and_store(true,
                                           dofs_per_face,
                                           data_points,
                                           dof_indices[2 * dim].data(),
                                           dst_vector + 2 * cell_dofs_per_comp +
                                             (i2 - 1) * dofs_per_face);
          else
            for(unsigned int v = 0; v < n_filled_lanes; ++v)
            {
              Number * dst_ptr = dst_vector + dof_indices[2 * dim][v] + (i2 - 1) * dofs_per_face +
                                 2 * cell_dofs_per_comp;
              for(unsigned int i = 0; i < dofs_per_face; ++i)
                dst_ptr[i] += data_points[i][v];
            }
        }
      }
  }

  // Pointers to the first entry of the entities (faces and cell interior)
  // of the cells in all lanes, given by their indices into the vector
  // starting at 'src', with nullptr for constrained entities. The face
  // reading functions take the entities as pointers, so that they can also
  // read cells of other processes directly in shared memory.
  dealii::ndarray<const Number *, 2 * dim + 1, n_lanes>
  entity_pointers(const Number *                                              src,
                  const dealii::ndarray<unsigned int, 2 * dim + 1, n_lanes> & indices) const
  {
    dealii::ndarray<const Number *, 2 * dim + 1, n_lanes> entities;
    for(unsigned int e = 0; e < 2 * dim + 1; ++e)
      for(unsigned int v = 0; v < n_lanes; ++v)
        entities[e][v] =
          indices[e][v] == numbers::invalid_unsigned_int ? nullptr : src + indices[e][v];
    return entities;
  }

  // Replace the entities of the lanes whose neighbor across 'face' is a
  // cell of another process in the same shared-memory domain by pointers
  // into the source vector of the processes owning the entities
  void
  set_remote_cell_entities(const VectorType &                                      src,
                           const unsigned int                                      cell,
                           const unsigned int                                      face,
                           dealii::ndarray<const Number *, 2 * dim + 1, n_lanes> & entities) const
  {
    const auto & locations   = single_exchange.remote_cell_locations[cell][face];
    const auto & shared_data = src.shared_vector_data();
    for(unsigned int v = 0; v < n_lanes; ++v)
      if(const auto * location = locations[v])
        for(unsigned int e = 0; e < 2 * dim + 1; ++e)
          entities[e][v] = (*location)[e].sm_rank == numbers::invalid_unsigned_int ?
                             nullptr :
                             shared_data[(*location)[e].sm_rank].data() + (*location)[e].offset;
  }

  // Load 'n_entries' entries starting 'shift' entries after the start of
  // the entity in each lane and transpose them into 'out'
  static void
  load_and_transpose(const unsigned int                          n_entries,
                     const std::array<const Number *, n_lanes> & entity,
                     const unsigned int                          shift,
                     VectorizedArray<Number> *                   out)
  {
    std::array<const Number *, n_lanes> pointers;
    for(unsigned int v = 0; v < n_lanes; ++v)
      pointers[v] = entity[v] + shift;
    vectorized_load_and_transpose(n_entries, pointers, out);
  }

  template<int n_t>
  void
  read_face_values(const dealii::ndarray<const Number *, 2 * dim + 1, n_lanes> & entities,
                   const unsigned int                                            face,
                   VectorizedArray<Number> *                                     out) const
  {
    const unsigned int face_direction = face / 2;
    const unsigned int side           = face % 2;

    constexpr unsigned int n_n            = n_t + 1;
    constexpr unsigned int dofs_per_face  = dealii::Utilities::pow(n_t, dim - 1);
    constexpr unsigned int dofs_per_plane = n_t * (n_t - 1);

    VectorizedArray<Number> data_f[2][dofs_per_face];
    VectorizedArray<Number> data_cell[n_t * n_t];

    constexpr unsigned int cell_dofs_per_comp = Utilities::pow(n_t, dim - 1) * (n_t - 1);

    const Number * DEAL_II_RESTRICT shape_data_n =
      shape_info.data[0].shape_data_on_face[side].data();
    const Number * DEAL_II_RESTRICT shape_data_t =
      shape_info.data[1].shape_data_on_face[side].data();

    // The normal component of the velocity is continuous over the face and
    // the shape functions are nodal in the normal direction, so we do not
    // need to read the dofs on the face itself (index 'side'). Only the
    // normal derivative of that component is computed here, and the
    // contribution of the dofs on the face to that derivative is added at
    // the quadrature points in compute_face() from the values of the
    // current cell. The output layout is, in component order, val+grad for
    // tangential components and grad only for the normal component.

    // x component
    for(unsigned int f = 0; f < 2; ++f)
    {
      if(face_direction == 0 && f == side)
        continue;

      // check if indices unconstrained
      bool all_indices_unconstrained = true;
      for(const Number * ptr : entities[f])
        if(ptr == nullptr)
        {
          all_indices_unconstrained = false;
          break;
        }
      if(all_indices_unconstrained)
        load_and_transpose(dofs_per_face, entities[f], 0, data_f[f]);
      else
      {
        for(unsigned int i = 0; i < dofs_per_face; ++i)
          data_f[f][i] = 0;
        for(unsigned int v = 0; v < n_lanes; ++v)
        {
          const Number * ptr = entities[f][v];
          if(ptr != nullptr)
            for(unsigned int i = 0; i < dofs_per_face; ++i)
              data_f[f][i][v] = ptr[i];
        }
      }
    }
    if(face_direction < 2)
    {
      for(unsigned int i2 = 0; i2 < (dim > 2 ? n_t : 1); ++i2)
      {
        load_and_transpose(dofs_per_plane, entities[2 * dim], i2 * dofs_per_plane, data_cell);
        if(face_direction == 0)
        {
          VectorizedArray<Number> * out_grad = out + i2 * n_t;
          for(unsigned int i1 = 0; i1 < n_t; ++i1)
          {
            VectorizedArray<Number> sum_g =
              data_f[1 - side][i2 * n_t + i1] * shape_data_n[n_n + (1 - side) * n_t];
            for(unsigned int i = 0; i < n_t - 1; ++i)
              sum_g += data_cell[i1 * (n_t - 1) + i] * shape_data_n[n_n + i + 1];

            out_grad[i1] = sum_g;
          }
        }
        else
        {
          // face_direction == 1
          VectorizedArray<Number> * out_val  = out + i2 * n_n;
          VectorizedArray<Number> * out_grad = out + n_t * n_n + i2 * n_n;
          for(unsigned int f = 0; f < 2; ++f)
          {
            VectorizedArray<Number> sum_v = data_f[f][i2 * n_t] * shape_data_t[0];
            VectorizedArray<Number> sum_g = data_f[f][i2 * n_t] * shape_data_t[n_t];
            for(unsigned int i = 1; i < n_t; ++i)
            {
              sum_v += data_f[f][i2 * n_t + i] * shape_data_t[i];
              sum_g += data_f[f][i2 * n_t + i] * shape_data_t[n_t + i];
            }
            out_val[f * n_t]  = sum_v;
            out_grad[f * n_t] = sum_g;
          }
          for(unsigned int i1 = 0; i1 < n_t - 1; ++i1)
          {
            VectorizedArray<Number> sum_v = data_cell[i1] * shape_data_t[0];
            VectorizedArray<Number> sum_g = data_cell[i1] * shape_data_t[n_t];
            for(unsigned int i = 1; i < n_t; ++i)
            {
              sum_v += data_cell[i * (n_t - 1) + i1] * shape_data_t[i];
              sum_g += data_cell[i * (n_t - 1) + i1] * shape_data_t[n_t + i];
            }

            out_val[i1 + 1]  = sum_v;
            out_grad[i1 + 1] = sum_g;
          }
        }
      }
    }
    else
    {
      Assert(dim == 3, ExcInternalError());
      Assert(face_direction == 2, ExcInternalError());
      VectorizedArray<Number> * out_val  = out;
      VectorizedArray<Number> * out_grad = out + n_t * n_n;
      for(unsigned int f = 0; f < 2; ++f)
        for(unsigned int i2 = 0; i2 < n_t; ++i2)
        {
          VectorizedArray<Number> sum_v = data_f[f][i2] * shape_data_t[0];
          VectorizedArray<Number> sum_g = data_f[f][i2] * shape_data_t[n_t];
          for(unsigned int i = 1; i < n_t; ++i)
          {
            sum_v += data_f[f][i2 + i * n_t] * shape_data_t[i];
            sum_g += data_f[f][i2 + i * n_t] * shape_data_t[n_t + i];
          }
          out_val[i2 * n_n + f * n_t]  = sum_v;
          out_grad[i2 * n_n + f * n_t] = sum_g;
        }
      std::array<unsigned int, dofs_per_plane> indices_of_cell_data;
      for(unsigned int i2 = 0; i2 < n_t; ++i2)
        for(unsigned int i1 = 0; i1 < n_t - 1; ++i1)
          indices_of_cell_data[i2 * (n_t - 1) + i1] = i2 * (n_t + 1) + i1 + 1;
      if constexpr(dofs_per_plane >= 4)
        for(unsigned int i2 = 0; i2 < dofs_per_plane; i2 += 4)
        {
          const unsigned int      i = std::min(i2, dofs_per_plane - 4);
          VectorizedArray<Number> entries[4];
          load_and_transpose(4, entities[2 * dim], i, entries);
          std::array<VectorizedArray<Number>, 4> sum_v, sum_g;
          for(unsigned int e = 0; e < 4; ++e)
          {
            sum_v[e] = entries[e] * shape_data_t[0];
            sum_g[e] = entries[e] * shape_data_t[n_t];
          }
          for(unsigned int i1 = 1; i1 < n_t; ++i1)
          {
            load_and_transpose(4, entities[2 * dim], i + i1 * dofs_per_plane, entries);
            for(unsigned int e = 0; e < 4; ++e)
            {
              sum_v[e] += entries[e] * shape_data_t[i1];
              sum_g[e] += entries[e] * shape_data_t[n_t + i1];
            }
          }
          for(unsigned int e = 0; e < 4; ++e)
          {
            out_val[indices_of_cell_data[i + e]]  = sum_v[e];
            out_grad[indices_of_cell_data[i + e]] = sum_g[e];
          }
        }
      else
      {
        VectorizedArray<Number> tmp[cell_dofs_per_comp];
        load_and_transpose(cell_dofs_per_comp, entities[2 * dim], 0, tmp);
        for(unsigned int i2 = 0; i2 < dofs_per_plane; ++i2)
        {
          VectorizedArray<Number> sum_v = tmp[i2] * shape_data_t[0];
          VectorizedArray<Number> sum_g = tmp[i2] * shape_data_t[n_t];
          for(unsigned int i1 = 1; i1 < n_t; ++i1)
          {
            sum_v += tmp[i2 + i1 * dofs_per_plane] * shape_data_t[i1];
            sum_g += tmp[i2 + i1 * dofs_per_plane] * shape_data_t[n_t + i1];
          }
          out_val[indices_of_cell_data[i2]]  = sum_v;
          out_grad[indices_of_cell_data[i2]] = sum_g;
        }
      }
    }

    out += (face_direction == 0) ? n_t * n_t : 2 * n_t * n_n;

    // y component
    for(unsigned int f = 0; f < 2; ++f)
    {
      if(face_direction == 1 && f == side)
        continue;

      // check if indices unconstrained
      bool all_indices_unconstrained = true;
      for(const Number * ptr : entities[2 + f])
        if(ptr == nullptr)
        {
          all_indices_unconstrained = false;
          break;
        }
      if(all_indices_unconstrained)
        load_and_transpose(dofs_per_face, entities[f + 2], 0, data_f[f]);
      else
      {
        for(unsigned int i = 0; i < dofs_per_face; ++i)
          data_f[f][i] = 0;
        for(unsigned int v = 0; v < n_lanes; ++v)
        {
          const Number * ptr = entities[f + 2][v];
          if(ptr != nullptr)
            for(unsigned int i = 0; i < dofs_per_face; ++i)
              data_f[f][i][v] = ptr[i];
        }
      }
    }
    if(face_direction < 2)
    {
      for(unsigned int i2 = 0; i2 < (dim > 2 ? n_t : 1); ++i2)
      {
        load_and_transpose(dofs_per_plane,
                           entities[2 * dim],
                           cell_dofs_per_comp + i2 * dofs_per_plane,
                           data_cell);
        if(face_direction == 0)
        {
          VectorizedArray<Number> * out_val  = out + i2 * n_n;
          VectorizedArray<Number> * out_grad = out + n_t * n_n + i2 * n_n;
          for(unsigned int f = 0; f < 2; ++f)
          {
            VectorizedArray<Number> sum_v = data_f[f][i2 * n_t] * shape_data_t[0];
            VectorizedArray<Number> sum_g = data_f[f][i2 * n_t] * shape_data_t[n_t];
            for(unsigned int i = 1; i < n_t; ++i)
            {
              sum_v += data_f[f][i2 * n_t + i] * shape_data_t[i];
              sum_g += data_f[f][i2 * n_t + i] * shape_data_t[n_t + i];
            }
            out_val[f * n_t]  = sum_v;
            out_grad[f * n_t] = sum_g;
          }
          for(unsigned int i1 = 0; i1 < n_t - 1; ++i1)
          {
            VectorizedArray<Number> sum_v = data_cell[i1 * n_t] * shape_data_t[0];
            VectorizedArray<Number> sum_g = data_cell[i1 * n_t] * shape_data_t[n_t];
            for(unsigned int i = 1; i < n_t; ++i)
            {
              sum_v += data_cell[i1 * n_t + i] * shape_data_t[i];
              sum_g += data_cell[i1 * n_t + i] * shape_data_t[n_t + i];
            }

            out_val[i1 + 1]  = sum_v;
            out_grad[i1 + 1] = sum_g;
          }
        }
        else
        {
          // face_direction == 1
          VectorizedArray<Number> * out_grad = out + i2 * n_t;
          for(unsigned int i1 = 0; i1 < n_t; ++i1)
          {
            VectorizedArray<Number> sum_g =
              data_f[1 - side][i2 * n_t + i1] * shape_data_n[n_n + (1 - side) * n_t];
            for(unsigned int i = 0; i < n_t - 1; ++i)
              sum_g += data_cell[i * n_t + i1] * shape_data_n[n_n + i + 1];

            out_grad[i1] = sum_g;
          }
        }
      }
    }
    else
    {
      Assert(dim == 3, ExcInternalError());
      Assert(face_direction == 2, ExcInternalError());
      VectorizedArray<Number> * out_val  = out;
      VectorizedArray<Number> * out_grad = out + n_t * n_n;
      for(unsigned int f = 0; f < 2; ++f)
        for(unsigned int i2 = 0; i2 < n_t; ++i2)
        {
          VectorizedArray<Number> sum_v = data_f[f][i2] * shape_data_t[0];
          VectorizedArray<Number> sum_g = data_f[f][i2] * shape_data_t[n_t];
          for(unsigned int i = 1; i < n_t; ++i)
          {
            sum_v += data_f[f][i2 + i * n_t] * shape_data_t[i];
            sum_g += data_f[f][i2 + i * n_t] * shape_data_t[n_t + i];
          }
          out_val[i2 + f * n_t * n_t]  = sum_v;
          out_grad[i2 + f * n_t * n_t] = sum_g;
        }
      std::array<unsigned int, dofs_per_plane> indices_of_cell_data;
      for(unsigned int i2 = 0; i2 < n_t - 1; ++i2)
        for(unsigned int i1 = 0; i1 < n_t; ++i1)
          indices_of_cell_data[i2 * n_t + i1] = (i2 + 1) * n_t + i1;
      if(dofs_per_plane >= 4)
        for(unsigned int i2 = 0; i2 < dofs_per_plane; i2 += 4)
        {
          const unsigned int      i = std::min(i2, dofs_per_plane - 4);
          VectorizedArray<Number> entries[4];
          load_and_transpose(4, entities[2 * dim], cell_dofs_per_comp + i, entries);
          std::array<VectorizedArray<Number>, 4> sum_v, sum_g;
          for(unsigned int e = 0; e < 4; ++e)
          {
            sum_v[e] = entries[e] * shape_data_t[0];
            sum_g[e] = entries[e] * shape_data_t[n_t];
          }
          for(unsigned int i1 = 1; i1 < n_t; ++i1)
          {
            load_and_transpose(4,
                               entities[2 * dim],
                               cell_dofs_per_comp + i + i1 * dofs_per_plane,
                               entries);
            for(unsigned int e = 0; e < 4; ++e)
            {
              sum_v[e] += entries[e] * shape_data_t[i1];
              sum_g[e] += entries[e] * shape_data_t[n_t + i1];
            }
          }
          for(unsigned int e = 0; e < 4; ++e)
          {
            out_val[indices_of_cell_data[i + e]]  = sum_v[e];
            out_grad[indices_of_cell_data[i + e]] = sum_g[e];
          }
        }
      else
      {
        VectorizedArray<Number> tmp[cell_dofs_per_comp];
        load_and_transpose(cell_dofs_per_comp, entities[2 * dim], cell_dofs_per_comp, tmp);
        for(unsigned int i2 = 0; i2 < dofs_per_plane; ++i2)
        {
          VectorizedArray<Number> sum_v = tmp[i2] * shape_data_t[0];
          VectorizedArray<Number> sum_g = tmp[i2] * shape_data_t[n_t];
          for(unsigned int i1 = 1; i1 < n_t; ++i1)
          {
            sum_v += tmp[i2 + i1 * dofs_per_plane] * shape_data_t[i1];
            sum_g += tmp[i2 + i1 * dofs_per_plane] * shape_data_t[n_t + i1];
          }
          out_val[indices_of_cell_data[i2]]  = sum_v;
          out_grad[indices_of_cell_data[i2]] = sum_g;
        }
      }
    }

    // z component
    out += (face_direction == 1) ? n_t * n_t : 2 * n_t * n_n;
    if(dim > 2)
    {
      if(face_direction < 2)
      {
        for(unsigned int f = 0; f < 2; ++f)
        {
          // check if indices unconstrained
          bool all_indices_unconstrained = true;
          for(const Number * ptr : entities[4 + f])
            if(ptr == nullptr)
            {
              all_indices_unconstrained = false;
              break;
            }
          if(all_indices_unconstrained)
            load_and_transpose(dofs_per_face, entities[f + 4], 0, data_cell);
          else
          {
            for(unsigned int i = 0; i < dofs_per_face; ++i)
              data_cell[i] = 0;
            for(unsigned int v = 0; v < n_lanes; ++v)
            {
              const Number * ptr = entities[f + 4][v];
              if(ptr != nullptr)
                for(unsigned int i = 0; i < dofs_per_face; ++i)
                  data_cell[i][v] = ptr[i];
            }
          }
          if(face_direction == 0)
            for(unsigned int i2 = 0; i2 < n_t; ++i2)
            {
              VectorizedArray<Number> sum_v = data_cell[i2 * n_t] * shape_data_t[0];
              VectorizedArray<Number> sum_g = data_cell[i2 * n_t] * shape_data_t[n_t];
              for(unsigned int i = 1; i < n_t; ++i)
              {
                sum_v += data_cell[i2 * n_t + i] * shape_data_t[i];
                sum_g += data_cell[i2 * n_t + i] * shape_data_t[n_t + i];
              }
              out[f * n_t * n_t + i2]             = sum_v;
              out[n_n * n_t + f * n_t * n_t + i2] = sum_g;
            }
          else
            for(unsigned int i2 = 0; i2 < n_t; ++i2)
            {
              VectorizedArray<Number> sum_v = data_cell[i2] * shape_data_t[0];
              VectorizedArray<Number> sum_g = data_cell[i2] * shape_data_t[n_t];
              for(unsigned int i = 1; i < n_t; ++i)
              {
                sum_v += data_cell[i2 + i * n_t] * shape_data_t[i];
                sum_g += data_cell[i2 + i * n_t] * shape_data_t[n_t + i];
              }
              out[f * n_t * n_t + i2]             = sum_v;
              out[n_n * n_t + f * n_t * n_t + i2] = sum_g;
            }
        }
        for(unsigned int i2 = 0; i2 < n_t - 1; ++i2)
        {
          load_and_transpose(dofs_per_face,
                             entities[2 * dim],
                             2 * cell_dofs_per_comp + i2 * dofs_per_face,
                             data_cell);
          if(face_direction == 0)
            for(unsigned int i1 = 0; i1 < n_t; ++i1)
            {
              VectorizedArray<Number> sum_v = data_cell[i1 * n_t] * shape_data_t[0];
              VectorizedArray<Number> sum_g = data_cell[i1 * n_t] * shape_data_t[n_t];
              for(unsigned int i = 1; i < n_t; ++i)
              {
                sum_v += data_cell[i1 * n_t + i] * shape_data_t[i];
                sum_g += data_cell[i1 * n_t + i] * shape_data_t[n_t + i];
              }
              out[(i2 + 1) * n_t + i1]             = sum_v;
              out[n_t * n_n + (i2 + 1) * n_t + i1] = sum_g;
            }
          else
            for(unsigned int i1 = 0; i1 < n_t; ++i1)
            {
              VectorizedArray<Number> sum_v = data_cell[i1] * shape_data_t[0];
              VectorizedArray<Number> sum_g = data_cell[i1] * shape_data_t[n_t];
              for(unsigned int i = 1; i < n_t; ++i)
              {
                sum_v += data_cell[i1 + i * n_t] * shape_data_t[i];
                sum_g += data_cell[i1 + i * n_t] * shape_data_t[n_t + i];
              }
              out[(i2 + 1) * n_t + i1]             = sum_v;
              out[n_t * n_n + (i2 + 1) * n_t + i1] = sum_g;
            }
        }
      }
      else
      {
        // face_direction == 2
        const unsigned int f = 1 - side;
        {
          // check if indices unconstrained
          bool all_indices_unconstrained = true;
          for(const Number * ptr : entities[4 + f])
            if(ptr == nullptr)
            {
              all_indices_unconstrained = false;
              break;
            }
          if(all_indices_unconstrained)
            load_and_transpose(dofs_per_face, entities[f + 4], 0, data_f[f]);
          else
          {
            for(unsigned int i = 0; i < dofs_per_face; ++i)
              data_f[f][i] = 0;
            for(unsigned int v = 0; v < n_lanes; ++v)
            {
              const Number * ptr = entities[f + 4][v];
              if(ptr != nullptr)
                for(unsigned int i = 0; i < dofs_per_face; ++i)
                  data_f[f][i][v] = ptr[i];
            }
          }
        }
        AssertThrow(dofs_per_face >= 4, ExcNotImplemented());
        for(unsigned int i2 = 0; i2 < dofs_per_face; i2 += 4)
        {
          const unsigned int                     i = std::min(i2, dofs_per_face - 4);
          std::array<VectorizedArray<Number>, 4> sum_g;
          for(unsigned int e = 0; e < 4; ++e)
            sum_g[e] = data_f[f][i + e] * shape_data_n[n_n + f * n_t];
          for(unsigned int i1 = 0; i1 < n_t - 1; ++i1)
          {
            VectorizedArray<Number> entries[4];
            load_and_transpose(4,
                               entities[2 * dim],
                               2 * cell_dofs_per_comp + i + i1 * dofs_per_face,
                               entries);
            for(unsigned int e = 0; e < 4; ++e)
              sum_g[e] += entries[e] * shape_data_n[n_n + i1 + 1];
          }
          for(unsigned int e = 0; e < 4; ++e)
            out[i + e] = sum_g[e];
        }
      }
    }
  }

  // Whether compute_face_convective() computes the data on the interior side
  // of the faces from the dofs of the cell (stored by read_cell_values())
  // rather than from the values at the cell quadrature points. The latter
  // is faster for low degrees.
  static constexpr bool
  use_cell_dofs_for_faces(const int degree)
  {
    return degree > 3;
  }

  // Compute the data on face 'face' of the current cell from its dofs
  // 'dof_values' as stored by read_cell_values(). With derivatives, the
  // layout is the one of read_face_values(), followed by the values of the
  // normal component. Different from read_face_values(), the normal
  // derivative of the normal component includes the contribution of the
  // dofs on the face. Without derivatives, the values of the two tangential
  // components (as for read_face_values_only()) are followed by the values
  // of the normal component.
  template<int n_t, int face_direction, bool with_derivatives>
  void
  interpolate_cell_dofs_to_face(const VectorizedArray<Number> * dof_values,
                                const unsigned int              side,
                                VectorizedArray<Number> *       out) const
  {
    static_assert(dim == 3, "Only implemented for 3D");
    constexpr unsigned int n_n           = n_t + 1;
    constexpr unsigned int dofs_per_comp = n_t * n_t * n_n;

    const Number * DEAL_II_RESTRICT shape_data_n =
      shape_info.data[0].shape_data_on_face[side].data();
    const Number * DEAL_II_RESTRICT shape_data_t =
      shape_info.data[1].shape_data_on_face[side].data();

    // the two directions within the face, the first one running fastest
    constexpr unsigned int dir_a = face_direction == 0 ? 1 : 0;
    constexpr unsigned int dir_b = face_direction == 2 ? 1 : 2;

    VectorizedArray<Number> * out_normal_values =
      out + (with_derivatives ? n_t * n_t + 4 * n_t * n_n : 2 * n_t * n_n);

    const auto do_component = [&](auto component) {
      constexpr unsigned int c          = decltype(component)::value;
      constexpr unsigned int n_x        = c == 0 ? n_n : n_t;
      constexpr unsigned int n_y        = c == 1 ? n_n : n_t;
      constexpr unsigned int strides[3] = {1, n_x, n_x * n_y};
      constexpr unsigned int stride_n   = strides[face_direction];
      constexpr unsigned int stride_a   = strides[dir_a];
      constexpr unsigned int stride_b   = strides[dir_b];

      const VectorizedArray<Number> * dofs = dof_values + c * dofs_per_comp;
      if constexpr(c == face_direction)
      {
        // value from the dofs on the face (nodal in normal direction), and
        // normal derivative
        for(unsigned int i_b = 0; i_b < n_t; ++i_b)
          for(unsigned int i_a = 0; i_a < n_t; ++i_a)
          {
            const VectorizedArray<Number> * line = dofs + i_b * stride_b + i_a * stride_a;
            out_normal_values[i_b * n_t + i_a]   = line[side * n_t * stride_n];
            if constexpr(with_derivatives)
            {
              VectorizedArray<Number> sum_g = line[0] * shape_data_n[n_n];
              for(unsigned int i = 1; i < n_n; ++i)
                sum_g += line[i * stride_n] * shape_data_n[n_n + i];
              out[i_b * n_t + i_a] = sum_g;
            }
          }
        if constexpr(with_derivatives)
          out += n_t * n_t;
      }
      else
      {
        constexpr unsigned int n_a = c == dir_a ? n_n : n_t;
        constexpr unsigned int n_b = c == dir_b ? n_n : n_t;
        for(unsigned int i_b = 0; i_b < n_b; ++i_b)
          for(unsigned int i_a = 0; i_a < n_a; ++i_a)
          {
            const VectorizedArray<Number> * line  = dofs + i_b * stride_b + i_a * stride_a;
            VectorizedArray<Number>         sum_v = line[0] * shape_data_t[0];
            VectorizedArray<Number>         sum_g = line[0] * shape_data_t[n_t];
            for(unsigned int i = 1; i < n_t; ++i)
            {
              sum_v += line[i * stride_n] * shape_data_t[i];
              if constexpr(with_derivatives)
                sum_g += line[i * stride_n] * shape_data_t[n_t + i];
            }
            out[i_b * n_a + i_a] = sum_v;
            if constexpr(with_derivatives)
              out[n_t * n_n + i_b * n_a + i_a] = sum_g;
          }
        out += (with_derivatives ? 2 : 1) * n_t * n_n;
      }
    };
    do_component(std::integral_constant<unsigned int, 0>());
    do_component(std::integral_constant<unsigned int, 1>());
    do_component(std::integral_constant<unsigned int, 2>());
  }

  void
  write_vectorized_value(const Number              in,
                         const unsigned int        lane,
                         VectorizedArray<Number> & out) const
  {
    AssertIndexRange(lane, n_lanes);
    out[lane] = in;
  }

  void
  write_vectorized_value(const Number in, const unsigned int, Number & out) const
  {
    out = in;
  }

  template<int n_t, typename Number2>
  void
  read_face_values_only(const dealii::ndarray<const Number *, 2 * dim + 1, n_lanes> & entities,
                        const unsigned int                                            face,
                        const unsigned int                                            lane,
                        Number2 *                                                     out) const
  {
    constexpr unsigned int n_n               = n_t + 1;
    constexpr unsigned int hex_dofs_per_comp = Utilities::pow(n_t, dim - 1) * (n_t - 1);

    // The normal component of the velocity is continuous over the face, so
    // its value is taken from the current cell in compute_face_convective()
    // and we only read the dim-1 tangential components here, in component
    // order
    if(face / 2 == 0)
    {
      const unsigned int start = (face % 2) * (n_t - 1);
      const Number *     ptr   = entities[2][lane];
      unsigned int       idx   = start;
      if(ptr != nullptr)
        for(unsigned int i = 0; i < (dim == 3 ? n_t : 1); ++i)
          write_vectorized_value(ptr[idx + i * n_t], lane, out[i * n_n]);
      else
        for(unsigned int i = 0; i < (dim == 3 ? n_t : 1); ++i)
          write_vectorized_value(0.0, lane, out[i * n_n]);
      ptr = entities[3][lane];
      idx = start;
      if(ptr != nullptr)
        for(unsigned int i = 0; i < (dim == 3 ? n_t : 1); ++i)
          write_vectorized_value(ptr[idx + i * n_t], lane, out[i * n_n + n_t]);
      else
        for(unsigned int i = 0; i < (dim == 3 ? n_t : 1); ++i)
          write_vectorized_value(0.0, lane, out[i * n_n + n_t]);
      ptr = entities[2 * dim][lane];
      idx = hex_dofs_per_comp + start;
      for(unsigned int i1 = 0; i1 < (dim == 3 ? n_t : 1); ++i1)
        for(unsigned int i0 = 0; i0 < n_t - 1; ++i0)
          write_vectorized_value(ptr[idx + (i1 * (n_t - 1) + i0) * n_t],
                                 lane,
                                 out[i1 * n_n + i0 + 1]);
      out += Utilities::pow(n_t, dim - 2) * n_n;

      if constexpr(dim == 3)
      {
        ptr = entities[4][lane];
        idx = start;
        if(ptr != nullptr)
          for(unsigned int i = 0; i < n_t; ++i)
            write_vectorized_value(ptr[idx + i * n_t], lane, out[i]);
        else
          for(unsigned int i = 0; i < n_t; ++i)
            write_vectorized_value(0.0, lane, out[i]);
        ptr = entities[5][lane];
        idx = start;
        if(ptr != nullptr)
          for(unsigned int i = 0; i < n_t; ++i)
            write_vectorized_value(ptr[idx + i * n_t], lane, out[n_t * n_t + i]);
        else
          for(unsigned int i = 0; i < n_t; ++i)
            write_vectorized_value(0.0, lane, out[n_t * n_t + i]);
        ptr = entities[2 * dim][lane];
        idx = 2 * hex_dofs_per_comp + start;
        for(unsigned int i1 = 0; i1 < n_t - 1; ++i1)
          for(unsigned int i0 = 0; i0 < n_t; ++i0)
            write_vectorized_value(ptr[idx + (i1 * n_t + i0) * n_t],
                                   lane,
                                   out[(i1 + 1) * n_t + i0]);
      }
    }
    else if(face / 2 == 1)
    {
      const unsigned int start = (face % 2) * (n_t - 1);
      const Number *     ptr   = entities[0][lane];
      unsigned int       idx   = start;
      if(ptr != nullptr)
        for(unsigned int i = 0; i < (dim == 3 ? n_t : 1); ++i)
          write_vectorized_value(ptr[idx + i * n_t], lane, out[i * n_n]);
      else
        for(unsigned int i = 0; i < (dim == 3 ? n_t : 1); ++i)
          write_vectorized_value(0.0, lane, out[i * n_n]);
      ptr = entities[1][lane];
      idx = start;
      if(ptr != nullptr)
        for(unsigned int i = 0; i < (dim == 3 ? n_t : 1); ++i)
          write_vectorized_value(ptr[idx + i * n_t], lane, out[i * n_n + n_t]);
      else
        for(unsigned int i = 0; i < (dim == 3 ? n_t : 1); ++i)
          write_vectorized_value(0.0, lane, out[i * n_n + n_t]);
      ptr = entities[2 * dim][lane];
      idx = start * (n_t - 1);
      for(unsigned int i1 = 0; i1 < (dim == 3 ? n_t : 1); ++i1)
        for(unsigned int i0 = 0; i0 < n_t - 1; ++i0)
          write_vectorized_value(ptr[idx + i1 * (n_t - 1) * n_t + i0],
                                 lane,
                                 out[i1 * n_n + i0 + 1]);
      out += Utilities::pow(n_t, dim - 2) * n_n;

      if constexpr(dim == 3)
      {
        ptr = entities[4][lane];
        idx = start * n_t;
        if(ptr != nullptr)
          for(unsigned int i = 0; i < n_t; ++i)
            write_vectorized_value(ptr[idx + i], lane, out[i]);
        else
          for(unsigned int i = 0; i < n_t; ++i)
            write_vectorized_value(0.0, lane, out[i]);
        ptr = entities[5][lane];
        idx = start * n_t;
        if(ptr != nullptr)
          for(unsigned int i = 0; i < n_t; ++i)
            write_vectorized_value(ptr[idx + i], lane, out[n_t * n_t + i]);
        else
          for(unsigned int i = 0; i < n_t; ++i)
            write_vectorized_value(0.0, lane, out[n_t * n_t + i]);
        ptr = entities[2 * dim][lane];
        idx = 2 * hex_dofs_per_comp + start * n_t;
        for(unsigned int i1 = 0; i1 < n_t - 1; ++i1)
          for(unsigned int i0 = 0; i0 < n_t; ++i0)
            write_vectorized_value(ptr[idx + i1 * n_t * n_t + i0], lane, out[(i1 + 1) * n_t + i0]);
      }
    }
    else
    {
      AssertDimension(dim, 3);
      const unsigned int start = (face % 2) * (n_t - 1) * n_t;
      const Number *     ptr   = entities[0][lane];
      unsigned int       idx   = start;
      if(ptr != nullptr)
        for(unsigned int i = 0; i < n_t; ++i)
          write_vectorized_value(ptr[idx + i], lane, out[i * n_n]);
      else
        for(unsigned int i = 0; i < n_t; ++i)
          write_vectorized_value(0.0, lane, out[i * n_n]);
      ptr = entities[1][lane];
      idx = start;
      if(ptr != nullptr)
        for(unsigned int i = 0; i < n_t; ++i)
          write_vectorized_value(ptr[idx + i], lane, out[i * n_n + n_t]);
      else
        for(unsigned int i = 0; i < n_t; ++i)
          write_vectorized_value(0.0, lane, out[i * n_n + n_t]);
      ptr = entities[2 * dim][lane];
      idx = start * (n_t - 1);
      for(unsigned int i1 = 0; i1 < n_t; ++i1)
        for(unsigned int i0 = 0; i0 < n_t - 1; ++i0)
          write_vectorized_value(ptr[idx + i1 * (n_t - 1) + i0], lane, out[i1 * n_n + i0 + 1]);
      out += Utilities::pow(n_t, dim - 2) * n_n;

      ptr = entities[2][lane];

      idx = start;
      if(ptr != nullptr)
        for(unsigned int i = 0; i < n_t; ++i)
          write_vectorized_value(ptr[idx + i], lane, out[i]);
      else
        for(unsigned int i = 0; i < n_t; ++i)
          write_vectorized_value(0.0, lane, out[i]);
      ptr = entities[3][lane];
      idx = start;
      if(ptr != nullptr)
        for(unsigned int i = 0; i < n_t; ++i)
          write_vectorized_value(ptr[idx + i], lane, out[i + n_t * n_t]);
      else
        for(unsigned int i = 0; i < n_t; ++i)
          write_vectorized_value(0.0, lane, out[i + n_t * n_t]);
      ptr = entities[2 * dim][lane];
      idx = hex_dofs_per_comp + start * (n_t - 1);
      for(unsigned int i1 = 0; i1 < n_t - 1; ++i1)
        for(unsigned int i0 = 0; i0 < n_t; ++i0)
          write_vectorized_value(ptr[idx + i1 * n_t + i0], lane, out[(i1 + 1) * n_t + i0]);
    }
  }

  // Evaluate the face data (values and derivatives) of the faces
  // [first_face, end_face) of send_data_cell_index and send_data_face_index
  // and store them contiguously starting at 'out', vectorized over faces
  // with the same face number
  template<int n_t>
  void
  pack_face_data(const Number *     src,
                 const unsigned int first_face,
                 const unsigned int end_face,
                 Number *           out) const
  {
    constexpr unsigned int n_n = n_t + 1;
    constexpr unsigned int data_per_face =
      Utilities::pow(n_t, dim - 1) + 2 * n_n * Utilities::pow(n_t, dim - 2) * (dim - 1);

    std::array<VectorizedArray<Number>, dim * 3 * n_n * n_t> tmp_vec;

    for(unsigned int count = first_face; count < end_face;)
    {
      const unsigned int face = send_data_face_index[count];

      // to use vectorized code path, must check that faces are available,
      // faces are sorted by face number within each partner
      unsigned int n_faces = 1;
      if(count + n_lanes <= end_face && send_data_face_index[count + n_lanes - 1] == face)
        n_faces = n_lanes;
      else
        while(count + n_faces < end_face && send_data_face_index[count + n_faces] == face)
          ++n_faces;
      AssertIndexRange(n_faces, n_lanes + 1);

      dealii::ndarray<unsigned int, 2 * dim + 1, n_lanes> dof_indices_vec;
      std::array<unsigned int, n_lanes>                   indices;
      for(unsigned int v = 0; v < n_faces; ++v, ++count)
      {
        indices[v]              = (count - first_face) * data_per_face;
        const unsigned int cell = send_data_cell_index[count] / n_lanes;
        const unsigned int lane = send_data_cell_index[count] % n_lanes;
        AssertDimension(face, send_data_face_index[count]);
        for(unsigned int f = 0; f < 2 * dim + 1; ++f)
          dof_indices_vec[f][v] = dof_indices[cell][f][lane];
      }
      for(unsigned int v = n_faces; v < n_lanes; ++v)
        for(unsigned int f = 0; f < 2 * dim + 1; ++f)
          dof_indices_vec[f][v] = dof_indices_vec[f][0];
      read_face_values<n_t>(entity_pointers(src, dof_indices_vec), face, tmp_vec.data());

      // copy to dedicated data field
      if(n_faces == n_lanes)
        vectorized_transpose_and_store(false, data_per_face, tmp_vec.data(), indices.data(), out);
      else
        for(unsigned int v = 0; v < n_faces; ++v)
          for(unsigned int i = 0; i < data_per_face; ++i)
            out[indices[v] + i] = tmp_vec[i][v];
    }
  }

  void
  diagonal_operation(const unsigned int cell, VectorType & dst) const
  {
    const unsigned int degree = shape_info.data[0].fe_degree;
    AssertDimension(degree, shape_info.data[1].fe_degree + 1);
    if(degree == 2)
      do_cell_operation<2, false>(cell, dst, dst);
    else if(degree == 3)
      do_cell_operation<3, false>(cell, dst, dst);
    else if(degree == 4)
      do_cell_operation<4, false>(cell, dst, dst);
#ifndef DEBUG
    else if(degree == 5)
      do_cell_operation<5, false>(cell, dst, dst);
    else if(degree == 6)
      do_cell_operation<6, false>(cell, dst, dst);
    else if(degree == 7)
      do_cell_operation<7, false>(cell, dst, dst);
    else if(degree == 8)
      do_cell_operation<8, false>(cell, dst, dst);
    else if(degree == 9)
      do_cell_operation<9, false>(cell, dst, dst);
#endif
    else
      AssertThrow(false, ExcMessage("Degree " + std::to_string(degree) + " not instantiated"));
  }

  template<int degree, bool compute_exterior>
  void
  do_cell_operation(const unsigned int cell, const VectorType & src, VectorType & dst) const
  {
    do_cell_operation<degree, compute_exterior>(cell, src, dof_indices[cell], dst.begin());
  }

  // Variant of the cell operation that writes the result into the array
  // 'dst' at the positions given by 'dst_indices' (in the same format as
  // dof_indices), whereas src is read with the regular dof_indices. The
  // parameter 'last_stage' truncates the kernel for benchmarking, see
  // benchmark_cell_range().
  template<int degree, bool compute_exterior, int last_stage = 3>
  void
  do_cell_operation(const unsigned int                                          cell,
                    const VectorType &                                          src,
                    const dealii::ndarray<unsigned int, 2 * dim + 1, n_lanes> & dst_indices,
                    Number *                                                    dst) const
  {
    constexpr unsigned int dofs_per_component = Utilities::pow(degree, dim - 1) * (degree + 1);
    constexpr unsigned int dofs_per_cell      = dim * dofs_per_component;
    constexpr unsigned int n_q_points_1d      = degree + 1;
    constexpr unsigned int n_points           = Utilities::pow(n_q_points_1d, dim);
    const auto &           shape_data         = shape_info.data;

    VectorizedArray<Number> quad_values[dim * n_points], out_values[dim * n_points];
    if(compute_exterior)
    {
      read_cell_values<degree, n_q_points_1d>(src.begin(), dof_indices[cell], quad_values);
      if constexpr(last_stage == 0)
      {
        benchmark_escape(quad_values);
        return;
      }

      // If the second derivative terms have a non-zero factor, compute the
      // relevant parts of cell integrals and the associated face integrals
      if(factor_laplace != 0)
      {
        compute_cell<CellOperation::helmholtz, n_q_points_1d>(shape_data,
                                                              cell,
                                                              quad_values,
                                                              out_values);
        if constexpr(last_stage >= 2)
          for(unsigned int f = 0; f < 2 * dim; ++f)
            compute_face<degree, true>(shape_data, src, cell, f, quad_values, out_values);
      }
      else
        compute_cell_mass<n_q_points_1d>(cell, this->factor_mass, quad_values, out_values);
      if constexpr(last_stage < 3)
      {
        benchmark_escape(out_values);
        return;
      }

      integrate_cell_scatter<degree, n_q_points_1d>(dst_indices,
                                                    n_active_entries_per_cell_batch(cell),
                                                    out_values,
                                                    dst);
    }
    else
    {
      // TODO: Do we include the cross terms between the two cells that
      // share face DoFs (as the RT space is continuous in normal direction),
      // and the associated Dirichlet values? Check this carefully.
      VectorizedArray<Number> diagonal[dofs_per_cell];
      for(unsigned int d = 0; d < dim; ++d)
        for(unsigned int i = 0; i < dofs_per_component; ++i)
        {
          {
            internal::EvaluatorTensorProductAnisotropic<dim, degree, n_q_points_1d, true> eval;
            for(unsigned int j = 0; j < dofs_per_component; ++j)
              out_values[j] = 0;
            out_values[i] = 1;
            if(d == 0)
            {
              // always put the data last for the component in question,
              // the rest will be zero
              VectorizedArray<Number> * values = quad_values + (dim - 1) * n_points;
              eval.template normal<0>(shape_data[0], out_values, values);
              eval.template tangential<1, 0>(shape_data[1], values, values);
              if constexpr(dim > 2)
                eval.template tangential<2, 0>(shape_data[1], values, values);
            }
            else if(d == 1)
            {
              VectorizedArray<Number> * values = quad_values + (dim - 1) * n_points;
              eval.template normal<1>(shape_data[0], out_values, values);
              eval.template tangential<0, 1>(shape_data[1], values, values);
              if constexpr(dim > 2)
                eval.template tangential<2, 1>(shape_data[1], values, values);
            }
            else if(d == 2)
            {
              VectorizedArray<Number> * values = quad_values + (dim - 1) * n_points;

              eval.template normal<2>(shape_data[0], out_values, values);
              eval.template tangential<0, 2>(shape_data[1], values, values);
              eval.template tangential<1, 2>(shape_data[1], values, values);
            }
          }

          for(unsigned int i = 0; i < n_points; ++i)
          {
            quad_values[i * dim + d] = quad_values[(dim - 1) * n_points + i];
            for(unsigned int e = 0; e < dim; ++e)
              if(e != d)
                quad_values[i * dim + e] = 0;
          }

          if(factor_laplace != 0)
          {
            compute_cell<CellOperation::helmholtz, n_q_points_1d>(shape_data,
                                                                  cell,
                                                                  quad_values,
                                                                  out_values);
            for(unsigned int f = 0; f < 2 * dim; ++f)
              compute_face<degree, false>(shape_data, src, cell, f, quad_values, out_values);
          }
          else
            compute_cell_mass<n_q_points_1d>(cell, this->factor_mass, quad_values, out_values);

          for(unsigned int i = 0; i < n_points; ++i)
            quad_values[i] = out_values[i * dim + d];

          {
            internal::EvaluatorTensorProductAnisotropic<dim, degree, n_q_points_1d, false> eval;
            VectorizedArray<Number> * values = quad_values;

            if(d == 0)
            {
              if constexpr(dim > 2)
                eval.template tangential<2, 0>(shape_data[1], values, values);
              eval.template tangential<1, 0>(shape_data[1], values, values);
              eval.template normal<0>(shape_data[0], values, values);
              diagonal[i] = values[i];
            }

            if(d == 1)
            {
              if constexpr(dim > 2)
                eval.template tangential<2, 1>(shape_data[1], values, values);
              eval.template tangential<0, 1>(shape_data[1], values, values);
              eval.template normal<1>(shape_data[0], values, values);
              diagonal[dofs_per_component + i] = values[i];
            }

            if(d == 2)
            {
              eval.template tangential<1, 2>(shape_data[1], values, values);
              eval.template tangential<0, 2>(shape_data[1], values, values);
              eval.template normal<2>(shape_data[0], values, values);
              diagonal[2 * dofs_per_component + i] = values[i];
            }
          }
        }
      distribute_local_to_global_rt_compressed<dim, degree>(diagonal,
                                                            n_active_entries_per_cell_batch(cell),
                                                            dst_indices,
                                                            dst);
    }
  }

  template<int degree>
  void
  benchmark_cell_range(const BenchmarkKernel kernel,
                       const unsigned int    last_stage,
                       const unsigned int    begin,
                       const unsigned int    end,
                       const VectorType &    src,
                       VectorType &          dst,
                       VectorType &          convective_divergence,
                       VectorType &          divergence) const
  {
    if(last_stage == 0)
      benchmark_cell_range<degree, 0>(
        kernel, begin, end, src, dst, convective_divergence, divergence);
    else if(last_stage == 1)
      benchmark_cell_range<degree, 1>(
        kernel, begin, end, src, dst, convective_divergence, divergence);
    else if(last_stage == 2)
      benchmark_cell_range<degree, 2>(
        kernel, begin, end, src, dst, convective_divergence, divergence);
    else
      benchmark_cell_range<degree, 3>(
        kernel, begin, end, src, dst, convective_divergence, divergence);
  }

  template<int degree, int last_stage>
  void
  benchmark_cell_range(const BenchmarkKernel kernel,
                       const unsigned int    begin,
                       const unsigned int    end,
                       const VectorType &    src,
                       VectorType &          dst,
                       VectorType &          convective_divergence,
                       VectorType &          divergence) const
  {
    if(kernel == BenchmarkKernel::momentum)
      for(unsigned int cell = begin; cell < end; ++cell)
        do_cell_operation<degree, true, last_stage>(cell, src, dof_indices[cell], dst.begin());
    else if(kernel == BenchmarkKernel::convective)
    {
      if constexpr(last_stage == 3)
        for(unsigned int cell = begin; cell < end; ++cell)
          do_convective_on_cell<degree>(cell, src, Number(1.), dst);
    }
    else
    {
      VectorizedArray<Number> cfl = 0.;
      for(unsigned int cell = begin; cell < end; ++cell)
        cfl = std::max(cfl,
                       do_convective_and_divergence_on_cell<degree, last_stage>(
                         cell, src, dst, convective_divergence, divergence));
      benchmark_escape(&cfl);
    }
  }

  // Make the compiler assume that the memory behind 'pointer' is used, so
  // that the work of kernels truncated by 'last_stage' is not removed
  static void
  benchmark_escape(const void * pointer)
  {
    asm volatile("" : : "r"(pointer) : "memory");
  }

  template<int degree>
  void
  do_convective_on_cell(const unsigned int cell,
                        const VectorType & src,
                        const Number       factor_convective,
                        VectorType &       dst) const
  {
    constexpr unsigned int n_q_points_1d = degree + (degree + 1) / 2;
    constexpr unsigned int n_points      = Utilities::pow(n_q_points_1d, dim);
    const auto &           shape_data    = convective_shape_info.data;

    // compute_cell() overwrites the values at the quadrature points by the
    // test values, see do_convective_and_divergence_on_cell()
    VectorizedArray<Number> quad_values[dim * n_points];
    VectorizedArray<Number> face_data[2 * dim * (dim + 1) * Utilities::pow(n_q_points_1d, dim - 1)];
    VectorizedArray<Number> dof_values[dim * Utilities::pow(degree, dim - 1) * (degree + 1)];
    read_cell_values<degree, n_q_points_1d>(src.begin(),
                                            dof_indices[cell],
                                            quad_values,
                                            use_cell_dofs_for_faces(degree) ? dof_values : nullptr);

    // face integrals, applied together with the cell integrals
    for(unsigned int f = 0; f < 2 * dim; ++f)
      compute_face_convective<false, degree, n_q_points_1d>(
        shape_data, src, cell, f, factor_convective, quad_values, dof_values, face_data);

    compute_cell<CellOperation::convective, n_q_points_1d>(
      shape_data, cell, quad_values, quad_values, factor_convective, nullptr, face_data);

    integrate_cell_scatter<degree, n_q_points_1d>(dof_indices[cell],
                                                  n_active_entries_per_cell_batch(cell),
                                                  quad_values,
                                                  dst.begin());
  }

  // The parameter 'last_stage' truncates the kernel for benchmarking, see
  // benchmark_cell_range().
  template<int degree, int last_stage = 3>
  VectorizedArray<Number>
  do_convective_and_divergence_on_cell(const unsigned int cell,
                                       const VectorType & src,
                                       VectorType &       velocity_dst,
                                       VectorType &       convective_divergence,
                                       VectorType &       divergence) const
  {
    constexpr unsigned int n_q_points_1d = degree + (degree + 1) / 2;
    constexpr unsigned int n_points      = Utilities::pow(n_q_points_1d, dim);
    const auto &           shape_data    = convective_shape_info.data;

    constexpr unsigned int n_points_plane = Utilities::pow(n_q_points_1d, dim - 1);
    constexpr unsigned int n_pressure     = Utilities::pow(degree, dim);

    // The values at the quadrature points (quad_values), which compute_cell()
    // overwrites by the velocity test values followed by the CFL velocity,
    // and the three fields for the pressure test functions in the layout
    // described in compute_cell() (pressure_values). The test values can
    // replace the values in place because compute_cell() reads the values
    // of a plane of constant y only while processing that plane (the
    // derivatives in y direction are computed for the whole cell before),
    // and reads each point and line before writing it. The faces are
    // computed before compute_cell().
    VectorizedArray<Number> quad_values[dim * n_points + 1];
    VectorizedArray<Number> pressure_values[3 * n_points];
    VectorizedArray<Number> pressure_dofs[n_pressure];
    VectorizedArray<Number> face_data[2 * dim * (dim + 1) * n_points_plane];
    VectorizedArray<Number> dof_values[dim * Utilities::pow(degree, dim - 1) * (degree + 1)];
    read_cell_values<degree, n_q_points_1d>(src.begin(),
                                            dof_indices[cell],
                                            quad_values,
                                            use_cell_dofs_for_faces(degree) ? dof_values : nullptr);
    if constexpr(last_stage == 0)
    {
      benchmark_escape(quad_values);
      return quad_values[0];
    }

    // face integrals, applied together with the cell integrals
    for(unsigned int f = 0; f < 2 * dim; ++f)
      compute_face_convective<true, degree, n_q_points_1d>(
        shape_data, src, cell, f, 1.0, quad_values, dof_values, face_data);

    if constexpr(last_stage == 1)
    {
      benchmark_escape(face_data);
      return face_data[0];
    }

    compute_cell<CellOperation::convective_and_divergence, n_q_points_1d>(
      shape_data, cell, quad_values, quad_values, 1.0, pressure_values, face_data);

    // Add the part of the convective divergence tested by the derivative of
    // the pressure test functions in y direction, which compute_cell()
    // cannot do plane by plane
    {
      const Number * shape_grad = shape_data[0].shape_gradients_collocation_eo.data();
      for(unsigned int i1 = 0; i1 < (dim == 3 ? n_q_points_1d : 1); ++i1)
        for(unsigned int i0 = 0; i0 < n_q_points_1d; ++i0)
          internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                internal::EvaluatorQuantity::gradient,
                                                n_q_points_1d,
                                                n_q_points_1d,
                                                3 * n_points_plane,
                                                3 * n_points_plane,
                                                false,
                                                true>(shape_grad,
                                                      pressure_values + i1 * n_q_points_1d + i0,
                                                      pressure_values + n_points_plane +
                                                        i1 * n_q_points_1d + i0);
    }

    // integrate divergence
    integrate_pressure_field<degree, n_q_points_1d>(shape_data[1].shape_values_eo.data(),
                                                    pressure_values + 2 * n_points_plane,
                                                    pressure_dofs);

    const std::array<unsigned int, n_lanes> pressure_indices      = (*pressure_dof_indices)[cell];
    bool                                    all_indices_available = true;
    for(unsigned int v = 0; v < n_lanes; ++v)
      if(pressure_indices[v] == numbers::invalid_unsigned_int)
      {
        all_indices_available = false;
        break;
      }
    const auto store_pressure = [&](VectorType & dst) {
      if(all_indices_available)
        vectorized_transpose_and_store(
          false, n_pressure, pressure_dofs, pressure_indices.data(), dst.begin());
      else
        for(unsigned int v = 0; v < n_lanes && pressure_indices[v] != numbers::invalid_unsigned_int;
            ++v)
          for(unsigned int i = 0; i < n_pressure; ++i)
            dst.local_element(pressure_indices[v] + i) = pressure_dofs[i][v];
    };
    store_pressure(divergence);

    if constexpr(last_stage == 2)
    {
      benchmark_escape(quad_values);
      benchmark_escape(pressure_values);
      return quad_values[dim * n_points];
    }

    integrate_cell_scatter<degree, n_q_points_1d>(dof_indices[cell],
                                                  n_active_entries_per_cell_batch(cell),
                                                  quad_values,
                                                  velocity_dst.begin());

    // integrate convective divergence
    integrate_pressure_field<degree, n_q_points_1d>(shape_data[1].shape_values_eo.data(),
                                                    pressure_values + n_points_plane,
                                                    pressure_dofs);
    store_pressure(convective_divergence);

    return quad_values[dim * n_points];
  }

  // Integrate one of the fields for the pressure test functions in the
  // layout of compute_cell() at the quadrature points against the pressure
  // basis of degree n_t - 1, i.e., apply the transpose of the interpolation
  // from the pressure basis to the quadrature points. The result is
  // written to 'out' in lexicographic order. The field 'in' is overwritten.
  template<int n_t, int n_points_1d>
  void
  integrate_pressure_field(const Number * DEAL_II_RESTRICT shape_values,
                           VectorizedArray<Number> *       in,
                           VectorizedArray<Number> *       out) const
  {
    constexpr unsigned int n_points_plane = Utilities::pow(n_points_1d, dim - 1);
    constexpr unsigned int stride_y       = 3 * n_points_plane;

    // z and x direction within the planes of constant y
    for(unsigned int i1 = 0; i1 < n_points_1d; ++i1)
    {
      VectorizedArray<Number> * plane = in + i1 * stride_y;
      if constexpr(dim == 3)
        for(unsigned int i0 = 0; i0 < n_points_1d; ++i0)
          internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                internal::EvaluatorQuantity::value,
                                                n_t,
                                                n_points_1d,
                                                n_points_1d,
                                                n_points_1d,
                                                false,
                                                false>(shape_values, plane + i0, plane + i0);
      for(unsigned int i2 = 0; i2 < (dim == 3 ? n_t : 1); ++i2)
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::value,
                                              n_t,
                                              n_points_1d,
                                              1,
                                              1,
                                              false,
                                              false>(shape_values,
                                                     plane + i2 * n_points_1d,
                                                     plane + i2 * n_points_1d);
    }

    // y direction into the output
    for(unsigned int i2 = 0; i2 < (dim == 3 ? n_t : 1); ++i2)
      for(unsigned int i0 = 0; i0 < n_t; ++i0)
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::value,
                                              n_t,
                                              n_points_1d,
                                              stride_y,
                                              n_t,
                                              false,
                                              false>(shape_values,
                                                     in + i2 * n_points_1d + i0,
                                                     out + i2 * n_t * n_t + i0);
  }

  template<CellOperation cell_operation, int n_q_points_1d>
  void
  compute_cell(
    const std::vector<internal::MatrixFreeFunctions::UnivariateShapeData<Number>> & shape_data,
    const unsigned int                                                              cell,
    const VectorizedArray<Number> *                                                 quad_values,
    VectorizedArray<Number> *                                                       out_values,
    const Number                    factor_convective = 1.0,
    VectorizedArray<Number> *       pressure_values   = nullptr,
    const VectorizedArray<Number> * face_data         = nullptr) const
  {
    AssertThrow(cell_operation == CellOperation::convective || factor_convective == 1.0,
                ExcInternalError());
    constexpr unsigned int n_q_points_2d = n_q_points_1d * n_q_points_1d;
    constexpr unsigned int n_q_points    = Utilities::pow(n_q_points_1d, dim);

    // 'out_values' may be the same array as 'quad_values': the derivatives
    // in y direction are computed for all points before the loop over the
    // planes of constant y, and within the loop, the values of a plane are
    // only read while processing that plane, with each line in z direction
    // and each point read before its test values are written.
    //
    // For the convective terms, the face integrals have been computed
    // before and are stored in 'face_data' in the format described in
    // compute_face_convective(). Their terms are added together with the
    // cell terms at each point, which avoids separate passes through the
    // cell for each face.
    constexpr bool with_faces = cell_operation != CellOperation::helmholtz && dim == 3;
    AssertThrow(with_faces == (face_data != nullptr), ExcInternalError());
    const std::array<Number, 2> * shape_left =
      with_faces ? convective_interpolate_quad_to_boundary[0].data() : nullptr;
    const std::array<Number, 2> * shape_right =
      with_faces ? convective_interpolate_quad_to_boundary[1].data() : nullptr;
    constexpr unsigned int face_data_size = (dim + 1) * n_q_points_2d;
    const auto             face_term      = [&](const unsigned int component,
                               const unsigned int qx,
                               const unsigned int qy,
                               const unsigned int qz) {
      const VectorizedArray<Number> * data = face_data + component * n_q_points_2d;
      VectorizedArray<Number>         sum =
        shape_left[qx][0] * data[qy + qz * n_q_points_1d] +
        shape_right[qx][0] * data[face_data_size + qy + qz * n_q_points_1d];
      sum += shape_left[qy][0] * data[2 * face_data_size + qx + qz * n_q_points_1d] +
             shape_right[qy][0] * data[3 * face_data_size + qx + qz * n_q_points_1d];
      sum += shape_left[qz][0] * data[4 * face_data_size + qx + qy * n_q_points_1d] +
             shape_right[qz][0] * data[5 * face_data_size + qx + qy * n_q_points_1d];
      return sum;
    };

    // For the convective term with divergence, the terms tested by the
    // pressure test functions go to 'pressure_values' (of size 3 *
    // n_q_points) in blocks of 3 * n_points_plane entries per plane of
    // constant y, with the points within a plane numbered with x running
    // fastest, followed by z. The block contains three fields:
    //   0: the term of the convective divergence tested by the derivative
    //      of the pressure test functions in y direction, at the quadrature
    //      points, which the caller integrates over the planes,
    //   1: the terms of the convective divergence tested by the
    //      derivatives in x and z direction, already integrated in these
    //      two directions within the plane,
    //   2: the divergence tested by the values of the pressure test
    //      functions.
    // With this layout, the integration in x and z direction happens within
    // a plane, and the gradients in y direction of the velocity can share
    // the memory in 3D: they are no longer needed for a plane once its
    // fields are written.
    constexpr bool with_pressure      = cell_operation == CellOperation::convective_and_divergence;
    constexpr bool grad_y_in_pressure = with_pressure && dim == 3;
    constexpr unsigned int    n_points_plane = Utilities::pow(n_q_points_1d, dim - 1);
    VectorizedArray<Number>   grad_y_data[grad_y_in_pressure ? 1 : n_q_points * dim];
    VectorizedArray<Number> * grad_y = grad_y_in_pressure ? pressure_values : grad_y_data;
    VectorizedArray<Number>   flux_x_plane[with_pressure ? n_points_plane : 1];
    VectorizedArray<Number>   flux_y_plane[with_pressure ? n_points_plane : 1];
    VectorizedArray<Number>   flux_xz_plane[with_pressure ? n_points_plane : 1];
    VectorizedArray<Number>   divergence_plane[with_pressure ? n_points_plane : 1];
    VectorizedArray<Number>   flux_z_line[with_pressure ? n_q_points_1d : 1];
    VectorizedArray<Number>   grad_x[Utilities::pow(n_q_points_1d, dim - 1) * dim];
    VectorizedArray<Number>   grad_z[n_q_points_1d * dim];

    const Extruded::MappingInfo<dim, Number> & mapping_info =
      cell_operation == CellOperation::helmholtz ? this->mapping_info :
                                                   this->convective_mapping_info;

    std::array<unsigned int, n_lanes> shifted_data_indices;
    for(unsigned int v = 0; v < n_lanes; ++v)
      shifted_data_indices[v] = mapping_info.mapping_data_index[cell][v] * 4;
    std::array<unsigned int, n_lanes> shifted_data_indices_gr;
    for(unsigned int v = 0; v < n_lanes; ++v)
      shifted_data_indices_gr[v] = mapping_info.mapping_data_index[cell][v] * 6;

    const Number factor_mass = this->factor_mass;
    const Number factor_lapl = this->factor_laplace;
    const Number h_z         = mapping_info.h_z;
    const Number h_z_inverse = mapping_info.h_z_inverse;

    constexpr unsigned int nn         = n_q_points_1d;
    const Number *         shape_grad = shape_data[0].shape_gradients_collocation_eo.data();

    VectorizedArray<Number> cfl_velocity = 0;

    for(unsigned int qy = 0, q1 = 0; qy < nn; ++qy)
    {
      for(unsigned int i0 = 0; i0 < (dim == 3 ? nn : 1); ++i0)
        for(unsigned int d = 0; d < dim; ++d)
          internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                internal::EvaluatorQuantity::gradient,
                                                nn,
                                                nn,
                                                dim,
                                                (dim == 3 ? nn : 1) * dim,
                                                true,
                                                false>(shape_grad,
                                                       quad_values +
                                                         dim * (qy * nn + i0 * n_q_points_2d) + d,
                                                       grad_x + dim * i0 + d);

      for(unsigned int qx = 0; qx < nn; ++qx, ++q1)
      {
        Tensor<2, 2, VectorizedArray<Number>> jac_xy;
        vectorized_load_and_transpose(4,
                                      &mapping_info.jacobians_xy[q1][0][0],
                                      shifted_data_indices.data(),
                                      &jac_xy[0][0]);
        const VectorizedArray<Number>               inv_jac_det = Number(1.0) / determinant(jac_xy);
        const Tensor<2, 2, VectorizedArray<Number>> inv_jac_xy  = transpose(invert(jac_xy));
        Tensor<1, 3, Tensor<1, 2, VectorizedArray<Number>>> jac_grad_xy;
        vectorized_load_and_transpose(6,
                                      &mapping_info.jacobian_grads[q1][0][0],
                                      shifted_data_indices_gr.data(),
                                      &jac_grad_xy[0][0]);

        // Prepare for -(J^{-T} * jac_grad * J^{-1} * J * values *
        // det(J^{-1})), which can be expressed as a rank-1 update tmp[d] *
        // tmp4[e], where tmp = J * values and tmp4 = (J^{-T} * jac_grad *
        // J^{-1})
        VectorizedArray<Number> jac_grad_rank1[2];
        {
          VectorizedArray<Number> tmp[2];
          for(unsigned int d = 0; d < 2; ++d)
            tmp[d] =
              (inv_jac_xy[0][d] * jac_grad_xy[d][0] + inv_jac_xy[1][d] * jac_grad_xy[d][1] +
               inv_jac_xy[0][1 - d] * jac_grad_xy[2][0] + inv_jac_xy[1][1 - d] * jac_grad_xy[2][1]);
          for(unsigned int d = 0; d < 2; ++d)
            jac_grad_rank1[d] = (tmp[0] * inv_jac_xy[d][0] + tmp[1] * inv_jac_xy[d][1]);
        }

        if constexpr(dim == 2)
        {
          if(qy == 0)
            for(unsigned int d = 0; d < dim; ++d)
              internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                    internal::EvaluatorQuantity::gradient,
                                                    nn,
                                                    nn,
                                                    nn * dim,
                                                    nn * dim,
                                                    true,
                                                    false>(shape_grad,
                                                           quad_values + dim * qx + d,
                                                           grad_y + dim * qx + d);

          const VectorizedArray<Number> val[2] = {quad_values[q1 * dim], quad_values[q1 * dim + 1]};
          const VectorizedArray<Number> grad[2][2]  = {{grad_x[qx * dim], grad_x[qx * dim + 1]},
                                                      {grad_y[q1 * dim], grad_y[q1 * dim + 1]}};
          const VectorizedArray<Number> t0          = jac_xy[0][0] * val[0] + jac_xy[0][1] * val[1];
          const VectorizedArray<Number> t1          = jac_xy[1][0] * val[0] + jac_xy[1][1] * val[1];
          VectorizedArray<Number>       val_real[2] = {t0, t1};

          VectorizedArray<Number> grad_real[dim][dim];
          for(unsigned int d = 0; d < dim; ++d)
          {
            // (J * grad_quad) * J^-1 * det(J^-1), part in braces
            VectorizedArray<Number> tmp[dim];
            for(unsigned int e = 0; e < dim; ++e)
              tmp[e] = (jac_xy[d][0] * grad[0][e] + jac_xy[d][1] * grad[1][e]);

            // Add (jac_grad * values) * J^{-1} * det(J^{-1}), combine
            // terms outside braces with gradient part from above
            tmp[0] += jac_grad_xy[0][d] * val[0];
            tmp[0] += jac_grad_xy[2][d] * val[1];
            tmp[1] += jac_grad_xy[1][d] * val[1];
            tmp[1] += jac_grad_xy[2][d] * val[0];

            for(unsigned int e = 0; e < dim; ++e)
              grad_real[d][e] = (tmp[0] * inv_jac_xy[e][0] + tmp[1] * inv_jac_xy[e][1]);

            VectorizedArray<Number> tmp2 = jac_xy[d][0] * val[0] + jac_xy[d][1] * val[1];

            for(unsigned int e = 0; e < dim; ++e)
              grad_real[d][e] -= jac_grad_rank1[e] * tmp2;
          }

          if(cell_operation == CellOperation::convective)
          {
            VectorizedArray<Number> factor = inv_jac_det * inv_jac_det;
            VectorizedArray<Number> tmp[2] = {grad_real[0][0] * t0 + grad_real[0][1] * t1,
                                              grad_real[1][0] * t0 + grad_real[1][1] * t1};
            for(unsigned int d = 0; d < dim; ++d)
              for(unsigned int e = 0; e < dim; ++e)
                grad_real[d][e] = (factor_convective - Number(1.0)) * factor *
                                  mapping_info.quad_weights_xy[q1] * val_real[d] * val_real[e];
            val_real[0] = tmp[0] * (factor_convective * factor * mapping_info.quad_weights_xy[q1]);
            val_real[1] = tmp[1] * (factor_convective * factor * mapping_info.quad_weights_xy[q1]);
          }
          else if(cell_operation == CellOperation::convective_and_divergence)
          {
            {
              // scaled velocity = inv_jac * val_real = 1/det(jac) * val_ref
              Tensor<1, dim, VectorizedArray<Number>> scaled_velocity;
              for(unsigned int d = 0; d < dim; ++d)
                scaled_velocity[d] = val[d];
              cfl_velocity =
                std::max(cfl_velocity, scaled_velocity.norm_square() * (inv_jac_det * inv_jac_det));
            }
            VectorizedArray<Number> tmp[2] = {grad_real[0][0] * t0 + grad_real[0][1] * t1,
                                              grad_real[1][0] * t0 + grad_real[1][1] * t1};
            for(unsigned int d = 0; d < dim; ++d)
              val_real[d] = tmp[d] * (inv_jac_det * mapping_info.quad_weights_xy[q1] * inv_jac_det);
            flux_x_plane[qx] = tmp[0] * (-inv_jac_det * mapping_info.quad_weights_xy[q1]);
            flux_y_plane[qx] = tmp[1] * (-inv_jac_det * mapping_info.quad_weights_xy[q1]);

            // divergence
            divergence_plane[qx] = mapping_info.quad_weights_xy[q1] * (grad[0][0] + grad[1][1]);
          }
          else
          {
            // Laplacian
            for(unsigned int d = 0; d < dim; ++d)
              for(unsigned int e = 0; e < dim; ++e)
                // multiply by det(J^{-1}) necessary in all
                // contributions above and the factors in the equation
                grad_real[d][e] *= mapping_info.quad_weights_xy[q1] * factor_lapl * inv_jac_det;
            val_real[0] *= inv_jac_det * factor_mass * mapping_info.quad_weights_xy[q1];
            val_real[1] *= inv_jac_det * factor_mass * mapping_info.quad_weights_xy[q1];
          }

          if(cell_operation == CellOperation::helmholtz ||
             (cell_operation == CellOperation::convective && factor_convective != Number(1.0)))
          {
            // J * (J^{-1} * (grad_in * factor))
            VectorizedArray<Number> tmp[dim][dim];
            for(unsigned int d = 0; d < dim; ++d)
              for(unsigned int e = 0; e < dim; ++e)
                tmp[d][e] =
                  (inv_jac_xy[0][d] * grad_real[e][0] + inv_jac_xy[1][d] * grad_real[e][1]);

            for(unsigned int d = 0; d < dim; ++d)
            {
              grad_x[qx * dim + d] = (jac_xy[0][0] * tmp[d][0] + jac_xy[1][0] * tmp[d][1]);
              grad_y[q1 * dim + d] = (jac_xy[0][1] * tmp[d][0] + jac_xy[1][1] * tmp[d][1]);
            }

            // jac_grad * (J^{-1} * (grad_in * factor)), re-use part in
            // braces as 'tmp' from above
            VectorizedArray<Number> value_tmp[dim];
            for(unsigned int d = 0; d < dim; ++d)
              value_tmp[d] =
                (tmp[d][0] * jac_grad_xy[d][0] + tmp[d][1] * jac_grad_xy[d][1] +
                 tmp[1 - d][0] * jac_grad_xy[2][0] + tmp[1 - d][1] * jac_grad_xy[2][1]);

            //   -(grad_in * factor) * J * (J^{-T} * jac_grad * J^{-1})
            // = -(grad_in * factor) * J * ( \------- tmp4 ---------/ )
            for(unsigned int d = 0; d < dim; ++d)
            {
              VectorizedArray<Number> tmp2 =
                (grad_real[d][0] * jac_grad_rank1[0] + grad_real[d][1] * jac_grad_rank1[1]);
              for(unsigned int e = 0; e < dim; ++e)
                value_tmp[e] -= jac_xy[d][e] * tmp2;
            }

            out_values[q1 * dim] =
              value_tmp[0] + val_real[0] * jac_xy[0][0] + val_real[1] * jac_xy[1][0];
            out_values[q1 * dim + 1] =
              value_tmp[1] + val_real[0] * jac_xy[0][1] + val_real[1] * jac_xy[1][1];
          }
          else
          {
            out_values[q1 * dim]     = val_real[0] * jac_xy[0][0] + val_real[1] * jac_xy[1][0];
            out_values[q1 * dim + 1] = val_real[0] * jac_xy[0][1] + val_real[1] * jac_xy[1][1];
          }

          if(qy == nn - 1)
            for(unsigned int d = 0; d < dim; ++d)
              internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                    internal::EvaluatorQuantity::gradient,
                                                    nn,
                                                    nn,
                                                    nn * dim,
                                                    nn * dim,
                                                    false,
                                                    true>(shape_grad,
                                                          grad_y + dim * qx + d,
                                                          out_values + dim * qx + d);
        }
        else // now to dim == 3
        {
          if(qy == 0)
            for(unsigned int i0 = 0; i0 < nn; ++i0)
              for(unsigned int d = 0; d < dim; ++d)
                internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                      internal::EvaluatorQuantity::gradient,
                                                      nn,
                                                      nn,
                                                      nn * dim,
                                                      nn * nn * dim,
                                                      true,
                                                      false>(shape_grad,
                                                             quad_values +
                                                               dim * (i0 * nn * nn + qx) + d,
                                                             grad_y + dim * (qx * nn + i0) + d);
          for(unsigned int d = 0; d < dim; ++d)
            internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                  internal::EvaluatorQuantity::gradient,
                                                  nn,
                                                  nn,
                                                  nn * nn * dim,
                                                  dim,
                                                  true,
                                                  false>(shape_grad,
                                                         quad_values + q1 * dim + d,
                                                         grad_z + d);

          for(unsigned int qz = 0, q = q1; qz < nn; ++qz, q += n_q_points_2d)
          {
            const unsigned int            ix         = dim * (qz + qx * nn);
            const unsigned int            iy         = dim * (qz + qx * nn + qy * nn * nn);
            const unsigned int            iz         = dim * qz;
            const VectorizedArray<Number> val[3]     = {quad_values[q * dim],
                                                    quad_values[q * dim + 1],
                                                    quad_values[q * dim + 2]};
            const VectorizedArray<Number> grad[3][3] = {
              {grad_x[ix + 0], grad_y[iy + 0], grad_z[iz + 0]},
              {grad_x[ix + 1], grad_y[iy + 1], grad_z[iz + 1]},
              {grad_x[ix + 2], grad_y[iy + 2], grad_z[iz + 2]}};

            Tensor<1, dim, VectorizedArray<Number>> val_real;
            val_real[0] = jac_xy[0][0] * val[0] + jac_xy[0][1] * val[1];
            val_real[1] = jac_xy[1][0] * val[0] + jac_xy[1][1] * val[1];
            val_real[2] = h_z * val[2];

            const Number weight_z = mapping_info.quad_weights_z[qz];

            VectorizedArray<Number> grad_real[dim][dim];
            for(unsigned int d = 0; d < 2; ++d)
            {
              // (J * grad_quad) * J^-1 * det(J^-1), part in braces
              VectorizedArray<Number> tmp[dim];
              for(unsigned int e = 0; e < dim; ++e)
                tmp[e] = (jac_xy[d][0] * grad[0][e] + jac_xy[d][1] * grad[1][e]);

              // Add (jac_grad * values) * J^{-1} * det(J^{-1}),
              // combine terms outside braces with gradient part from
              // above
              tmp[0] += jac_grad_xy[0][d] * val[0];
              tmp[0] += jac_grad_xy[2][d] * val[1];
              tmp[1] += jac_grad_xy[1][d] * val[1];
              tmp[1] += jac_grad_xy[2][d] * val[0];

              for(unsigned int e = 0; e < 2; ++e)
                grad_real[d][e] = (tmp[0] * inv_jac_xy[e][0] + tmp[1] * inv_jac_xy[e][1]);

              for(unsigned int e = 0; e < 2; ++e)
                grad_real[d][e] -= jac_grad_rank1[e] * val_real[d];

              const VectorizedArray<Number> tmp_z_grad2 =
                (inv_jac_xy[d][0] * grad[2][0] + inv_jac_xy[d][1] * grad[2][1]);

              grad_real[2][d] = (tmp_z_grad2 - jac_grad_rank1[d] * val[2]) * h_z;
              grad_real[d][2] = h_z_inverse * tmp[2];
            }
            grad_real[2][2] = grad[2][2];

            if(cell_operation == CellOperation::convective)
            {
              const VectorizedArray<Number> factor =
                inv_jac_det * inv_jac_det * h_z_inverse * h_z_inverse;
              VectorizedArray<Number> tmp[3];
              for(unsigned int d = 0; d < dim; ++d)
              {
                VectorizedArray<Number> sum = grad_real[d][0] * val_real[0];
                for(unsigned int e = 1; e < dim; ++e)
                  sum += grad_real[d][e] * val_real[e];
                tmp[d] = sum;
              }
              for(unsigned int d = 0; d < dim; ++d)
                for(unsigned int e = 0; e < dim; ++e)
                  grad_real[d][e] = (factor_convective - Number(1.0)) * factor *
                                    mapping_info.quad_weights_xy[q1] * weight_z * val_real[d] *
                                    val_real[e];
              for(unsigned int d = 0; d < dim; ++d)
                val_real[d] = tmp[d] * (factor_convective * factor *
                                        mapping_info.quad_weights_xy[q1] * weight_z);
            }
            else if(cell_operation == CellOperation::convective_and_divergence)
            {
              const VectorizedArray<Number> inv_jac_det_3d = inv_jac_det * h_z_inverse;

              {
                // scaled velocity = inv_jac * val_real = 1/det(jac) * val_ref
                Tensor<1, dim, VectorizedArray<Number>> scaled_velocity;
                for(unsigned int d = 0; d < dim; ++d)
                  scaled_velocity[d] = val[d];
                cfl_velocity =
                  std::max(cfl_velocity,
                           scaled_velocity.norm_square() * (inv_jac_det_3d * inv_jac_det_3d));
              }
              VectorizedArray<Number> tmp[3];
              for(unsigned int d = 0; d < dim; ++d)
              {
                tmp[d] = grad_real[d][0] * val_real[0];
                for(unsigned int e = 1; e < dim; ++e)
                  tmp[d] += grad_real[d][e] * val_real[e];
              }
              for(unsigned int d = 0; d < dim; ++d)
              {
                val_real[d] = tmp[d] * (inv_jac_det_3d * mapping_info.quad_weights_xy[q1] *
                                        weight_z * inv_jac_det_3d);
                tmp[d] *= (-inv_jac_det_3d * mapping_info.quad_weights_xy[q1] * weight_z);
              }
              flux_x_plane[qz * nn + qx] = inv_jac_xy[0][0] * tmp[0] + inv_jac_xy[1][0] * tmp[1];
              flux_y_plane[qz * nn + qx] = inv_jac_xy[0][1] * tmp[0] + inv_jac_xy[1][1] * tmp[1];
              flux_z_line[qz]            = h_z_inverse * tmp[2];

              // divergence
              divergence_plane[qz * nn + qx] = (grad[0][0] + grad[1][1] + grad[2][2]) *
                                               mapping_info.quad_weights_xy[q1] * weight_z;
            }
            else
            {
              // Laplacian
              const VectorizedArray<Number> factor_deriv =
                (mapping_info.quad_weights_xy[q1] * inv_jac_det * factor_lapl * h_z_inverse) *
                weight_z;
              const VectorizedArray<Number> factor_ma =
                (mapping_info.quad_weights_xy[q1] * h_z_inverse * inv_jac_det * factor_mass) *
                weight_z;

              for(unsigned int d = 0; d < dim; ++d)
                for(unsigned int e = 0; e < dim; ++e)
                  grad_real[d][e] *= factor_deriv;
              for(unsigned int d = 0; d < dim; ++d)
                val_real[d] *= factor_ma;
            }

            // For the test function part, the Jacobian determinant
            // that is part of the integration factor cancels with the
            // inverse Jacobian determinant present in the factor of
            // the derivative

            // J * (J^{-1} * (grad_in * factor))
            if(cell_operation == CellOperation::helmholtz ||
               (cell_operation == CellOperation::convective && factor_convective != Number(1.0)))
            {
              VectorizedArray<Number> tmp[2][dim];
              for(unsigned int d = 0; d < 2; ++d)
                for(unsigned int e = 0; e < dim; ++e)
                  tmp[d][e] =
                    (inv_jac_xy[0][d] * grad_real[e][0] + inv_jac_xy[1][d] * grad_real[e][1]);

              grad_x[ix]     = (jac_xy[0][0] * tmp[0][0] + jac_xy[1][0] * tmp[0][1]);
              grad_x[ix + 1] = (jac_xy[0][1] * tmp[0][0] + jac_xy[1][1] * tmp[0][1]);
              grad_x[ix + 2] = h_z * tmp[0][2];
              grad_y[iy]     = (jac_xy[0][0] * tmp[1][0] + jac_xy[1][0] * tmp[1][1]);
              grad_y[iy + 1] = (jac_xy[0][1] * tmp[1][0] + jac_xy[1][1] * tmp[1][1]);
              grad_y[iy + 2] = h_z * tmp[1][2];
              grad_z[iz]     = (jac_xy[0][0] * (grad_real[0][2] * h_z_inverse) +
                            jac_xy[1][0] * (grad_real[1][2] * h_z_inverse));
              grad_z[iz + 1] = (jac_xy[0][1] * (grad_real[0][2] * h_z_inverse) +
                                jac_xy[1][1] * (grad_real[1][2] * h_z_inverse));
              grad_z[iz + 2] = grad_real[2][2];

              // jac_grad * (J^{-1} * (grad_in * factor)), re-use part in
              // braces as 'tmp' from above
              VectorizedArray<Number> value_tmp[dim];
              for(unsigned int d = 0; d < 2; ++d)
                value_tmp[d] =
                  (tmp[d][0] * jac_grad_xy[d][0] + tmp[d][1] * jac_grad_xy[d][1] +
                   tmp[1 - d][0] * jac_grad_xy[2][0] + tmp[1 - d][1] * jac_grad_xy[2][1]);

              //   -(grad_in * factor) * J * (J^{-T} * jac_grad * J^{-1})
              // = -(grad_in * factor) * J * ( \------- tmp4 ---------/ )
              for(unsigned int d = 0; d < 2; ++d)
              {
                VectorizedArray<Number> tmp2 =
                  (grad_real[d][0] * jac_grad_rank1[0] + grad_real[d][1] * jac_grad_rank1[1]);
                for(unsigned int e = 0; e < 2; ++e)
                  value_tmp[e] += jac_xy[d][e] * (val_real[d] - tmp2);
              }
              value_tmp[2] = h_z * (val_real[2] - grad_real[2][0] * jac_grad_rank1[0] -
                                    grad_real[2][1] * jac_grad_rank1[1]);

              if constexpr(with_faces)
                for(unsigned int d = 0; d < dim; ++d)
                  value_tmp[d] += face_term(d, qx, qy, qz);
              out_values[q * dim]     = value_tmp[0];
              out_values[q * dim + 1] = value_tmp[1];
              out_values[q * dim + 2] = value_tmp[2];
            }
            else
            {
              // shortcut for convective form of convection
              VectorizedArray<Number> value[dim];
              for(unsigned int d = 0; d < 2; ++d)
                value[d] = jac_xy[0][d] * val_real[0] + jac_xy[1][d] * val_real[1];
              value[2] = h_z * val_real[2];
              if constexpr(with_faces)
                for(unsigned int d = 0; d < dim; ++d)
                  value[d] += face_term(d, qx, qy, qz);
              for(unsigned int d = 0; d < dim; ++d)
                out_values[q * dim + d] = value[d];
            }
          }

          if(cell_operation == CellOperation::helmholtz ||
             (cell_operation == CellOperation::convective && factor_convective != Number(1.0)))
          {
            for(unsigned int d = 0; d < dim; ++d)
              internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                    internal::EvaluatorQuantity::gradient,
                                                    nn,
                                                    nn,
                                                    dim,
                                                    nn * nn * dim,
                                                    false,
                                                    true>(shape_grad,
                                                          grad_z + d,
                                                          out_values + q1 * dim + d);

            if(qy == nn - 1)
              for(unsigned int i0 = 0; i0 < nn; ++i0)
                for(unsigned int d = 0; d < dim; ++d)
                  internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                        internal::EvaluatorQuantity::gradient,
                                                        nn,
                                                        nn,
                                                        nn * nn * dim,
                                                        nn * dim,
                                                        false,
                                                        true>(shape_grad,
                                                              grad_y + (qx * nn + i0) * dim + d,
                                                              out_values +
                                                                (i0 * nn * nn + qx) * dim + d);
          }

          // test the z component of the convective divergence by the
          // derivative in z direction
          if constexpr(with_pressure)
            internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                  internal::EvaluatorQuantity::gradient,
                                                  nn,
                                                  nn,
                                                  1,
                                                  nn,
                                                  false,
                                                  false>(shape_grad,
                                                         flux_z_line,
                                                         flux_xz_plane + qx);
        }
      }

      // write the fields of this plane, after testing the x component of
      // the convective divergence by the derivative in x direction
      if constexpr(with_pressure)
      {
        VectorizedArray<Number> * plane = pressure_values + qy * 3 * n_points_plane;
        for(unsigned int i = 0; i < n_points_plane; ++i)
        {
          plane[i]                      = flux_y_plane[i];
          plane[2 * n_points_plane + i] = divergence_plane[i];
        }
        if constexpr(dim == 3)
          for(unsigned int qz = 0, i = 0; qz < nn; ++qz)
            for(unsigned int qx = 0; qx < nn; ++qx, ++i)
              plane[n_points_plane + i] = flux_xz_plane[i] + face_term(dim, qx, qy, qz);
        for(unsigned int i1 = 0; i1 < (dim == 3 ? nn : 1); ++i1)
          internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                internal::EvaluatorQuantity::gradient,
                                                nn,
                                                nn,
                                                1,
                                                1,
                                                false,
                                                dim == 3>(shape_grad,
                                                          flux_x_plane + i1 * nn,
                                                          plane + n_points_plane + i1 * nn);
      }

      if(cell_operation == CellOperation::helmholtz ||
         (cell_operation == CellOperation::convective && factor_convective != Number(1.0)))
        for(unsigned int i0 = 0; i0 < (dim == 3 ? nn : 1); ++i0)
          for(unsigned int d = 0; d < dim; ++d)
            internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                  internal::EvaluatorQuantity::gradient,
                                                  nn,
                                                  nn,
                                                  (dim == 3 ? nn : 1) * dim,
                                                  dim,
                                                  false,
                                                  true>(shape_grad,
                                                        grad_x + i0 * dim + d,
                                                        out_values +
                                                          (qy * nn + i0 * n_q_points_2d) * dim + d);
    }

    if(cell_operation == CellOperation::convective_and_divergence)
      out_values[dim * n_q_points] = std::sqrt(cfl_velocity);
  }

  template<int n_q_points_1d>
  void
  compute_cell_mass(const unsigned int        cell,
                    const Number              factor_mass,
                    VectorizedArray<Number> * quad_values,
                    VectorizedArray<Number> * out_values) const
  {
    constexpr unsigned int n_q_points_2d = n_q_points_1d * n_q_points_1d;

    std::array<unsigned int, n_lanes> shifted_data_indices;
    for(unsigned int v = 0; v < n_lanes; ++v)
      shifted_data_indices[v] = mapping_info.mapping_data_index[cell][v] * 4;

    const Number h_z_inverse = mapping_info.h_z_inverse;
    const Number h_z         = mapping_info.h_z;

    constexpr unsigned int nn = n_q_points_1d;

    for(unsigned int qy = 0, q1 = 0; qy < nn; ++qy)
    {
      for(unsigned int qx = 0; qx < nn; ++qx, ++q1)
      {
        Tensor<2, 2, VectorizedArray<Number>> jac_xy;
        vectorized_load_and_transpose(4,
                                      &mapping_info.jacobians_xy[q1][0][0],
                                      shifted_data_indices.data(),
                                      &jac_xy[0][0]);
        const VectorizedArray<Number> inv_jac_det = Number(1.0) / determinant(jac_xy);

        if constexpr(dim == 2)
        {
          const VectorizedArray<Number> val[2] = {quad_values[q1 * dim], quad_values[q1 * dim + 1]};
          const VectorizedArray<Number> t0     = jac_xy[0][0] * val[0] + jac_xy[0][1] * val[1];
          const VectorizedArray<Number> t1     = jac_xy[1][0] * val[0] + jac_xy[1][1] * val[1];
          const VectorizedArray<Number> s0     = jac_xy[0][0] * t0 + jac_xy[1][0] * t1;
          const VectorizedArray<Number> s1     = jac_xy[0][1] * t0 + jac_xy[1][1] * t1;

          quad_values[q1 * dim] =
            s0 * (inv_jac_det * factor_mass * mapping_info.quad_weights_xy[q1]);
          quad_values[q1 * dim + 1] =
            s1 * (inv_jac_det * factor_mass * mapping_info.quad_weights_xy[q1]);
        }
        else // now to dim == 3
        {
          for(unsigned int qz = 0, q = q1; qz < nn; ++qz, q += n_q_points_2d)
          {
            const VectorizedArray<Number> val[3] = {quad_values[q * dim],
                                                    quad_values[q * dim + 1],
                                                    quad_values[q * dim + 2]};

            Tensor<1, dim, VectorizedArray<Number>> val_real;
            val_real[0] = jac_xy[0][0] * val[0] + jac_xy[0][1] * val[1];
            val_real[1] = jac_xy[1][0] * val[0] + jac_xy[1][1] * val[1];
            val_real[2] = h_z * val[2];

            const Number weight_z = mapping_info.quad_weights_z[qz];
            // need factors without h_z_inverse in code below because
            // it cancels with h_z
            const VectorizedArray<Number> factor_ma =
              (mapping_info.quad_weights_xy[q1] * h_z_inverse * inv_jac_det * factor_mass) *
              weight_z;

            // For the test function part, the Jacobian determinant
            // that is part of the integration factor cancels with the
            // inverse Jacobian determinant present in the factor of
            // the derivative

            // jac_grad * (J^{-1} * (grad_in * factor)), re-use part in
            // braces as 'tmp' from above
            VectorizedArray<Number> value_tmp[dim];
            for(unsigned int d = 0; d < dim; ++d)
              val_real[d] *= factor_ma;
            for(unsigned int d = 0; d < 2; ++d)
              value_tmp[d] = jac_xy[0][d] * val_real[0] + jac_xy[1][d] * val_real[1];
            value_tmp[2] = h_z * val_real[2];

            out_values[q * dim]     = value_tmp[0];
            out_values[q * dim + 1] = value_tmp[1];
            out_values[q * dim + 2] = value_tmp[2];
          }
        }
      }
    }
  }

  // The normal component of the velocity is continuous across faces, so
  // the value on the neighbor is the value interpolated from the interior
  // of the current cell. As read_face_values() skips the dofs on the face
  // when computing the normal derivative of that component, add their
  // contribution, using that the shape functions are nodal in normal
  // direction and only the shape function on the face is non-zero there.
  template<int n_q_points_face>
  void
  add_face_value_to_normal_derivative(
    const unsigned int            f,
    const VectorizedArray<Number> values_face[2][dim][n_q_points_face + 1],
    VectorizedArray<Number>       grads_face[2][dim][n_q_points_face * dim + 1]) const
  {
    const unsigned int face_direction = f / 2;
    const unsigned int side_neighbor  = 1 - f % 2;
    const unsigned int n_t            = shape_info.data[0].fe_degree;
    const unsigned int n_n            = n_t + 1;
    const Number       shape_derivative_on_face =
      shape_info.data[0].shape_data_on_face[side_neighbor][n_n + side_neighbor * n_t];
    for(unsigned int q = 0; q < n_q_points_face; ++q)
      grads_face[1][face_direction][q * dim + face_direction] +=
        shape_derivative_on_face * values_face[0][face_direction][q];
  }

  // Set up pointers to the values and derivatives on the exterior side of
  // a face. As the normal component of the velocity is continuous, its value
  // and its tangential derivatives on the exterior side are the ones of the
  // interior side, and only the normal derivative is taken from the
  // exterior.
  template<int n_q_points_face>
  void
  set_exterior_pointers(const unsigned int            face_direction,
                        const VectorizedArray<Number> values_face[2][dim][n_q_points_face + 1],
                        const VectorizedArray<Number> grads_face[2][dim][n_q_points_face * dim + 1],
                        const VectorizedArray<Number> * values_exterior[dim],
                        const VectorizedArray<Number> * grads_exterior[dim][dim]) const
  {
    for(unsigned int d = 0; d < dim; ++d)
    {
      values_exterior[d] = values_face[d == face_direction ? 0 : 1][d];
      for(unsigned int e = 0; e < dim; ++e)
        grads_exterior[d][e] =
          grads_face[(d == face_direction && e != face_direction) ? 0 : 1][d] + e;
    }
  }

  template<int degree, bool compute_exterior>
  void
  compute_face(
    const std::vector<internal::MatrixFreeFunctions::UnivariateShapeData<Number>> & shape_data,
    const VectorType &                                                              src,
    const unsigned int                                                              cell,
    const unsigned int                                                              f,
    const VectorizedArray<Number> *                                                 quad_values,
    VectorizedArray<Number> * out_values) const
  {
    constexpr bool use_face_buffer = true && compute_exterior;

    if(use_face_buffer && f % 2 == 0 && all_left_face_fluxes_from_buffer[cell][f / 2])
      return;

    constexpr unsigned int n_q_points_1d = degree + 1;
    constexpr unsigned int n_q_points_2d = n_q_points_1d * n_q_points_1d;
    AssertThrow(n_q_points_2d == mapping_info.quad_weights_xy.size(),
                ExcDimensionMismatch(n_q_points_2d, mapping_info.quad_weights_xy.size()));
    constexpr unsigned int n_lanes = VectorizedArray<Number>::size();

    ndarray<unsigned int, 2 * dim + 1, n_lanes> dof_indices_neighbor;
    unsigned int                                lane_with_neighbor = 0;
    for(unsigned int v = 0; v < n_lanes; ++v)
      if(neighbor_cells[cell][f][v] != numbers::invalid_unsigned_int)
      {
        lane_with_neighbor = v;
        break;
      }

    // TODO: 0: interior face, 1: Neumann, -1: Dirichlet, apply from actual b.c. in operator
    VectorizedArray<Number> boundary_mask;
    for(unsigned int v = 0; v < n_lanes; ++v)
      if(neighbor_cells[cell][f][v] == numbers::invalid_unsigned_int)
        boundary_mask[v] = -1.;
      else
        boundary_mask[v] = 0.;

    if(compute_exterior)
    {
      for(unsigned int v = 0; v < n_lanes; ++v)
      {
        // use a dummy neighbor with nearby data for cells at boundary to make
        // sure vectorized functions continue working
        const unsigned int neighbor_idx =
          neighbor_cells[cell][f][v] != numbers::invalid_unsigned_int ?
            neighbor_cells[cell][f][v] :
            neighbor_cells[cell][f][lane_with_neighbor];
        if(neighbor_idx != numbers::invalid_unsigned_int)
          for(unsigned int i = 0; i < 2 * dim + 1; ++i)
            dof_indices_neighbor[i][v] =
              dof_indices[neighbor_idx / n_lanes][i][neighbor_idx % n_lanes];
        else
          for(unsigned int i = 0; i < 2 * dim + 1; ++i)
            dof_indices_neighbor[i][v] = dof_indices[0][i][0];
      }
    }

    VectorizedArray<Number> face_array[dim * 2 * Utilities::pow(n_q_points_1d, dim - 1)];
    constexpr int           n_n         = degree + 1;
    constexpr int           n_t         = degree;
    const Number            factor_lapl = this->factor_laplace;

    internal::EvaluatorTensorProduct<internal::evaluate_evenodd,
                                     dim - 1,
                                     n_q_points_1d,
                                     n_q_points_1d,
                                     VectorizedArray<Number>,
                                     Number>
      eval_g({}, shape_data[0].shape_gradients_collocation_eo.data(), {});
    internal::EvaluatorTensorProduct<internal::evaluate_evenodd,
                                     dim - 1,
                                     degree,
                                     n_q_points_1d,
                                     VectorizedArray<Number>,
                                     Number>
      eval_iso(shape_data[1].shape_values_eo.data(), {}, {});
    internal::EvaluatorTensorProductAnisotropic<dim - 1, degree, n_q_points_1d, true> eval_aniso;

    const unsigned int      face_direction  = f / 2;
    constexpr unsigned int  n_q_points_face = Utilities::pow(n_q_points_1d, dim - 1);
    VectorizedArray<Number> scratch_data[n_q_points_face + 1];
    VectorizedArray<Number> values_face[2][dim][n_q_points_face + 1];
    VectorizedArray<Number> grads_face[2][dim][n_q_points_face * dim + 1];

    if constexpr(compute_exterior)
    {
      auto neighbor_entities = entity_pointers(src.begin(), dof_indices_neighbor);
      set_remote_cell_entities(src, cell, f, neighbor_entities);
      read_face_values<degree>(neighbor_entities, f + (f % 2 ? -1 : 1), face_array);

      // For faces located on MPI-remote processes, we replace the dummy
      // data read out above by the data sent to us
      constexpr unsigned int data_per_face = n_t * n_t + 2 * 2 * n_t * n_n;
      for(unsigned int v = 0; v < n_lanes; ++v)
        if(const Number * remote_data = single_exchange.remote_face_data[cell][f][v])
          for(unsigned int i = 0; i < data_per_face; ++i)
            face_array[i][v] = remote_data[i];

      // Only the normal derivative of the normal component is read from
      // the neighbor, the value and the contribution of the value to the
      // derivative is added below once the values of this cell are known
      const VectorizedArray<Number> * values_dofs = face_array + face_direction * n_t * n_n * 2;
      VectorizedArray<Number> *       values      = nullptr;
      VectorizedArray<Number> *       gradients   = grads_face[1][face_direction];
      eval_iso.template values<0, true, false>(values_dofs, scratch_data);
      eval_iso.template values<1, true, false, dim>(scratch_data, gradients + face_direction);

      if(face_direction == 0)
      {
        values_dofs = face_array + n_t * n_t;
        values      = values_face[1][1];
        gradients   = grads_face[1][1];
      }
      else
      {
        values_dofs = face_array;
        values      = values_face[1][0];
        gradients   = grads_face[1][0];
      }
      eval_aniso.template normal<0>(shape_data[0], values_dofs, values);
      eval_aniso.template tangential<1, 0>(shape_data[1], values, values);
      eval_aniso.template normal<0>(shape_data[0], values_dofs + n_t * n_n, scratch_data);
      eval_aniso.template tangential<1, 0, dim>(shape_data[1],
                                                scratch_data,
                                                gradients + face_direction);

      if(face_direction < 2)
      {
        values_dofs = face_array + n_t * n_t + 2 * n_t * n_n;
        values      = values_face[1][2];
        gradients   = grads_face[1][2];
      }
      else
      {
        values_dofs = face_array + 2 * n_t * n_n;
        values      = values_face[1][1];
        gradients   = grads_face[1][1];
      }
      eval_aniso.template normal<1>(shape_data[0], values_dofs, values);
      eval_aniso.template tangential<0, 1>(shape_data[1], values, values);

      eval_aniso.template normal<1>(shape_data[0], values_dofs + n_t * n_n, scratch_data);
      eval_aniso.template tangential<0, 1, dim>(shape_data[1],
                                                scratch_data,
                                                gradients + face_direction);
    }

    // interpolate from interior to face
    const std::array<Number, 2> * shape = interpolate_quad_to_boundary[f % 2].data();

    for(unsigned int i1 = 0; i1 < n_q_points_1d; ++i1)
      for(unsigned int i0 = 0; i0 < n_q_points_1d; ++i0)
      {
        unsigned int idx = 0;
        if(face_direction == 0)
          idx = n_q_points_1d * (n_q_points_1d * i1 + i0);
        else if(face_direction == 1)
          idx = i0 + i1 * n_q_points_1d * n_q_points_1d;
        else
          idx = i0 + i1 * n_q_points_1d;

        const VectorizedArray<Number> * my_vals = quad_values + idx * dim;
        VectorizedArray<Number>         v0      = shape[0][0] * my_vals[0];
        VectorizedArray<Number>         d0      = shape[0][1] * my_vals[0];
        VectorizedArray<Number>         v1      = shape[0][0] * my_vals[1];
        VectorizedArray<Number>         d1      = shape[0][1] * my_vals[1];
        VectorizedArray<Number>         v2      = shape[0][0] * my_vals[2];
        VectorizedArray<Number>         d2      = shape[0][1] * my_vals[2];
        const unsigned int stride = Utilities::pow(n_q_points_1d, face_direction) * dim;
        for(unsigned int i = 1; i < n_q_points_1d; ++i)
        {
          v0 += shape[i][0] * my_vals[i * stride + 0];
          d0 += shape[i][1] * my_vals[i * stride + 0];
          v1 += shape[i][0] * my_vals[i * stride + 1];
          d1 += shape[i][1] * my_vals[i * stride + 1];
          v2 += shape[i][0] * my_vals[i * stride + 2];
          d2 += shape[i][1] * my_vals[i * stride + 2];
        }
        const unsigned int val_idx   = i1 * n_q_points_1d + i0;
        values_face[0][0][val_idx]   = v0;
        values_face[0][1][val_idx]   = v1;
        values_face[0][2][val_idx]   = v2;
        const unsigned int deriv_idx = val_idx * dim + face_direction;
        grads_face[0][0][deriv_idx]  = d0;
        grads_face[0][1][deriv_idx]  = d1;
        grads_face[0][2][deriv_idx]  = d2;
      }

    if constexpr(compute_exterior)
      add_face_value_to_normal_derivative<n_q_points_face>(f, values_face, grads_face);

    // Only need some tangential derivative in xy plane, as the domain is
    // extruded in z direction. The tangential derivatives of the normal
    // component on the exterior are the same as on the interior side, so
    // skip them.
    if(face_direction < 2)
      for(unsigned int side = 0; side < (compute_exterior ? 2 : 1); ++side)
      {
        for(unsigned int d = 0; d < dim; ++d)
        {
          if(side == 1 && d == face_direction)
            continue;
          const unsigned int first  = face_direction == 0 ? 1 : 0;
          const unsigned int second = face_direction == 1 ? 2 : first + 1;
          eval_g.template gradients<0, true, false, dim>(values_face[side][d],
                                                         grads_face[side][d] + first);
          // This is a logical contradiction to the above if statement, so
          // this is actually never executed, but we keep it for
          // completeness
          if(face_direction == 2)
            eval_g.template gradients<1, true, false, dim>(values_face[side][d],
                                                           grads_face[side][d] + second);
        }
      }

    const VectorizedArray<Number> * values_ext[dim];
    const VectorizedArray<Number> * grads_ext[dim][dim];
    set_exterior_pointers<n_q_points_face>(
      face_direction, values_face, grads_face, values_ext, grads_ext);

    const Number                  h_z         = mapping_info.h_z;
    const Number                  h_z_inverse = mapping_info.h_z_inverse;
    const VectorizedArray<Number> sigmaF      = penalty_parameters[cell][f] * factor_lapl;

    if(face_direction < 2)
    {
      std::array<unsigned int, n_lanes> shifted_data_indices;
      std::array<unsigned int, n_lanes> shifted_data_indices_gr;
      std::array<unsigned int, n_lanes> shifted_data_indices_norm;
      std::array<unsigned int, n_lanes> shifted_data_indices_neigh;
      std::array<unsigned int, n_lanes> shifted_data_indices_gr_neigh;
      for(unsigned int v = 0; v < n_lanes; ++v)
      {
        const unsigned int idx           = mapping_info.face_mapping_data_index[cell][f][0][v];
        shifted_data_indices[v]          = idx * 4;
        shifted_data_indices_gr[v]       = idx * 6;
        shifted_data_indices_norm[v]     = idx * 2;
        const unsigned int neighbor_idx  = mapping_info.face_mapping_data_index[cell][f][1][v];
        shifted_data_indices_neigh[v]    = neighbor_idx * 4;
        shifted_data_indices_gr_neigh[v] = neighbor_idx * 6;
      }

      for(unsigned int q1 = 0; q1 < n_q_points_1d; ++q1)
      {
        Tensor<2, 2, VectorizedArray<Number>> jac_xy[2];
        vectorized_load_and_transpose(4,
                                      &mapping_info.face_jacobians_xy[q1][0][0],
                                      shifted_data_indices.data(),
                                      &jac_xy[0][0][0]);
        vectorized_load_and_transpose(4,
                                      &mapping_info.face_jacobians_xy[q1][0][0],
                                      shifted_data_indices_neigh.data(),
                                      &jac_xy[1][0][0]);

        const VectorizedArray<Number> area_element_xy =
          std::sqrt(jac_xy[0][0][1 - face_direction] * jac_xy[0][0][1 - face_direction] +
                    jac_xy[0][1][1 - face_direction] * jac_xy[0][1][1 - face_direction]);

        const VectorizedArray<Number> inv_jac_det[2] = {Number(1.0) / determinant(jac_xy[0]),
                                                        Number(1.0) / determinant(jac_xy[1])};
        const Tensor<2, 2, VectorizedArray<Number>> inv_jac_xy[2] = {transpose(invert(jac_xy[0])),
                                                                     transpose(invert(jac_xy[1]))};

        Tensor<1, 3, Tensor<1, 2, VectorizedArray<Number>>> jac_grad_xy[2];
        vectorized_load_and_transpose(6,
                                      &mapping_info.face_jacobian_grads[q1][0][0],
                                      shifted_data_indices_gr.data(),
                                      &jac_grad_xy[0][0][0]);
        if(compute_exterior)
          vectorized_load_and_transpose(6,
                                        &mapping_info.face_jacobian_grads[q1][0][0],
                                        shifted_data_indices_gr_neigh.data(),
                                        &jac_grad_xy[1][0][0]);

        Tensor<1, 2, VectorizedArray<Number>> normal;
        vectorized_load_and_transpose(2,
                                      &mapping_info.face_normal_vector_xy[q1][0],
                                      shifted_data_indices_norm.data(),
                                      &normal[0]);

        Tensor<1, 2, VectorizedArray<Number>> jac_x_normal[2] = {normal * inv_jac_xy[0],
                                                                 normal * inv_jac_xy[1]};

        // Prepare for -(J^{-T} * jac_grad * J^{-1} * J * values *
        // det(J^{-1})), which can be expressed as a rank-1 update
        // tmp[d] * tmp4[e], where tmp = J * values and tmp4 =
        // (J^{-T} * jac_grad * J^{-1})
        VectorizedArray<Number> jac_grad_rank1_normal[2];
        for(unsigned int s = 0; s < (compute_exterior ? 2 : 1); ++s)
        {
          VectorizedArray<Number> tmp[2];
          for(unsigned int d = 0; d < 2; ++d)
            tmp[d] = (inv_jac_xy[s][0][d] * jac_grad_xy[s][d][0] +
                      inv_jac_xy[s][1][d] * jac_grad_xy[s][d][1] +
                      inv_jac_xy[s][0][1 - d] * jac_grad_xy[s][2][0] +
                      inv_jac_xy[s][1][1 - d] * jac_grad_xy[s][2][1]);
          jac_grad_rank1_normal[s] = (tmp[0] * jac_x_normal[s][0] + tmp[1] * jac_x_normal[s][1]);
        }

        for(unsigned int qz = 0, q = q1; qz < n_q_points_1d; ++qz, q += n_q_points_1d)
        {
          const VectorizedArray<Number> val[2][3] = {
            {values_face[0][0][q], values_face[0][1][q], values_face[0][2][q]},
            {values_ext[0][q], values_ext[1][q], values_ext[2][q]}};
          const VectorizedArray<Number> grad[2][3][2] = {
            {{grads_face[0][0][q * dim], grads_face[0][0][q * dim + 1]},
             {grads_face[0][1][q * dim], grads_face[0][1][q * dim + 1]},
             {grads_face[0][2][q * dim], grads_face[0][2][q * dim + 1]}},
            {{grads_ext[0][0][q * dim], grads_ext[0][1][q * dim]},
             {grads_ext[1][0][q * dim], grads_ext[1][1][q * dim]},
             {grads_ext[2][0][q * dim], grads_ext[2][1][q * dim]}}};

          Tensor<1, dim, VectorizedArray<Number>> val_real[2];
          Tensor<1, dim, VectorizedArray<Number>> normal_derivatives[2];
          for(unsigned int s = 0; s < (compute_exterior ? 2 : 1); ++s)
          {
            val_real[s][0] = jac_xy[s][0][0] * val[s][0] + jac_xy[s][0][1] * val[s][1];
            val_real[s][1] = jac_xy[s][1][0] * val[s][0] + jac_xy[s][1][1] * val[s][1];

            for(unsigned int d = 0; d < 2; ++d)
            {
              // (J * grad_quad) * J^-1 * det(J^-1), part in
              // braces
              VectorizedArray<Number> tmp[2];
              for(unsigned int e = 0; e < 2; ++e)
                tmp[e] = (jac_xy[s][d][0] * grad[s][0][e] + jac_xy[s][d][1] * grad[s][1][e]);

              // Add (jac_grad * values) * J^{-1} * det(J^{-1}),
              // combine terms outside braces with gradient part
              // from above
              tmp[0] += jac_grad_xy[s][0][d] * val[s][0];
              tmp[0] += jac_grad_xy[s][2][d] * val[s][1];
              tmp[1] += jac_grad_xy[s][1][d] * val[s][1];
              tmp[1] += jac_grad_xy[s][2][d] * val[s][0];

              VectorizedArray<Number> normal_deriv_d =
                tmp[0] * jac_x_normal[s][0] + tmp[1] * jac_x_normal[s][1];

              normal_deriv_d -= jac_grad_rank1_normal[s] * val_real[s][d];

              normal_derivatives[s][d] = (h_z_inverse * inv_jac_det[s]) * normal_deriv_d;
            }

            // jac[2][2] (= h_z) cancels with h_z_inverse contained in
            // determinant
            normal_derivatives[s][2] = inv_jac_det[s] * (jac_x_normal[s][0] * grad[s][2][0] +
                                                         jac_x_normal[s][1] * grad[s][2][1] -
                                                         jac_grad_rank1_normal[s] * val[s][2]);

            for(unsigned int d = 0; d < 2; ++d)
              val_real[s][d] *= inv_jac_det[s] * h_z_inverse;
            val_real[s][2] = inv_jac_det[s] * val[s][2];
          }

          if(compute_exterior)
          {
            val_real[1] =
              boundary_mask * val_real[0] + (Number(1.0) - std::abs(boundary_mask)) * val_real[1];
            normal_derivatives[1] = -boundary_mask * normal_derivatives[0] +
                                    (Number(1.0) - std::abs(boundary_mask)) * normal_derivatives[1];
          }

          // the length element h_z cancels with h_z_inverse of
          // determinant in Piola transformation for test function
          const VectorizedArray<Number> integrate_factor =
            (mapping_info.quad_weights_z[q1] * area_element_xy) * mapping_info.quad_weights_z[qz];

          // physical terms
          const VectorizedArray<Number> effective_factor =
            integrate_factor * make_vectorized_array<Number>(0.5 * factor_lapl);

          const auto viscous_value_flux =
            (integrate_factor * sigmaF) * (val_real[0] - val_real[1]) -
            effective_factor * (normal_derivatives[0] + normal_derivatives[1]);

          const auto viscous_gradient_flux = effective_factor * (val_real[1] - val_real[0]);

          // apply test functions via transpose of the above
          for(unsigned int s = 0; s < (use_face_buffer ? 2 : 1); ++s)
          {
            const auto my_viscous_gradient = viscous_gradient_flux * inv_jac_det[s];
            const auto my_viscous_value =
              viscous_value_flux * (s == 1 ? -inv_jac_det[s] : inv_jac_det[s]);
            VectorizedArray<Number> tmp[2][2];
            for(unsigned int d = 0; d < 2; ++d)
              for(unsigned int e = 0; e < 2; ++e)
                tmp[d][e] = jac_x_normal[s][d] * my_viscous_gradient[e];

            for(unsigned int d = 0; d < 2; ++d)
            {
              grads_face[s][0][q * dim + d] =
                (jac_xy[s][0][0] * tmp[d][0] + jac_xy[s][1][0] * tmp[d][1]);
              grads_face[s][1][q * dim + d] =
                (jac_xy[s][0][1] * tmp[d][0] + jac_xy[s][1][1] * tmp[d][1]);
              grads_face[s][2][q * dim + d] = (h_z * my_viscous_gradient[2]) * jac_x_normal[s][d];
            }

            // jac_grad * (J^{-1} * (grad_in * factor)), re-use
            // part in braces as 'tmp' from above
            VectorizedArray<Number> value_tmp[dim];
            for(unsigned int d = 0; d < 2; ++d)
              value_tmp[d] =
                (tmp[d][0] * jac_grad_xy[s][d][0] + tmp[d][1] * jac_grad_xy[s][d][1] +
                 tmp[1 - d][0] * jac_grad_xy[s][2][0] + tmp[1 - d][1] * jac_grad_xy[s][2][1]);

            //   -(grad_in * factor) * J * (J^{-T} * jac_grad * J^{-1})
            // = -(grad_in * factor) * J * ( \------- tmp4 ---------/ )
            for(unsigned int d = 0; d < 2; ++d)
            {
              VectorizedArray<Number> tmp2 =
                my_viscous_value[d] - my_viscous_gradient[d] * jac_grad_rank1_normal[s];
              for(unsigned int e = 0; e < 2; ++e)
                value_tmp[e] += jac_xy[s][d][e] * tmp2;
            }
            values_face[s][0][q] = value_tmp[0];
            values_face[s][1][q] = value_tmp[1];
            values_face[s][2][q] =
              h_z * (my_viscous_value[2] - my_viscous_gradient[2] * jac_grad_rank1_normal[s]);
          }
        }
      }
    }
    else
    {
      std::array<unsigned int, n_lanes> shifted_data_indices;
      for(unsigned int v = 0; v < n_lanes; ++v)
        shifted_data_indices[v] = mapping_info.mapping_data_index[cell][v] * 4;

      const Number normal_sign = (f % 2) ? 1 : -1;

      for(unsigned int q = 0; q < n_q_points_2d; ++q)
      {
        Tensor<2, 2, VectorizedArray<Number>> jac_xy;
        vectorized_load_and_transpose(4,
                                      &mapping_info.jacobians_xy[q][0][0],
                                      shifted_data_indices.data(),
                                      &jac_xy[0][0]);
        const VectorizedArray<Number> inv_jac_det = Number(1.0) / determinant(jac_xy);

        const VectorizedArray<Number> val[2][3] = {
          {values_face[0][0][q], values_face[0][1][q], values_face[0][2][q]},
          {values_ext[0][q], values_ext[1][q], values_ext[2][q]}};
        const VectorizedArray<Number> grad[2][3] = {{grads_face[0][0][q * dim + 2],
                                                     grads_face[0][1][q * dim + 2],
                                                     grads_face[0][2][q * dim + 2]},
                                                    {grads_ext[0][2][q * dim],
                                                     grads_ext[1][2][q * dim],
                                                     grads_ext[2][2][q * dim]}};

        Tensor<1, dim, VectorizedArray<Number>> val_real[2];
        Tensor<1, dim, VectorizedArray<Number>> normal_derivatives[2];
        for(unsigned int s = 0; s < (compute_exterior ? 2 : 1); ++s)
        {
          for(unsigned int d = 0; d < 2; ++d)
            normal_derivatives[s][d] = (h_z_inverse * h_z_inverse * normal_sign * inv_jac_det) *
                                       (jac_xy[d][0] * grad[s][0] + jac_xy[d][1] * grad[s][1]);
          normal_derivatives[s][2] = (h_z_inverse * normal_sign * inv_jac_det) * grad[s][2];

          for(unsigned int d = 0; d < 2; ++d)
            val_real[s][d] =
              (h_z_inverse * inv_jac_det) * (jac_xy[d][0] * val[s][0] + jac_xy[d][1] * val[s][1]);

          // h_z cancels with factor h_z_inverse from Jacobian
          // determinant
          val_real[s][2] = inv_jac_det * val[s][2];
        }

        const VectorizedArray<Number> integrate_factor =
          mapping_info.quad_weights_xy[q] * determinant(jac_xy);

        // physical terms
        const VectorizedArray<Number> effective_factor =
          integrate_factor * make_vectorized_array<Number>(0.5 * factor_lapl);
        const auto viscous_value_flux =
          (integrate_factor * sigmaF) * (val_real[0] - val_real[1]) -
          effective_factor * (normal_derivatives[0] + normal_derivatives[1]);

        const auto viscous_gradient_flux = effective_factor * (val_real[1] - val_real[0]);

        // transpose of evaluate part
        for(unsigned int d = 0; d < 2; ++d)
          values_face[0][d][q] =
            (h_z_inverse * inv_jac_det) *
            (jac_xy[0][d] * viscous_value_flux[0] + jac_xy[1][d] * viscous_value_flux[1]);
        values_face[0][2][q] = inv_jac_det * viscous_value_flux[2];
        for(unsigned int d = 0; d < 2; ++d)
          grads_face[0][d][q * dim + 2] =
            (h_z_inverse * h_z_inverse * normal_sign * inv_jac_det) *
            (jac_xy[0][d] * viscous_gradient_flux[0] + jac_xy[1][d] * viscous_gradient_flux[1]);
        grads_face[0][2][q * dim + 2] =
          (h_z_inverse * normal_sign * inv_jac_det) * viscous_gradient_flux[2];
        if(use_face_buffer)
        {
          for(unsigned int d = 0; d < 3; ++d)
            values_face[1][d][q] = -values_face[0][d][q];
          for(unsigned int d = 0; d < 3; ++d)
            grads_face[1][d][q * dim + 2] = grads_face[0][d][q * dim + 2];
        }
      }
    }

    if(face_direction < 2)
      for(unsigned int s = 0; s < ((use_face_buffer && f % 2) ? 2 : 1); ++s)
        for(unsigned int d = 0; d < dim; ++d)
        {
          const unsigned int first  = face_direction == 0 ? 1 : 0;
          const unsigned int second = face_direction == 1 ? 2 : first + 1;
          eval_g.template gradients<0, false, true, dim>(grads_face[s][d] + first,
                                                         values_face[s][d]);
          if(face_direction == 2)
            eval_g.template gradients<1, false, true, dim>(grads_face[s][d] + second,
                                                           values_face[s][d]);
        }

    // fill flux buffer
    if(use_face_buffer && f % 2)
    {
      for(unsigned int d = 0; d < dim; ++d)
        vectorized_transpose_and_store(false,
                                       n_q_points_face,
                                       values_face[1][d],
                                       face_flux_buffer_index[cell][f].data(),
                                       face_flux_buffer.data() + d * n_q_points_face);
      for(unsigned int d = 0; d < dim; ++d)
      {
        for(unsigned int q = 0; q < n_q_points_face; ++q)
          values_face[1][0][q] = grads_face[1][d][q * dim + face_direction];
        vectorized_transpose_and_store(false,
                                       n_q_points_face,
                                       values_face[1][0],
                                       face_flux_buffer_index[cell][f].data(),
                                       face_flux_buffer.data() + (dim + d) * n_q_points_face);
      }
    }
    if(use_face_buffer && all_left_face_fluxes_from_buffer[cell][face_direction])
    {
      for(unsigned int d = 0; d < dim; ++d)
      {
        vectorized_load_and_transpose(n_q_points_face,
                                      face_flux_buffer.data() + (dim + d) * n_q_points_face,
                                      face_flux_buffer_index[cell][f - 1].data(),
                                      values_face[1][0]);
        for(unsigned int q = 0; q < n_q_points_face; ++q)
          grads_face[1][d][q * dim + face_direction] = values_face[1][0][q];
      }
      for(unsigned int d = 0; d < dim; ++d)
        vectorized_load_and_transpose(n_q_points_face,
                                      face_flux_buffer.data() + d * n_q_points_face,
                                      face_flux_buffer_index[cell][f - 1].data(),
                                      values_face[1][d]);
    }

    const std::array<Number, 2> * shape_o = interpolate_quad_to_boundary[0].data();

    // interpolate from face to interior
    const unsigned int stride = Utilities::pow(n_q_points_1d, face_direction) * dim;
    if(use_face_buffer && f % 2 == 1 && all_left_face_fluxes_from_buffer[cell][face_direction])
      for(unsigned int i1 = 0; i1 < n_q_points_1d; ++i1)
        for(unsigned int i0 = 0; i0 < n_q_points_1d; ++i0)
        {
          unsigned int idx;
          if(face_direction == 0)
            idx = n_q_points_1d * (n_q_points_1d * i1 + i0);
          else if(face_direction == 1)
            idx = i0 + i1 * n_q_points_1d * n_q_points_1d;
          else
            idx = i0 + i1 * n_q_points_1d;

          VectorizedArray<Number> *     vals      = out_values + idx * dim;
          const unsigned int            val_idx   = i1 * n_q_points_1d + i0;
          const unsigned int            deriv_idx = val_idx * dim + face_direction;
          const VectorizedArray<Number> v0        = values_face[0][0][val_idx];
          const VectorizedArray<Number> v1        = values_face[0][1][val_idx];
          const VectorizedArray<Number> v2        = values_face[0][2][val_idx];
          const VectorizedArray<Number> d0        = grads_face[0][0][deriv_idx];
          const VectorizedArray<Number> d1        = grads_face[0][1][deriv_idx];
          const VectorizedArray<Number> d2        = grads_face[0][2][deriv_idx];
          const VectorizedArray<Number> vo0       = values_face[1][0][val_idx];
          const VectorizedArray<Number> vo1       = values_face[1][1][val_idx];
          const VectorizedArray<Number> vo2       = values_face[1][2][val_idx];
          const VectorizedArray<Number> do0       = grads_face[1][0][deriv_idx];
          const VectorizedArray<Number> do1       = grads_face[1][1][deriv_idx];
          const VectorizedArray<Number> do2       = grads_face[1][2][deriv_idx];
          for(unsigned int i = 0; i < n_q_points_1d; ++i)
          {
            vals[i * stride + 0] +=
              shape[i][0] * v0 + shape[i][1] * d0 + shape_o[i][0] * vo0 + shape_o[i][1] * do0;
            vals[i * stride + 1] +=
              shape[i][0] * v1 + shape[i][1] * d1 + shape_o[i][0] * vo1 + shape_o[i][1] * do1;
            vals[i * stride + 2] +=
              shape[i][0] * v2 + shape[i][1] * d2 + shape_o[i][0] * vo2 + shape_o[i][1] * do2;
          }
        }
    else
      for(unsigned int i1 = 0; i1 < n_q_points_1d; ++i1)
        for(unsigned int i0 = 0; i0 < n_q_points_1d; ++i0)
        {
          unsigned int idx;
          if(face_direction == 0)
            idx = n_q_points_1d * (n_q_points_1d * i1 + i0);
          else if(face_direction == 1)
            idx = i0 + i1 * n_q_points_1d * n_q_points_1d;
          else
            idx = i0 + i1 * n_q_points_1d;

          VectorizedArray<Number> *     vals      = out_values + idx * dim;
          const unsigned int            val_idx   = i1 * n_q_points_1d + i0;
          const VectorizedArray<Number> v0        = values_face[0][0][val_idx];
          const VectorizedArray<Number> v1        = values_face[0][1][val_idx];
          const VectorizedArray<Number> v2        = values_face[0][2][val_idx];
          const unsigned int            deriv_idx = val_idx * dim + face_direction;
          const VectorizedArray<Number> d0        = grads_face[0][0][deriv_idx];
          const VectorizedArray<Number> d1        = grads_face[0][1][deriv_idx];
          const VectorizedArray<Number> d2        = grads_face[0][2][deriv_idx];
          for(unsigned int i = 0; i < n_q_points_1d; ++i)
          {
            vals[i * stride + 0] += shape[i][0] * v0 + shape[i][1] * d0;
            vals[i * stride + 1] += shape[i][0] * v1 + shape[i][1] * d1;
            vals[i * stride + 2] += shape[i][0] * v2 + shape[i][1] * d2;
          }
        }
  }

  template<bool include_convective_divergence, int degree, int n_q_points_1d>
  void
  compute_face_convective(
    const std::vector<internal::MatrixFreeFunctions::UnivariateShapeData<Number>> & shape_data,
    const VectorType &                                                              src,
    const unsigned int                                                              cell,
    const unsigned int                                                              f,
    const Number                    factor_convective,
    const VectorizedArray<Number> * quad_values,
    const VectorizedArray<Number> * dof_values,
    VectorizedArray<Number> *       face_data) const
  {
    static_assert(dim == 3, "Only implemented for 3D");

    // The face integrals do not add their contributions to the cell, but
    // store the terms tested by the velocity test functions (dim
    // components) and, for the convective divergence, the pressure test
    // functions at the face quadrature points in 'face_data', with (dim + 1)
    // * n_q_points_face entries per face. The face quadrature points are
    // numbered by the first tangential direction (y for faces in x
    // direction, x otherwise), followed by the second one (y for faces in z
    // direction, z otherwise). compute_cell() applies the test functions
    // when it writes the cell terms, see add_face_terms_velocity().
    //
    // Like for the viscous term in compute_face(), the face integrals on the
    // right face of a cell (odd f) also compute the contribution of the test
    // functions of the neighbor and store it in the flux buffer, from where
    // the neighbor picks it up into the data of its left face when computing
    // its right face
    if(use_convective_face_flux_buffer && f % 2 == 0 &&
       all_left_face_fluxes_from_buffer[cell][f / 2])
      return;

    constexpr unsigned int n_q_points_2d = n_q_points_1d * n_q_points_1d;
    AssertThrow(n_q_points_2d == convective_mapping_info.quad_weights_xy.size(),
                ExcDimensionMismatch(n_q_points_2d,
                                     convective_mapping_info.quad_weights_xy.size()));
    constexpr unsigned int n_lanes = VectorizedArray<Number>::size();

    ndarray<unsigned int, 2 * dim + 1, n_lanes> dof_indices_neighbor;
    unsigned int                                lane_with_neighbor = 0;
    for(unsigned int v = 0; v < n_lanes; ++v)
      if(neighbor_cells[cell][f][v] != numbers::invalid_unsigned_int)
      {
        lane_with_neighbor = v;
        break;
      }

    // TODO: 0: interior face, 1: Neumann, -1: Dirichlet, apply from actual b.c. in operator
    VectorizedArray<Number> boundary_mask;
    for(unsigned int v = 0; v < n_lanes; ++v)
      if(neighbor_cells[cell][f][v] == numbers::invalid_unsigned_int)
        boundary_mask[v] = -1.;
      else
        boundary_mask[v] = 0.;

    for(unsigned int v = 0; v < n_lanes; ++v)
    {
      // use a dummy neighbor with nearby data for cells at boundary to make
      // sure vectorized functions continue working
      const unsigned int neighbor_idx =
        neighbor_cells[cell][f][v] != numbers::invalid_unsigned_int ?
          neighbor_cells[cell][f][v] :
          neighbor_cells[cell][f][lane_with_neighbor];
      if(neighbor_idx != numbers::invalid_unsigned_int)
        for(unsigned int i = 0; i < 2 * dim + 1; ++i)
          dof_indices_neighbor[i][v] =
            dof_indices[neighbor_idx / n_lanes][i][neighbor_idx % n_lanes];
      else
        for(unsigned int i = 0; i < 2 * dim + 1; ++i)
          dof_indices_neighbor[i][v] = dof_indices[0][i][0];
    }

    VectorizedArray<Number> face_array[dim * 2 * Utilities::pow(n_q_points_1d, dim - 1)];
    constexpr int           n_n = degree + 1;
    constexpr int           n_t = degree;

    internal::EvaluatorTensorProduct<internal::evaluate_evenodd,
                                     dim - 1,
                                     n_q_points_1d,
                                     n_q_points_1d,
                                     VectorizedArray<Number>,
                                     Number>
      eval_g({}, shape_data[0].shape_gradients_collocation_eo.data(), {});
    internal::EvaluatorTensorProduct<internal::evaluate_evenodd,
                                     dim - 1,
                                     degree,
                                     n_q_points_1d,
                                     VectorizedArray<Number>,
                                     Number>
      eval_iso(shape_data[1].shape_values_eo.data(), {}, {});
    internal::EvaluatorTensorProductAnisotropic<dim - 1, degree, n_q_points_1d, true> eval_aniso;

    const unsigned int      face_direction  = f / 2;
    constexpr unsigned int  n_q_points_face = Utilities::pow(n_q_points_1d, dim - 1);
    VectorizedArray<Number> scratch_data[n_q_points_face + 1];
    VectorizedArray<Number> values_face[2][dim][n_q_points_face + 1];
    VectorizedArray<Number> grads_face[2][dim][n_q_points_face * dim + 1];

    VectorizedArray<Number> * my_face_data = face_data + f * (dim + 1) * n_q_points_face;

    // flux of the convective divergence tested by the pressure functions of
    // the neighbor
    VectorizedArray<Number> pressure_flux_ext[n_q_points_face];
    const bool              fill_buffer = use_convective_face_flux_buffer && f % 2 == 1;

    auto neighbor_entities = entity_pointers(src.begin(), dof_indices_neighbor);
    set_remote_cell_entities(src, cell, f, neighbor_entities);
    if(include_convective_divergence)
      read_face_values<degree>(neighbor_entities, f + (f % 2 ? -1 : 1), face_array);
    else
      for(unsigned int lane = 0; lane < n_lanes; ++lane)
        read_face_values_only<degree>(neighbor_entities, f + (f % 2 ? -1 : 1), lane, face_array);

    // For faces located on MPI-remote processes, we replace the dummy
    // data read out above by the data sent to us, which contains the values
    // and derivatives of the tangential components and the normal
    // derivative of the normal component. For the convective term without
    // divergence, we only use the values of the tangential components,
    // which are stored for each tangential component before its derivatives
    // and after the normal derivative of the normal component if that comes
    // first.
    constexpr unsigned int data_per_face = n_t * n_t + 2 * 2 * n_t * n_n;
    constexpr unsigned int n_values      = n_t * n_n;
    const unsigned int     first_values  = face_direction == 0 ? n_t * n_t : 0;
    const unsigned int     second_values =
      face_direction == 2 ? 2 * n_values : n_t * n_t + 2 * n_values;
    for(unsigned int v = 0; v < n_lanes; ++v)
      if(const Number * remote_data = single_exchange.remote_face_data[cell][f][v])
      {
        if(include_convective_divergence)
          for(unsigned int i = 0; i < data_per_face; ++i)
            face_array[i][v] = remote_data[i];
        else
          for(unsigned int i = 0; i < n_values; ++i)
          {
            face_array[i][v]            = remote_data[first_values + i];
            face_array[n_values + i][v] = remote_data[second_values + i];
          }
      }

    // Interpolate the face data in the layout of read_face_values() (with
    // derivatives) or read_face_values_only() to the face quadrature points
    // of the interior (side = 0) or exterior (side = 1) side. For the
    // exterior side, only the normal derivative of the normal component is
    // read from the neighbor, its value and the contribution of the value to
    // the derivative are taken from the interior side below.
    const auto interpolate_face_dofs = [&](const unsigned int              side,
                                           const VectorizedArray<Number> * face_array) {
      if constexpr(include_convective_divergence)
      {
        const VectorizedArray<Number> * values_dofs = face_array + face_direction * n_t * n_n * 2;
        VectorizedArray<Number> *       values      = nullptr;
        VectorizedArray<Number> *       gradients   = grads_face[side][face_direction];
        eval_iso.template values<0, true, false>(values_dofs, scratch_data);
        eval_iso.template values<1, true, false, dim>(scratch_data, gradients + face_direction);

        if(face_direction == 0)
        {
          values_dofs = face_array + n_t * n_t;
          values      = values_face[side][1];
          gradients   = grads_face[side][1];
        }
        else
        {
          values_dofs = face_array;
          values      = values_face[side][0];
          gradients   = grads_face[side][0];
        }
        eval_aniso.template normal<0>(shape_data[0], values_dofs, values);
        eval_aniso.template tangential<1, 0>(shape_data[1], values, values);
        eval_aniso.template normal<0>(shape_data[0], values_dofs + n_t * n_n, scratch_data);
        eval_aniso.template tangential<1, 0, dim>(shape_data[1],
                                                  scratch_data,
                                                  gradients + face_direction);

        if(face_direction < 2)
        {
          values_dofs = face_array + n_t * n_t + 2 * n_t * n_n;
          values      = values_face[side][2];
          gradients   = grads_face[side][2];
        }
        else
        {
          values_dofs = face_array + 2 * n_t * n_n;
          values      = values_face[side][1];
          gradients   = grads_face[side][1];
        }
        eval_aniso.template normal<1>(shape_data[0], values_dofs, values);
        eval_aniso.template tangential<0, 1>(shape_data[1], values, values);

        eval_aniso.template normal<1>(shape_data[0], values_dofs + n_t * n_n, scratch_data);
        eval_aniso.template tangential<0, 1, dim>(shape_data[1],
                                                  scratch_data,
                                                  gradients + face_direction);
      }
      else
      {
        const VectorizedArray<Number> * values_dofs = face_array;
        VectorizedArray<Number> *       values = values_face[side][face_direction == 0 ? 1 : 0];
        eval_aniso.template normal<0>(shape_data[0], values_dofs, values);
        eval_aniso.template tangential<1, 0>(shape_data[1], values, values);

        values_dofs = face_array + n_t * n_n;
        values      = values_face[side][face_direction < 2 ? 2 : 1];
        eval_aniso.template normal<1>(shape_data[0], values_dofs, values);
        eval_aniso.template tangential<0, 1>(shape_data[1], values, values);
      }
    };
    interpolate_face_dofs(1, face_array);

    // The data of the interior side: for higher degrees from the dofs of
    // this cell, with the value of the normal component (and its normal
    // derivative including the dofs on the face) after the data of
    // read_face_values(). For low degrees, the interpolation from the
    // values at the cell quadrature points is faster.
    if constexpr(use_cell_dofs_for_faces(degree))
    {
      constexpr unsigned int n_data_own =
        (include_convective_divergence ? 2 * n_t * n_t + 4 * n_t * n_n : n_t * n_t + 2 * n_t * n_n);
      VectorizedArray<Number> face_array_own[n_data_own];
      if(face_direction == 0)
        interpolate_cell_dofs_to_face<n_t, 0, include_convective_divergence>(dof_values,
                                                                             f % 2,
                                                                             face_array_own);
      else if(face_direction == 1)
        interpolate_cell_dofs_to_face<n_t, 1, include_convective_divergence>(dof_values,
                                                                             f % 2,
                                                                             face_array_own);
      else
        interpolate_cell_dofs_to_face<n_t, 2, include_convective_divergence>(dof_values,
                                                                             f % 2,
                                                                             face_array_own);
      interpolate_face_dofs(0, face_array_own);
      eval_iso.template values<0, true, false>(face_array_own + n_data_own - n_t * n_t,
                                               scratch_data);
      eval_iso.template values<1, true, false>(scratch_data, values_face[0][face_direction]);
    }
    else
    {
      // interpolate from interior to face
      const std::array<Number, 2> * shape = convective_interpolate_quad_to_boundary[f % 2].data();

      for(unsigned int i1 = 0; i1 < n_q_points_1d; ++i1)
        for(unsigned int i0 = 0; i0 < n_q_points_1d; ++i0)
        {
          unsigned int idx = 0;
          if(face_direction == 0)
            idx = n_q_points_1d * (n_q_points_1d * i1 + i0);
          else if(face_direction == 1)
            idx = i0 + i1 * n_q_points_1d * n_q_points_1d;
          else
            idx = i0 + i1 * n_q_points_1d;

          const VectorizedArray<Number> * my_vals = quad_values + idx * dim;
          VectorizedArray<Number>         v0      = shape[0][0] * my_vals[0];
          VectorizedArray<Number>         v1      = shape[0][0] * my_vals[1];
          VectorizedArray<Number>         v2      = shape[0][0] * my_vals[2];
          const unsigned int stride = Utilities::pow(n_q_points_1d, face_direction) * dim;
          for(unsigned int i = 1; i < n_q_points_1d; ++i)
          {
            v0 += shape[i][0] * my_vals[i * stride + 0];
            v1 += shape[i][0] * my_vals[i * stride + 1];
            v2 += shape[i][0] * my_vals[i * stride + 2];
          }
          const unsigned int val_idx = i1 * n_q_points_1d + i0;
          values_face[0][0][val_idx] = v0;
          values_face[0][1][val_idx] = v1;
          values_face[0][2][val_idx] = v2;
          if(include_convective_divergence)
          {
            VectorizedArray<Number> d0 = shape[0][1] * my_vals[0];
            VectorizedArray<Number> d1 = shape[0][1] * my_vals[1];
            VectorizedArray<Number> d2 = shape[0][1] * my_vals[2];
            for(unsigned int i = 1; i < n_q_points_1d; ++i)
            {
              d0 += shape[i][1] * my_vals[i * stride + 0];
              d1 += shape[i][1] * my_vals[i * stride + 1];
              d2 += shape[i][1] * my_vals[i * stride + 2];
            }
            const unsigned int deriv_idx = val_idx * dim + face_direction;
            grads_face[0][0][deriv_idx]  = d0;
            grads_face[0][1][deriv_idx]  = d1;
            grads_face[0][2][deriv_idx]  = d2;
          }
        }
    }

    if(include_convective_divergence)
    {
      add_face_value_to_normal_derivative<n_q_points_face>(f, values_face, grads_face);

      // The tangential derivatives of the normal component on the exterior
      // are the same as on the interior side, so skip them
      for(unsigned int side = 0; side < 2; ++side)
      {
        for(unsigned int d = 0; d < dim; ++d)
        {
          if(side == 1 && d == face_direction)
            continue;
          const unsigned int first  = face_direction == 0 ? 1 : 0;
          const unsigned int second = face_direction == 1 ? 2 : first + 1;
          eval_g.template gradients<0, true, false, dim>(values_face[side][d],
                                                         grads_face[side][d] + first);
          eval_g.template gradients<1, true, false, dim>(values_face[side][d],
                                                         grads_face[side][d] + second);
        }
      }
    }

    const VectorizedArray<Number> * values_ext[dim];
    const VectorizedArray<Number> * grads_ext[dim][dim];
    set_exterior_pointers<n_q_points_face>(
      face_direction, values_face, grads_face, values_ext, grads_ext);

    const Number h_z         = mapping_info.h_z;
    const Number h_z_inverse = mapping_info.h_z_inverse;

    if(face_direction < 2)
    {
      std::array<unsigned int, n_lanes> shifted_data_indices;
      std::array<unsigned int, n_lanes> shifted_data_indices_gr;
      std::array<unsigned int, n_lanes> shifted_data_indices_norm;
      std::array<unsigned int, n_lanes> shifted_data_indices_neigh;
      std::array<unsigned int, n_lanes> shifted_data_indices_gr_neigh;
      for(unsigned int v = 0; v < n_lanes; ++v)
      {
        const unsigned int idx     = convective_mapping_info.face_mapping_data_index[cell][f][0][v];
        shifted_data_indices[v]    = idx * 4;
        shifted_data_indices_gr[v] = idx * 6;
        shifted_data_indices_norm[v] = idx * 2;
        const unsigned int neighbor_idx =
          convective_mapping_info.face_mapping_data_index[cell][f][1][v];
        shifted_data_indices_neigh[v]    = neighbor_idx * 4;
        shifted_data_indices_gr_neigh[v] = neighbor_idx * 6;
      }

      for(unsigned int q1 = 0; q1 < n_q_points_1d; ++q1)
      {
        Tensor<2, 2, VectorizedArray<Number>> jac_xy[2];
        vectorized_load_and_transpose(4,
                                      &convective_mapping_info.face_jacobians_xy[q1][0][0],
                                      shifted_data_indices.data(),
                                      &jac_xy[0][0][0]);
        vectorized_load_and_transpose(4,
                                      &convective_mapping_info.face_jacobians_xy[q1][0][0],
                                      shifted_data_indices_neigh.data(),
                                      &jac_xy[1][0][0]);

        const VectorizedArray<Number> area_element_xy =
          std::sqrt(jac_xy[0][0][1 - face_direction] * jac_xy[0][0][1 - face_direction] +
                    jac_xy[0][1][1 - face_direction] * jac_xy[0][1][1 - face_direction]);

        const VectorizedArray<Number> inv_jac_det[2] = {Number(1.0) / determinant(jac_xy[0]),
                                                        Number(1.0) / determinant(jac_xy[1])};
        const Tensor<2, 2, VectorizedArray<Number>> inv_jac_xy[2] = {transpose(invert(jac_xy[0])),
                                                                     transpose(invert(jac_xy[1]))};

        Tensor<1, 3, Tensor<1, 2, VectorizedArray<Number>>> jac_grad_xy[2];
        if(include_convective_divergence)
        {
          vectorized_load_and_transpose(6,
                                        &convective_mapping_info.face_jacobian_grads[q1][0][0],
                                        shifted_data_indices_gr.data(),
                                        &jac_grad_xy[0][0][0]);
          vectorized_load_and_transpose(6,
                                        &convective_mapping_info.face_jacobian_grads[q1][0][0],
                                        shifted_data_indices_gr_neigh.data(),
                                        &jac_grad_xy[1][0][0]);
        }

        Tensor<1, 2, VectorizedArray<Number>> normal;
        vectorized_load_and_transpose(2,
                                      &convective_mapping_info.face_normal_vector_xy[q1][0],
                                      shifted_data_indices_norm.data(),
                                      &normal[0]);
        VectorizedArray<Number> jac_grad_rank1[2][2];
        if(include_convective_divergence)
        {
          for(unsigned int s = 0; s < 2; ++s)
          {
            VectorizedArray<Number> tmp[2];
            for(unsigned int d = 0; d < 2; ++d)
              tmp[d] = (inv_jac_xy[s][0][d] * jac_grad_xy[s][d][0] +
                        inv_jac_xy[s][1][d] * jac_grad_xy[s][d][1] +
                        inv_jac_xy[s][0][1 - d] * jac_grad_xy[s][2][0] +
                        inv_jac_xy[s][1][1 - d] * jac_grad_xy[s][2][1]);
            for(unsigned int d = 0; d < 2; ++d)
              jac_grad_rank1[s][d] = (tmp[0] * inv_jac_xy[s][d][0] + tmp[1] * inv_jac_xy[s][d][1]);
          }
        }

        for(unsigned int qz = 0, q = q1; qz < n_q_points_1d; ++qz, q += n_q_points_1d)
        {
          const VectorizedArray<Number> val[2][3] = {
            {values_face[0][0][q], values_face[0][1][q], values_face[0][2][q]},
            {values_ext[0][q], values_ext[1][q], values_ext[2][q]}};

          Tensor<1, dim, VectorizedArray<Number>> val_real[2];
          for(unsigned int s = 0; s < 2; ++s)
          {
            val_real[s][0] = jac_xy[s][0][0] * val[s][0] + jac_xy[s][0][1] * val[s][1];
            val_real[s][1] = jac_xy[s][1][0] * val[s][0] + jac_xy[s][1][1] * val[s][1];

            for(unsigned int d = 0; d < 2; ++d)
              val_real[s][d] *= inv_jac_det[s] * h_z_inverse;
            val_real[s][2] = inv_jac_det[s] * val[s][2];
          }

          // At boundary, set exterior value to zero
          {
            val_real[1] = (Number(1.0) - std::abs(boundary_mask)) * val_real[1];
          }

          // the length element h_z cancels with h_z_inverse of
          // determinant in Piola transformation for test function
          const VectorizedArray<Number> integrate_factor =
            (convective_mapping_info.quad_weights_z[q1] * area_element_xy) *
            convective_mapping_info.quad_weights_z[qz];

          if(include_convective_divergence)
          {
            // For the pressure term, ignore entries at the Dirichlet boundary
            // - besides the val_real[1] that is already zero due to code
            // above, also set the interior value to zero to avoid adding
            // anything here
            const Tensor<1, dim, VectorizedArray<Number>> my_val_boundary =
              (Number(1.0) - std::abs(boundary_mask)) * val_real[0];
            const unsigned int            qd            = q * dim;
            const VectorizedArray<Number> grad[2][3][3] = {
              {{grads_face[0][0][qd], grads_face[0][0][qd + 1], grads_face[0][0][qd + 2]},
               {grads_face[0][1][qd], grads_face[0][1][qd + 1], grads_face[0][1][qd + 2]},
               {grads_face[0][2][qd], grads_face[0][2][qd + 1], grads_face[0][2][qd + 2]}},
              {{grads_ext[0][0][qd], grads_ext[0][1][qd], grads_ext[0][2][qd]},
               {grads_ext[1][0][qd], grads_ext[1][1][qd], grads_ext[1][2][qd]},
               {grads_ext[2][0][qd], grads_ext[2][1][qd], grads_ext[2][2][qd]}}};

            // skip extruded part, second slot [dim] -> [2]
            VectorizedArray<Number> grad_real[2][2][dim];
            for(unsigned int s = 0; s < 2; ++s)
            {
              for(unsigned int d = 0; d < 2; ++d)
              {
                // (J * grad_quad) * J^-1 * det(J^-1), part in braces
                VectorizedArray<Number> tmp[dim];
                for(unsigned int e = 0; e < dim; ++e)
                  tmp[e] = (jac_xy[s][d][0] * grad[s][0][e] + jac_xy[s][d][1] * grad[s][1][e]);

                // Add (jac_grad * values) * J^{-1} * det(J^{-1}),
                // combine terms outside braces with gradient part from
                // above
                tmp[0] += jac_grad_xy[s][0][d] * val[s][0];
                tmp[0] += jac_grad_xy[s][2][d] * val[s][1];
                tmp[1] += jac_grad_xy[s][1][d] * val[s][1];
                tmp[1] += jac_grad_xy[s][2][d] * val[s][0];

                for(unsigned int e = 0; e < 2; ++e)
                  grad_real[s][d][e] =
                    (tmp[0] * inv_jac_xy[s][e][0] + tmp[1] * inv_jac_xy[s][e][1]);

                VectorizedArray<Number> val_tmp[2];
                val_tmp[0] = jac_xy[s][0][0] * val[s][0] + jac_xy[s][0][1] * val[s][1];
                val_tmp[1] = jac_xy[s][1][0] * val[s][0] + jac_xy[s][1][1] * val[s][1];
                for(unsigned int e = 0; e < 2; ++e)
                  grad_real[s][d][e] -= jac_grad_rank1[s][e] * val_tmp[d];

                grad_real[s][d][2] = h_z_inverse * tmp[2];
              }
            }

            // skip extruded part [dim] -> [2] because normal is zero in that direction
            VectorizedArray<Number> avg_u_grad_u[2];
            if(q == 1)
            {
              // std::cout << "my  " << f << " ";
              // for (unsigned int d = 0; d < 3; ++d)
              // for (unsigned int e = 0; e < 3; ++e)
              //  std::cout << grad_real[0][d][e] * h_z_inverse * inv_jac_det[0] << "  ";
              // std::cout << std::endl;
              // for (unsigned int d = 0; d < 3; ++d)
              // for (unsigned int e = 0; e < 3; ++e)
              //  std::cout << grad_real[1][d][e] * h_z_inverse * inv_jac_det[1] << "  ";
              // std::cout << std::endl;
              // for (unsigned int d = 0; d < 3; ++d)
              // std::cout << val_real[0][d] << "  ";
              // for (unsigned int d = 0; d < 3; ++d)
              // std::cout << val_real[1][d] << "  ";
              // std::cout << std::endl;
            }
            for(unsigned int d = 0; d < 2; ++d)
            {
              VectorizedArray<Number> sum_0 = grad_real[0][d][0] * my_val_boundary[0];
              for(unsigned int e = 1; e < dim; ++e)
                sum_0 += grad_real[0][d][e] * my_val_boundary[e];
              VectorizedArray<Number> sum_1 = grad_real[1][d][0] * val_real[1][0];
              for(unsigned int e = 1; e < dim; ++e)
                sum_1 += grad_real[1][d][e] * val_real[1][e];
              // if (q == 1)
              // std::cout << sum_0 * h_z_inverse * inv_jac_det[0] << "    " << sum_1 *
              // h_z_inverse
              // * inv_jac_det[1] << "   ";
              avg_u_grad_u[d] =
                Number(0.5) * h_z_inverse * (sum_0 * inv_jac_det[0] + sum_1 * inv_jac_det[1]);
            }
            const VectorizedArray<Number> flux_term =
              (avg_u_grad_u[0] * normal[0] + avg_u_grad_u[1] * normal[1]) * integrate_factor * h_z;
            // if (q == 1)
            //  std::cout << flux_term << std::endl;

            my_face_data[dim * n_q_points_face + q] = flux_term;

            // the average is the same on the neighbor, but the normal
            // points in the opposite direction
            if(fill_buffer)
              pressure_flux_ext[q] = -flux_term;
          }

          // physical terms: for normal speed use normal continuity of RT
          // space, so only pick minus values
          VectorizedArray<Number> normal_speed =
            (val_real[0][0] * normal[0] + val_real[0][1] * normal[1]);

          // test functions of the neighbor, which sees the normal speed
          // with the opposite sign and the two sides swapped
          if(fill_buffer)
          {
            VectorizedArray<Number> flux_ext[dim];
            for(unsigned int d = 0; d < dim; ++d)
              flux_ext[d] = integrate_factor *
                            (Number(0.5) * (std::abs(normal_speed) + normal_speed) *
                               (val_real[1][d] - val_real[0][d]) -
                             (Number(1.0) - factor_convective) * normal_speed * val_real[1][d]);
            const VectorizedArray<Number> tmp0 = flux_ext[0] * inv_jac_det[1];
            const VectorizedArray<Number> tmp1 = flux_ext[1] * inv_jac_det[1];
            values_face[1][0][q]               = jac_xy[1][0][0] * tmp0 + jac_xy[1][1][0] * tmp1;
            values_face[1][1][q]               = jac_xy[1][0][1] * tmp0 + jac_xy[1][1][1] * tmp1;
            values_face[1][2][q]               = h_z * inv_jac_det[1] * flux_ext[2];
          }

          for(unsigned int d = 0; d < dim; ++d)
          {
            val_real[0][d] = integrate_factor *
                             (Number(0.5) * (std::abs(normal_speed) - normal_speed) *
                                (val_real[0][d] - val_real[1][d]) +
                              (Number(1.0) - factor_convective) * normal_speed * val_real[0][d]);
          }

          // apply test functions
          const VectorizedArray<Number> tmp0 = val_real[0][0] * inv_jac_det[0];
          const VectorizedArray<Number> tmp1 = val_real[0][1] * inv_jac_det[0];
          values_face[0][0][q]               = jac_xy[0][0][0] * tmp0 + jac_xy[0][1][0] * tmp1;
          values_face[0][1][q]               = jac_xy[0][0][1] * tmp0 + jac_xy[0][1][1] * tmp1;
          values_face[0][2][q]               = h_z * inv_jac_det[0] * val_real[0][2];
        }
      }
    }
    else
    {
      std::array<unsigned int, n_lanes> shifted_data_indices, shifted_data_indices_gr;
      for(unsigned int v = 0; v < n_lanes; ++v)
      {
        shifted_data_indices[v]    = convective_mapping_info.mapping_data_index[cell][v] * 4;
        shifted_data_indices_gr[v] = convective_mapping_info.mapping_data_index[cell][v] * 6;
      }

      const Number normal_sign = (f % 2) ? 1 : -1;

      for(unsigned int q = 0; q < n_q_points_2d; ++q)
      {
        Tensor<2, 2, VectorizedArray<Number>> jac_xy;
        vectorized_load_and_transpose(4,
                                      &convective_mapping_info.jacobians_xy[q][0][0],
                                      shifted_data_indices.data(),
                                      &jac_xy[0][0]);
        const VectorizedArray<Number>               inv_jac_det = Number(1.0) / determinant(jac_xy);
        const Tensor<2, 2, VectorizedArray<Number>> inv_jac_xy  = transpose(invert(jac_xy));

        const VectorizedArray<Number> val[2][3] = {
          {values_face[0][0][q], values_face[0][1][q], values_face[0][2][q]},
          {values_ext[0][q], values_ext[1][q], values_ext[2][q]}};

        Tensor<1, dim, VectorizedArray<Number>> val_real[2];
        for(unsigned int s = 0; s < 2; ++s)
        {
          for(unsigned int d = 0; d < 2; ++d)
            val_real[s][d] =
              (h_z_inverse * inv_jac_det) * (jac_xy[d][0] * val[s][0] + jac_xy[d][1] * val[s][1]);

          // h_z cancels with factor h_z_inverse from Jacobian determinant
          val_real[s][2] = inv_jac_det * val[s][2];
        }

        const VectorizedArray<Number> integrate_factor =
          convective_mapping_info.quad_weights_xy[q] * determinant(jac_xy);

        if(include_convective_divergence)
        {
          Tensor<1, 3, Tensor<1, 2, VectorizedArray<Number>>> jac_grad_xy;
          vectorized_load_and_transpose(6,
                                        &convective_mapping_info.jacobian_grads[q][0][0],
                                        shifted_data_indices_gr.data(),
                                        &jac_grad_xy[0][0]);
          VectorizedArray<Number> jac_grad_rank1[2];
          VectorizedArray<Number> tmp[2];
          for(unsigned int d = 0; d < 2; ++d)
            tmp[d] =
              (inv_jac_xy[0][d] * jac_grad_xy[d][0] + inv_jac_xy[1][d] * jac_grad_xy[d][1] +
               inv_jac_xy[0][1 - d] * jac_grad_xy[2][0] + inv_jac_xy[1][1 - d] * jac_grad_xy[2][1]);
          for(unsigned int d = 0; d < 2; ++d)
            jac_grad_rank1[d] = (tmp[0] * inv_jac_xy[d][0] + tmp[1] * inv_jac_xy[d][1]);

          // only need last component of velocity because normal = [0, 0, +- 1]
          const unsigned int            qd         = q * dim;
          const VectorizedArray<Number> grad[2][3] = {
            {grads_face[0][2][qd], grads_face[0][2][qd + 1], grads_face[0][2][qd + 2]},
            {grads_ext[2][0][qd], grads_ext[2][1][qd], grads_ext[2][2][qd]}};
          VectorizedArray<Number> grad_real[2][3];
          for(unsigned int s = 0; s < 2; ++s)
          {
            for(unsigned int d = 0; d < 2; ++d)
            {
              const VectorizedArray<Number> tmp_z_grad2 =
                (inv_jac_xy[d][0] * grad[s][0] + inv_jac_xy[d][1] * grad[s][1]);

              grad_real[s][d] = (tmp_z_grad2 - jac_grad_rank1[d] * val[s][2]) * h_z;
            }
            grad_real[s][2] = grad[s][2];
          }
          const VectorizedArray<Number> factor         = h_z_inverse * inv_jac_det;
          VectorizedArray<Number>       avg_u_grad_u_z = grad_real[0][0] * val_real[0][0];
          for(unsigned int e = 1; e < dim; ++e)
            avg_u_grad_u_z += grad_real[0][e] * val_real[0][e];
          for(unsigned int e = 0; e < dim; ++e)
            avg_u_grad_u_z += grad_real[1][e] * val_real[1][e];
          const VectorizedArray<Number> flux_term =
            Number(0.5) * normal_sign * integrate_factor * factor * avg_u_grad_u_z;

          // if (q == 1)
          // std::cout << "my  " << f << " " << flux_term << std::endl;

          my_face_data[dim * n_q_points_face + q] = flux_term;
          if(fill_buffer)
            pressure_flux_ext[q] = -flux_term;
        }

        VectorizedArray<Number> normal_speed = normal_sign * val_real[0][2];

        // test functions of the neighbor, which has the same Jacobian in
        // the extruded direction
        if(fill_buffer)
        {
          VectorizedArray<Number> flux_ext[dim];
          for(unsigned int d = 0; d < dim; ++d)
            flux_ext[d] = integrate_factor *
                          (Number(0.5) * (std::abs(normal_speed) + normal_speed) *
                             (val_real[1][d] - val_real[0][d]) -
                           (Number(1.0) - factor_convective) * normal_speed * val_real[1][d]);
          for(unsigned int d = 0; d < 2; ++d)
            values_face[1][d][q] = (h_z_inverse * inv_jac_det) *
                                   (jac_xy[0][d] * flux_ext[0] + jac_xy[1][d] * flux_ext[1]);
          values_face[1][2][q] = inv_jac_det * flux_ext[2];
        }

        for(unsigned int d = 0; d < dim; ++d)
        {
          val_real[0][d] =
            integrate_factor * (Number(0.5) * (std::abs(normal_speed) - normal_speed) *
                                  (val_real[0][d] - val_real[1][d]) +
                                (Number(1.0) - factor_convective) * normal_speed * val_real[0][d]);
        }

        // transpose of evaluate part
        for(unsigned int d = 0; d < 2; ++d)
          values_face[0][d][q] = (h_z_inverse * inv_jac_det) *
                                 (jac_xy[0][d] * val_real[0][0] + jac_xy[1][d] * val_real[0][1]);
        values_face[0][2][q] = inv_jac_det * val_real[0][2];
      }
    }

    // fill flux buffer with the contributions of the neighbor, in the layout
    // velocity components followed by the convective divergence
    const unsigned int * buffer_index = convective_face_flux_buffer_index[cell][f].data();
    if(fill_buffer)
    {
      for(unsigned int d = 0; d < dim; ++d)
        vectorized_transpose_and_store(false,
                                       n_q_points_face,
                                       values_face[1][d],
                                       buffer_index,
                                       convective_face_flux_buffer.data() + d * n_q_points_face);
      if(include_convective_divergence)
        vectorized_transpose_and_store(false,
                                       n_q_points_face,
                                       pressure_flux_ext,
                                       buffer_index,
                                       convective_face_flux_buffer.data() + dim * n_q_points_face);
    }

    for(unsigned int d = 0; d < dim; ++d)
      for(unsigned int q = 0; q < n_q_points_face; ++q)
        my_face_data[d * n_q_points_face + q] = values_face[0][d][q];

    // get the contributions of the left face computed by the neighbor
    if(use_convective_face_flux_buffer && f % 2 == 1 &&
       all_left_face_fluxes_from_buffer[cell][face_direction])
    {
      buffer_index = convective_face_flux_buffer_index[cell][f - 1].data();
      for(unsigned int d = 0; d < (include_convective_divergence ? dim + 1 : dim); ++d)
        vectorized_load_and_transpose(n_q_points_face,
                                      convective_face_flux_buffer.data() + d * n_q_points_face,
                                      buffer_index,
                                      face_data + (f - 1) * (dim + 1) * n_q_points_face +
                                        d * n_q_points_face);
    }
  }

  template<int degree>
  void
  do_momentum_rhs_on_cell(const unsigned int                             cell,
                          const VectorType &                             pressure,
                          const VectorType &                             velocity_old,
                          const Number                                   velocity_scale,
                          const std::shared_ptr<dealii::Function<dim>> & rhs_function,
                          VectorType &                                   rhs) const
  {
    constexpr unsigned int n_q_points_1d   = degree + 1;
    constexpr unsigned int n_points        = Utilities::pow(n_q_points_1d, dim);
    const auto &           shape_data      = shape_info.data;
    const auto &           shape_data_pres = pressure_shape_info.data[0].shape_values_eo;

    VectorizedArray<Number> quad_values[dim * n_points], pressure_values[n_points];
    read_cell_values<degree, n_q_points_1d>(velocity_old.begin(), dof_indices[cell], quad_values);

    const std::array<unsigned int, n_lanes> pressure_indices      = (*pressure_dof_indices)[cell];
    bool                                    all_indices_available = true;
    for(unsigned int v = 0; v < n_lanes; ++v)
      if(pressure_indices[v] == numbers::invalid_unsigned_int)
      {
        all_indices_available = false;
        break;
      }
    if(all_indices_available)
      vectorized_load_and_transpose(Utilities::pow(degree, dim),
                                    pressure.begin(),
                                    pressure_indices.data(),
                                    pressure_values);
    else
    {
      // first broadcast valid entries to all lanes, then fill the remaining
      // ones
      for(unsigned int i = 0; i < Utilities::pow(degree, dim); ++i)
        pressure_values[i] = pressure.local_element(pressure_indices[0] + i);
      for(unsigned int v = 1; v < n_lanes && pressure_indices[v] != numbers::invalid_unsigned_int;
          ++v)
        for(unsigned int i = 0; i < Utilities::pow(degree, dim); ++i)
          pressure_values[i][v] = pressure.local_element(pressure_indices[v] + i);
    }

    internal::FEEvaluationImplBasisChange<dealii::internal::evaluate_evenodd,
                                          dealii::internal::EvaluatorQuantity::value,
                                          dim,
                                          degree,
                                          n_q_points_1d>::do_forward(1,
                                                                     shape_data_pres,
                                                                     pressure_values,
                                                                     pressure_values);

    constexpr unsigned int n_q_points_2d = n_q_points_1d * n_q_points_1d;

    std::array<unsigned int, n_lanes> shifted_data_indices;
    for(unsigned int v = 0; v < n_lanes; ++v)
      shifted_data_indices[v] = mapping_info.mapping_data_index[cell][v] * 4;

    const Number h_z_inverse = mapping_info.h_z_inverse;
    const Number h_z         = mapping_info.h_z;

    constexpr unsigned int nn = n_q_points_1d;

    for(unsigned int q1 = 0; q1 < nn * nn; ++q1)
    {
      Tensor<2, 2, VectorizedArray<Number>> jac_xy;
      vectorized_load_and_transpose(4,
                                    &mapping_info.jacobians_xy[q1][0][0],
                                    shifted_data_indices.data(),
                                    &jac_xy[0][0]);
      const VectorizedArray<Number> inv_jac_det = Number(1.0) / determinant(jac_xy);

      // Assume body force does not depend on z direction
      Tensor<1, dim, VectorizedArray<Number>> body_force;
      if(rhs_function.get() != nullptr)
        for(unsigned int v = 0; v < n_lanes; ++v)
        {
          Point<dim> point;
          for(unsigned int d = 0; d < 2; ++d)
            point[d] =
              mapping_info.quadrature_points[mapping_info.mapping_data_index[cell][v] + q1][d];
          for(unsigned int d = 0; d < dim; ++d)
            body_force[d][v] = rhs_function->value(point, d);
        }

      if constexpr(dim == 2)
      {
        const VectorizedArray<Number> val[2] = {quad_values[q1 * dim], quad_values[q1 * dim + 1]};
        VectorizedArray<Number>       val_real[2] = {jac_xy[0][0] * val[0] + jac_xy[0][1] * val[1],
                                               jac_xy[1][0] * val[0] + jac_xy[1][1] * val[1]};
        for(unsigned int d = 0; d < dim; ++d)
          val_real[d] = val_real[d] * (inv_jac_det * velocity_scale) + body_force[d];

        const VectorizedArray<Number> s0 = jac_xy[0][0] * val_real[0] + jac_xy[1][0] * val_real[1];
        const VectorizedArray<Number> s1 = jac_xy[0][1] * val_real[0] + jac_xy[1][1] * val_real[1];

        quad_values[q1 * dim]     = s0 * (mapping_info.quad_weights_xy[q1]);
        quad_values[q1 * dim + 1] = s1 * (mapping_info.quad_weights_xy[q1]);

        pressure_values[q1] *= mapping_info.quad_weights_xy[q1];
      }
      else // now to dim == 3
      {
        for(unsigned int qz = 0, q = q1; qz < nn; ++qz, q += n_q_points_2d)
        {
          const VectorizedArray<Number> val[3] = {quad_values[q * dim],
                                                  quad_values[q * dim + 1],
                                                  quad_values[q * dim + 2]};

          Tensor<1, dim, VectorizedArray<Number>> val_real;
          val_real[0] = jac_xy[0][0] * val[0] + jac_xy[0][1] * val[1];
          val_real[1] = jac_xy[1][0] * val[0] + jac_xy[1][1] * val[1];
          val_real[2] = h_z * val[2];
          for(unsigned int d = 0; d < dim; ++d)
            val_real[d] =
              val_real[d] * (h_z_inverse * velocity_scale * inv_jac_det) + body_force[d];

          // For the test function part, the Jacobian determinant
          // that is part of the integration factor cancels with the
          // inverse Jacobian determinant present in the factor of
          // the derivative
          for(unsigned int d = 0; d < dim; ++d)
            val_real[d] *= mapping_info.quad_weights_xy[q1] * mapping_info.quad_weights_z[qz];

          VectorizedArray<Number> value_tmp[dim];
          for(unsigned int d = 0; d < 2; ++d)
            value_tmp[d] = jac_xy[0][d] * val_real[0] + jac_xy[1][d] * val_real[1];
          value_tmp[2] = h_z * val_real[2];

          quad_values[q * dim]     = value_tmp[0];
          quad_values[q * dim + 1] = value_tmp[1];
          quad_values[q * dim + 2] = value_tmp[2];

          pressure_values[q] *=
            (mapping_info.quad_weights_xy[q1] * mapping_info.quad_weights_z[qz]);
        }
      }
    }

    const Number * DEAL_II_RESTRICT shape_grad =
      shape_data[0].shape_gradients_collocation_eo.data();
    // add (div v, p)_K
    if(dim == 3)
      for(unsigned int i = 0; i < nn * nn; ++i)
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::gradient,
                                              nn,
                                              nn,
                                              nn * nn,
                                              nn * nn * dim,
                                              false,
                                              true>(shape_grad,
                                                    pressure_values + i,
                                                    quad_values + i * dim + 2);
    for(unsigned int i1 = 0; i1 < (dim == 3 ? nn : 1); ++i1)
    {
      for(unsigned int i0 = 0; i0 < nn; ++i0)
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::gradient,
                                              nn,
                                              nn,
                                              nn,
                                              nn * dim,
                                              false,
                                              true>(shape_grad,
                                                    pressure_values + i1 * nn * nn + i0,
                                                    quad_values + (i1 * nn * nn + i0) * dim + 1);
      for(unsigned int i0 = 0; i0 < nn; ++i0)
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::gradient,
                                              nn,
                                              nn,
                                              1,
                                              dim,
                                              false,
                                              true>(shape_grad,
                                                    pressure_values + i1 * nn * nn + i0 * nn,
                                                    quad_values + (i1 * nn + i0) * nn * dim + 0);
    }

    // (v \cdot normal, p)_{dK cap \Gamma_D} does not contribute, because the
    // normal component is subject to (homogeneous) Dirichlet boundary
    // conditions for H(div) spaces
    integrate_cell_scatter<degree, n_q_points_1d>(dof_indices[cell],
                                                  n_active_entries_per_cell_batch(cell),
                                                  quad_values,
                                                  rhs.begin());
  }

  template<int degree>
  void
  do_velocity_divergence_on_cell(const unsigned int cell,
                                 const VectorType & velocity,
                                 const Number       velocity_scale,
                                 VectorType &       pressure_rhs) const
  {
    constexpr unsigned int n_q_points_1d   = degree + 1;
    constexpr unsigned int nn              = n_q_points_1d;
    constexpr unsigned int n_points        = Utilities::pow(n_q_points_1d, dim);
    const auto &           shape_data      = shape_info.data;
    const auto &           shape_data_pres = pressure_shape_info.data[0].shape_values_eo;

    VectorizedArray<Number> quad_values[dim * n_points], pressure_values[n_points];
    read_cell_values<degree, n_q_points_1d>(velocity.begin(), dof_indices[cell], quad_values);

    const Number * DEAL_II_RESTRICT shape_grad =
      shape_data[0].shape_gradients_collocation_eo.data();

    for(unsigned int i1 = 0; i1 < (dim == 3 ? nn : 1); ++i1)
    {
      for(unsigned int i0 = 0; i0 < nn; ++i0)
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::gradient,
                                              nn,
                                              nn,
                                              dim,
                                              1,
                                              true,
                                              false>(shape_grad,
                                                     quad_values + (i1 * nn + i0) * nn * dim + 0,
                                                     pressure_values + i1 * nn * nn + i0 * nn);
      for(unsigned int i0 = 0; i0 < nn; ++i0)
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::gradient,
                                              nn,
                                              nn,
                                              nn * dim,
                                              nn,
                                              true,
                                              true>(shape_grad,
                                                    quad_values + (i1 * nn * nn + i0) * dim + 1,
                                                    pressure_values + i1 * nn * nn + i0);
    }
    if(dim == 3)
      for(unsigned int i = 0; i < nn * nn; ++i)
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::gradient,
                                              nn,
                                              nn,
                                              nn * nn * dim,
                                              nn * nn,
                                              true,
                                              true>(shape_grad,
                                                    quad_values + i * dim + 2,
                                                    pressure_values + i);

    for(unsigned int q1 = 0; q1 < nn * nn; ++q1)
    {
      if constexpr(dim == 2)
      {
        pressure_values[q1] *= velocity_scale * mapping_info.quad_weights_xy[q1];
      }
      else // now to dim == 3
      {
        for(unsigned int qz = 0, q = q1; qz < nn; ++qz, q += nn * nn)
        {
          const Number weight_z = mapping_info.quad_weights_z[qz];
          pressure_values[q] *= velocity_scale * mapping_info.quad_weights_xy[q1] * weight_z;
        }
      }
    }

    internal::FEEvaluationImplBasisChange<dealii::internal::evaluate_evenodd,
                                          dealii::internal::EvaluatorQuantity::value,
                                          dim,
                                          degree,
                                          n_q_points_1d>::do_backward(1,
                                                                      shape_data_pres,
                                                                      false,
                                                                      pressure_values,
                                                                      pressure_values);

    const std::array<unsigned int, n_lanes> pressure_indices      = (*pressure_dof_indices)[cell];
    bool                                    all_indices_available = true;
    for(unsigned int v = 0; v < n_lanes; ++v)
      if(pressure_indices[v] == numbers::invalid_unsigned_int)
      {
        all_indices_available = false;
        break;
      }
    if(all_indices_available)
      vectorized_transpose_and_store(true,
                                     Utilities::pow(degree, dim),
                                     pressure_values,
                                     pressure_indices.data(),
                                     pressure_rhs.begin());
    else
    {
      for(unsigned int v = 0; v < n_lanes && pressure_indices[v] != numbers::invalid_unsigned_int;
          ++v)
        for(unsigned int i = 0; i < Utilities::pow(degree, dim); ++i)
          pressure_rhs.local_element(pressure_indices[v] + i) += pressure_values[i][v];
    }
  }

  template<int degree>
  void
  do_body_force_on_cell(const unsigned int            cell,
                        const dealii::Function<dim> & rhs_function,
                        VectorType &                  dst) const
  {
    constexpr unsigned int n_q_points_1d = degree + 1;
    constexpr unsigned int n_points      = Utilities::pow(n_q_points_1d, dim);

    VectorizedArray<Number> quad_values[dim * n_points];

    std::array<unsigned int, n_lanes> shifted_data_indices;
    for(unsigned int v = 0; v < n_lanes; ++v)
      shifted_data_indices[v] = mapping_info.mapping_data_index[cell][v] * 4;

    const Number h_z = mapping_info.h_z;

    constexpr unsigned int nn = n_q_points_1d;

    for(unsigned int qy = 0, q1 = 0; qy < nn; ++qy)
    {
      for(unsigned int qx = 0; qx < nn; ++qx, ++q1)
      {
        Tensor<2, 2, VectorizedArray<Number>> jac_xy;
        vectorized_load_and_transpose(4,
                                      &mapping_info.jacobians_xy[q1][0][0],
                                      shifted_data_indices.data(),
                                      &jac_xy[0][0]);

        // Assume body force does not depend on z direction
        Tensor<1, dim, VectorizedArray<Number>> body_force;
        for(unsigned int v = 0; v < n_lanes; ++v)
        {
          Point<dim> point;
          for(unsigned int d = 0; d < 2; ++d)
            point[d] =
              mapping_info.quadrature_points[mapping_info.mapping_data_index[cell][v] + q1][d];
          for(unsigned int d = 0; d < dim; ++d)
            body_force[d][v] = rhs_function.value(point, d);
        }

        if constexpr(dim == 2)
        {
          const VectorizedArray<Number> s0 =
            jac_xy[0][0] * body_force[0] + jac_xy[1][0] * body_force[1];
          const VectorizedArray<Number> s1 =
            jac_xy[0][1] * body_force[0] + jac_xy[1][1] * body_force[1];

          quad_values[q1 * dim]     = s0 * (mapping_info.quad_weights_xy[q1]);
          quad_values[q1 * dim + 1] = s1 * (mapping_info.quad_weights_xy[q1]);
        }
        else // now to dim == 3
        {
          for(unsigned int qz = 0, q = q1; qz < nn; ++qz, q += nn * nn)
          {
            const Number weight_z = mapping_info.quad_weights_z[qz];
            // need factors without h_z_inverse in code below because
            // it cancels with h_z
            const VectorizedArray<Number> factor = mapping_info.quad_weights_xy[q1] * weight_z;

            // For the test function part, the Jacobian determinant
            // that is part of the integration factor cancels with the
            // inverse Jacobian determinant present in the factor of
            // the derivative

            // jac_grad * (J^{-1} * (grad_in * factor)), re-use part in
            // braces as 'tmp' from above
            VectorizedArray<Number>                       value_tmp[dim];
            const Tensor<1, dim, VectorizedArray<Number>> val_real = body_force * factor;
            for(unsigned int d = 0; d < 2; ++d)
              value_tmp[d] = jac_xy[0][d] * val_real[0] + jac_xy[1][d] * val_real[1];
            value_tmp[2] = h_z * val_real[2];

            quad_values[q * dim]     = value_tmp[0];
            quad_values[q * dim + 1] = value_tmp[1];
            quad_values[q * dim + 2] = value_tmp[2];
          }
        }
      }
    }

    integrate_cell_scatter<degree, n_q_points_1d>(dof_indices[cell],
                                                  n_active_entries_per_cell_batch(cell),
                                                  quad_values,
                                                  dst.begin());
  }

  template<int degree>
  void
  do_pressure_neumann_on_cell(const unsigned int cell,
                              const VectorType & velocity,
                              const bool         do_convective,
                              const Number       viscosity,
                              VectorType &       pressure_rhs) const
  {
    const unsigned int face           = cells_at_dirichlet_boundary[cell].first;
    const unsigned int face_direction = face / 2;

    dealii::ndarray<unsigned int, 2 * dim + 1, n_lanes> my_dof_indices;
    unsigned int                                        n_lanes_filled = 0;
    for(unsigned int v = 0; v < n_lanes; ++v)
    {
      const unsigned int cell_index = cells_at_dirichlet_boundary[cell].second[v];

      // use a dummy neighbor with nearby data for cells at boundary to make
      // sure vectorized functions continue working
      if(cell_index != numbers::invalid_unsigned_int)
      {
        for(unsigned int i = 0; i < 2 * dim + 1; ++i)
          my_dof_indices[i][v] = this->dof_indices[cell_index / n_lanes][i][cell_index % n_lanes];
        ++n_lanes_filled;
      }
      else
      {
        Assert(v > 0, ExcInternalError());
        for(unsigned int i = 0; i < 2 * dim + 1; ++i)
          my_dof_indices[i][v] = my_dof_indices[i][n_lanes_filled - 1];
      }
    }

    constexpr unsigned int nn              = degree + 1;
    constexpr unsigned int n_points        = Utilities::pow(nn, dim);
    const auto &           shape_data      = shape_info.data;
    const auto &           shape_data_pres = pressure_shape_info.data[0].shape_values_eo;

    VectorizedArray<Number> quad_values[2][dim * n_points + 1], grad_x[dim * nn * nn],
      grad_z[dim * nn], pressure_values[Utilities::pow(nn, dim - 1)];
    read_cell_values<degree, nn>(velocity.begin(), my_dof_indices, quad_values[0]);

    // Compute gradient at quadrature points
    std::array<unsigned int, n_lanes> shifted_data_indices;
    std::array<unsigned int, n_lanes> shifted_data_indices_gr;
    for(unsigned int v = 0; v < n_lanes; ++v)
    {
      const unsigned int cell_index =
        cells_at_dirichlet_boundary[cell].second[v] == numbers::invalid_unsigned_int ?
          cells_at_dirichlet_boundary[cell].second[n_lanes_filled - 1] :
          cells_at_dirichlet_boundary[cell].second[v];
      shifted_data_indices[v] =
        mapping_info.mapping_data_index[cell_index / n_lanes][cell_index % n_lanes] * 4;
      shifted_data_indices_gr[v] =
        mapping_info.mapping_data_index[cell_index / n_lanes][cell_index % n_lanes] * 6;
    }

    const Number h_z         = mapping_info.h_z;
    const Number h_z_inverse = mapping_info.h_z_inverse;

    const Number * shape_grad = shape_data[0].shape_gradients_collocation_eo.data();
    for(unsigned int i1 = 0; i1 < (dim == 3 ? nn : 1); ++i1)
      for(unsigned int i0 = 0; i0 < nn; ++i0)
      {
        const unsigned int idx = dim * (i1 * nn * nn + i0);
        for(unsigned int d = 0; d < dim; ++d)
          internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                internal::EvaluatorQuantity::gradient,
                                                nn,
                                                nn,
                                                nn * dim,
                                                nn * dim,
                                                true,
                                                false>(shape_grad,
                                                       quad_values[0] + idx + d,
                                                       quad_values[1] + idx + d);
      }

    for(unsigned int qy = 0, q1 = 0; qy < nn; ++qy)
    {
      for(unsigned int i0 = 0; i0 < (dim == 3 ? nn : 1); ++i0)
        for(unsigned int d = 0; d < dim; ++d)
          internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                internal::EvaluatorQuantity::gradient,
                                                nn,
                                                nn,
                                                dim,
                                                (dim == 3 ? nn : 1) * dim,
                                                true,
                                                false>(
            shape_grad, quad_values[0] + dim * (qy * nn + i0 * nn * nn) + d, grad_x + dim * i0 + d);

      for(unsigned int qx = 0; qx < nn; ++qx, ++q1)
      {
        Tensor<2, 2, VectorizedArray<Number>> jac_xy;
        vectorized_load_and_transpose(4,
                                      &mapping_info.jacobians_xy[q1][0][0],
                                      shifted_data_indices.data(),
                                      &jac_xy[0][0]);
        const VectorizedArray<Number>               inv_jac_det = Number(1.0) / determinant(jac_xy);
        const Tensor<2, 2, VectorizedArray<Number>> inv_jac_xy  = transpose(invert(jac_xy));
        Tensor<1, 3, Tensor<1, 2, VectorizedArray<Number>>> jac_grad_xy;
        vectorized_load_and_transpose(6,
                                      &mapping_info.jacobian_grads[q1][0][0],
                                      shifted_data_indices_gr.data(),
                                      &jac_grad_xy[0][0]);

        // Prepare for -(J^{-T} * jac_grad * J^{-1} * J * values *
        // det(J^{-1})), which can be expressed as a rank-1 update tmp[d] *
        // tmp4[e], where tmp = J * values and tmp4 = (J^{-T} * jac_grad *
        // J^{-1})
        VectorizedArray<Number> jac_grad_rank1[2];
        {
          VectorizedArray<Number> tmp[2];
          for(unsigned int d = 0; d < 2; ++d)
            tmp[d] =
              (inv_jac_xy[0][d] * jac_grad_xy[d][0] + inv_jac_xy[1][d] * jac_grad_xy[d][1] +
               inv_jac_xy[0][1 - d] * jac_grad_xy[2][0] + inv_jac_xy[1][1 - d] * jac_grad_xy[2][1]);
          for(unsigned int d = 0; d < 2; ++d)
            jac_grad_rank1[d] = (tmp[0] * inv_jac_xy[d][0] + tmp[1] * inv_jac_xy[d][1]);
        }

        if constexpr(dim == 2)
        {
          const VectorizedArray<Number> val[2]     = {quad_values[0][q1 * dim],
                                                  quad_values[0][q1 * dim + 1]};
          const VectorizedArray<Number> grad[2][2] = {{grad_x[qx * dim], grad_x[qx * dim + 1]},
                                                      {quad_values[1][q1 * dim],
                                                       quad_values[1][q1 * dim + 1]}};

          VectorizedArray<Number> grad_real[dim][dim];
          for(unsigned int d = 0; d < dim; ++d)
          {
            // (J * grad_quad) * J^-1 * det(J^-1), part in braces
            VectorizedArray<Number> tmp[dim];
            for(unsigned int e = 0; e < dim; ++e)
              tmp[e] = (jac_xy[d][0] * grad[0][e] + jac_xy[d][1] * grad[1][e]);

            // Add (jac_grad * values) * J^{-1} * det(J^{-1}), combine
            // terms outside braces with gradient part from above
            tmp[0] += jac_grad_xy[0][d] * val[0];
            tmp[0] += jac_grad_xy[2][d] * val[1];
            tmp[1] += jac_grad_xy[1][d] * val[1];
            tmp[1] += jac_grad_xy[2][d] * val[0];

            for(unsigned int e = 0; e < dim; ++e)
              grad_real[d][e] = (tmp[0] * inv_jac_xy[e][0] + tmp[1] * inv_jac_xy[e][1]);

            VectorizedArray<Number> tmp2 = jac_xy[d][0] * val[0] + jac_xy[d][1] * val[1];

            for(unsigned int e = 0; e < dim; ++e)
              grad_real[d][e] -= jac_grad_rank1[e] * tmp2;
          }
          quad_values[1][q1 * dim]     = inv_jac_det * (grad_real[1][0] - grad_real[0][1]);
          quad_values[1][q1 * dim + 1] = 0.;
        }
        else // now to dim == 3
        {
          for(unsigned int d = 0; d < dim; ++d)
            internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                  internal::EvaluatorQuantity::gradient,
                                                  nn,
                                                  nn,
                                                  nn * nn * dim,
                                                  dim,
                                                  true,
                                                  false>(shape_grad,
                                                         quad_values[0] + q1 * dim + d,
                                                         grad_z + d);

          for(unsigned int qz = 0, q = q1; qz < nn; ++qz, q += nn * nn)
          {
            const unsigned int            ix         = dim * (qz + qx * nn);
            const unsigned int            iz         = dim * qz;
            const VectorizedArray<Number> val[3]     = {quad_values[0][q * dim],
                                                    quad_values[0][q * dim + 1],
                                                    quad_values[0][q * dim + 2]};
            const VectorizedArray<Number> grad[3][3] = {
              {grad_x[ix + 0], quad_values[1][q * dim + 0], grad_z[iz + 0]},
              {grad_x[ix + 1], quad_values[1][q * dim + 1], grad_z[iz + 1]},
              {grad_x[ix + 2], quad_values[1][q * dim + 2], grad_z[iz + 2]}};

            VectorizedArray<Number> val_real[2];
            val_real[0] = jac_xy[0][0] * val[0] + jac_xy[0][1] * val[1];
            val_real[1] = jac_xy[1][0] * val[0] + jac_xy[1][1] * val[1];

            VectorizedArray<Number> grad_real[dim][dim];
            for(unsigned int d = 0; d < 2; ++d)
            {
              // (J * grad_quad) * J^-1 * det(J^-1), part in braces
              VectorizedArray<Number> tmp[dim];
              for(unsigned int e = 0; e < dim; ++e)
                tmp[e] = (jac_xy[d][0] * grad[0][e] + jac_xy[d][1] * grad[1][e]);

              // Add (jac_grad * values) * J^{-1} * det(J^{-1}),
              // combine terms outside braces with gradient part from
              // above
              tmp[0] += jac_grad_xy[0][d] * val[0];
              tmp[0] += jac_grad_xy[2][d] * val[1];
              tmp[1] += jac_grad_xy[1][d] * val[1];
              tmp[1] += jac_grad_xy[2][d] * val[0];

              for(unsigned int e = 0; e < 2; ++e)
                grad_real[d][e] = (tmp[0] * inv_jac_xy[e][0] + tmp[1] * inv_jac_xy[e][1]);

              for(unsigned int e = 0; e < 2; ++e)
                grad_real[d][e] -= jac_grad_rank1[e] * val_real[d];

              const VectorizedArray<Number> tmp_z_grad2 =
                (inv_jac_xy[d][0] * grad[2][0] + inv_jac_xy[d][1] * grad[2][1]);

              grad_real[2][d] = (tmp_z_grad2 - jac_grad_rank1[d] * val[2]) * h_z;
              grad_real[d][2] = h_z_inverse * tmp[2];
            }
            grad_real[2][2] = grad[2][2];

            const VectorizedArray<Number> J_det = h_z_inverse * inv_jac_det;
            quad_values[1][q * dim + 0]         = J_det * (grad_real[2][1] - grad_real[1][2]);
            quad_values[1][q * dim + 1]         = J_det * (grad_real[0][2] - grad_real[2][0]);
            quad_values[1][q * dim + 2]         = J_det * (grad_real[1][0] - grad_real[0][1]);
          }
        }
      }
    }

    // interpolate values, gradients and values/gradients of vorticity to
    // face; the vorticity is already set up in the collocation space, so no
    // need for a Piola transformation for those terms, whereas the gradient
    // for the convective term will be done here
    const unsigned int      n_q_points_face = Utilities::pow(nn, dim - 1);
    VectorizedArray<Number> values_face[2][dim][n_q_points_face + 1];
    VectorizedArray<Number> grads_face[2][dim][n_q_points_face * dim + 1];

    // interpolate from interior to face
    const std::array<Number, 2> * shape = interpolate_quad_to_boundary[face % 2].data();

    for(unsigned int s = (do_convective ? 0 : 1); s < 2; ++s)
      for(unsigned int i1 = 0; i1 < nn; ++i1)
        for(unsigned int i0 = 0; i0 < nn; ++i0)
        {
          unsigned int idx = 0;
          if(face_direction == 0)
            idx = nn * (nn * i1 + i0);
          else if(face_direction == 1)
            idx = i0 + i1 * nn * nn;
          else
            idx = i0 + i1 * nn;

          const VectorizedArray<Number> * my_vals = quad_values[s] + idx * dim;
          VectorizedArray<Number>         v0      = shape[0][0] * my_vals[0];
          VectorizedArray<Number>         d0      = shape[0][1] * my_vals[0];
          VectorizedArray<Number>         v1      = shape[0][0] * my_vals[1];
          VectorizedArray<Number>         d1      = shape[0][1] * my_vals[1];
          VectorizedArray<Number>         v2      = shape[0][0] * my_vals[2];
          VectorizedArray<Number>         d2      = shape[0][1] * my_vals[2];
          const unsigned int              stride  = Utilities::pow(nn, face_direction) * dim;
          for(unsigned int i = 1; i < nn; ++i)
          {
            v0 += shape[i][0] * my_vals[i * stride + 0];
            d0 += shape[i][1] * my_vals[i * stride + 0];
            v1 += shape[i][0] * my_vals[i * stride + 1];
            d1 += shape[i][1] * my_vals[i * stride + 1];
            v2 += shape[i][0] * my_vals[i * stride + 2];
            d2 += shape[i][1] * my_vals[i * stride + 2];
          }
          const unsigned int val_idx   = i1 * nn + i0;
          values_face[s][0][val_idx]   = v0;
          values_face[s][1][val_idx]   = v1;
          values_face[s][2][val_idx]   = v2;
          const unsigned int deriv_idx = val_idx * dim + face_direction;
          grads_face[s][0][deriv_idx]  = d0;
          grads_face[s][1][deriv_idx]  = d1;
          grads_face[s][2][deriv_idx]  = d2;
        }

    // Only need some tangential derivative in xy plane, as the domain is
    // extruded in z direction
    internal::EvaluatorTensorProduct<internal::evaluate_evenodd,
                                     dim - 1,
                                     nn,
                                     nn,
                                     VectorizedArray<Number>,
                                     Number>
      eval_g({}, shape_data[0].shape_gradients_collocation_eo.data(), {});
    for(unsigned int s = (do_convective ? 0 : 1); s < 2; ++s)
    {
      for(unsigned int d = 0; d < dim; ++d)
      {
        const unsigned int first  = face_direction == 0 ? 1 : 0;
        const unsigned int second = face_direction == 1 ? 2 : first + 1;
        eval_g.template gradients<0, true, false, dim>(values_face[s][d], grads_face[s][d] + first);
        eval_g.template gradients<1, true, false, dim>(values_face[s][d],
                                                       grads_face[s][d] + second);
      }
    }

    if(face_direction < 2)
    {
      if(do_convective)
      {
        std::array<unsigned int, n_lanes> shifted_data_indices;
        std::array<unsigned int, n_lanes> shifted_data_indices_gr;
        std::array<unsigned int, n_lanes> shifted_data_indices_norm;
        for(unsigned int v = 0; v < n_lanes; ++v)
        {
          const unsigned int cell_index =
            cells_at_dirichlet_boundary[cell].second[v] == numbers::invalid_unsigned_int ?
              cells_at_dirichlet_boundary[cell].second[n_lanes_filled - 1] :
              cells_at_dirichlet_boundary[cell].second[v];

          const unsigned int idx =
            mapping_info
              .face_mapping_data_index[cell_index / n_lanes][face][0][cell_index % n_lanes];
          shifted_data_indices[v]      = idx * 4;
          shifted_data_indices_gr[v]   = idx * 6;
          shifted_data_indices_norm[v] = idx * 2;
        }

        for(unsigned int q1 = 0; q1 < nn; ++q1)
        {
          Tensor<2, 2, VectorizedArray<Number>> jac_xy;
          vectorized_load_and_transpose(4,
                                        &mapping_info.face_jacobians_xy[q1][0][0],
                                        shifted_data_indices.data(),
                                        &jac_xy[0][0]);

          const VectorizedArray<Number> area_element_xy =
            std::sqrt(jac_xy[0][1 - face_direction] * jac_xy[0][1 - face_direction] +
                      jac_xy[1][1 - face_direction] * jac_xy[1][1 - face_direction]);

          const VectorizedArray<Number> inv_jac_det = Number(1.0) / determinant(jac_xy);
          const Tensor<2, 2, VectorizedArray<Number>> inv_jac_xy = transpose(invert(jac_xy));

          Tensor<1, 3, Tensor<1, 2, VectorizedArray<Number>>> jac_grad_xy;
          vectorized_load_and_transpose(6,
                                        &mapping_info.face_jacobian_grads[q1][0][0],
                                        shifted_data_indices_gr.data(),
                                        &jac_grad_xy[0][0]);

          Tensor<1, dim, VectorizedArray<Number>> normal;
          vectorized_load_and_transpose(2,
                                        &mapping_info.face_normal_vector_xy[q1][0],
                                        shifted_data_indices_norm.data(),
                                        &normal[0]);

          // Prepare for -(J^{-T} * jac_grad * J^{-1} * J * values *
          // det(J^{-1})), which can be expressed as a rank-1 update
          // tmp[d] * tmp4[e], where tmp = J * values and tmp4 =
          // (J^{-T} * jac_grad * J^{-1})
          VectorizedArray<Number> jac_grad_rank1[2];
          {
            VectorizedArray<Number> tmp[2];
            for(unsigned int d = 0; d < 2; ++d)
              tmp[d] =
                (inv_jac_xy[0][d] * jac_grad_xy[d][0] + inv_jac_xy[1][d] * jac_grad_xy[d][1] +
                 inv_jac_xy[0][1 - d] * jac_grad_xy[2][0] +
                 inv_jac_xy[1][1 - d] * jac_grad_xy[2][1]);
            for(unsigned int d = 0; d < 2; ++d)
              jac_grad_rank1[d] = (tmp[0] * inv_jac_xy[d][0] + tmp[1] * inv_jac_xy[d][1]);
          }

          for(unsigned int qz = 0, q = q1; qz < nn; ++qz, q += nn)
          {
            const VectorizedArray<Number> val[3]     = {values_face[0][0][q],
                                                    values_face[0][1][q],
                                                    values_face[0][2][q]};
            const VectorizedArray<Number> grad[3][3] = {{grads_face[0][0][q * dim],
                                                         grads_face[0][0][q * dim + 1],
                                                         grads_face[0][0][q * dim + 2]},
                                                        {grads_face[0][1][q * dim],
                                                         grads_face[0][1][q * dim + 1],
                                                         grads_face[0][1][q * dim + 2]},
                                                        {grads_face[0][2][q * dim],
                                                         grads_face[0][2][q * dim + 1],
                                                         grads_face[0][2][q * dim + 2]}};

            Tensor<1, dim, VectorizedArray<Number>> val_real;
            val_real[0] = jac_xy[0][0] * val[0] + jac_xy[0][1] * val[1];
            val_real[1] = jac_xy[1][0] * val[0] + jac_xy[1][1] * val[1];
            val_real[2] = h_z * val[2];

            VectorizedArray<Number> grad_real[dim][dim];
            for(unsigned int d = 0; d < 2; ++d)
            {
              // (J * grad_quad) * J^-1 * det(J^-1), part in braces
              VectorizedArray<Number> tmp[dim];
              for(unsigned int e = 0; e < dim; ++e)
                tmp[e] = (jac_xy[d][0] * grad[0][e] + jac_xy[d][1] * grad[1][e]);

              // Add (jac_grad * values) * J^{-1} * det(J^{-1}),
              // combine terms outside braces with gradient part from
              // above
              tmp[0] += jac_grad_xy[0][d] * val[0];
              tmp[0] += jac_grad_xy[2][d] * val[1];
              tmp[1] += jac_grad_xy[1][d] * val[1];
              tmp[1] += jac_grad_xy[2][d] * val[0];

              for(unsigned int e = 0; e < 2; ++e)
                grad_real[d][e] = (tmp[0] * inv_jac_xy[e][0] + tmp[1] * inv_jac_xy[e][1]);

              for(unsigned int e = 0; e < 2; ++e)
                grad_real[d][e] -= jac_grad_rank1[e] * val_real[d];

              const VectorizedArray<Number> tmp_z_grad2 =
                (inv_jac_xy[d][0] * grad[2][0] + inv_jac_xy[d][1] * grad[2][1]);

              grad_real[2][d] = (tmp_z_grad2 - jac_grad_rank1[d] * val[2]) * h_z;
              grad_real[d][2] = h_z_inverse * tmp[2];
            }
            grad_real[2][2] = grad[2][2];

            Tensor<1, dim, VectorizedArray<Number>> convective;
            for(unsigned int d = 0; d < dim; ++d)
            {
              convective[d] = grad_real[d][0] * val_real[0];
              for(unsigned int e = 1; e < dim; ++e)
                convective[d] += grad_real[d][e] * val_real[e];
              convective[d] *= inv_jac_det * inv_jac_det * h_z_inverse * h_z_inverse;
            }

            VectorizedArray<Number> vorticity_grad[3][3] = {{grads_face[1][0][q * dim],
                                                             grads_face[1][0][q * dim + 1],
                                                             grads_face[1][0][q * dim + 2]},
                                                            {grads_face[1][1][q * dim],
                                                             grads_face[1][1][q * dim + 1],
                                                             grads_face[1][1][q * dim + 2]},
                                                            {grads_face[1][2][q * dim],
                                                             grads_face[1][2][q * dim + 1],
                                                             grads_face[1][2][q * dim + 2]}};

            for(unsigned int d = 0; d < dim; ++d)
            {
              const VectorizedArray<Number> tmp = vorticity_grad[d][0];
              vorticity_grad[d][0] =
                inv_jac_xy[0][0] * tmp + inv_jac_xy[0][1] * vorticity_grad[d][1];
              vorticity_grad[d][1] =
                inv_jac_xy[1][0] * tmp + inv_jac_xy[1][1] * vorticity_grad[d][1];
              vorticity_grad[d][2] *= h_z_inverse;
            }
            Tensor<1, dim, VectorizedArray<Number>> curl_vorticity;
            if constexpr(dim == 2)
            {
              curl_vorticity[0] = vorticity_grad[0][1];
              curl_vorticity[1] = -vorticity_grad[0][0];
            }
            else
            {
              curl_vorticity[0] = vorticity_grad[2][1] - vorticity_grad[1][2];
              curl_vorticity[1] = vorticity_grad[0][2] - vorticity_grad[2][0];
              curl_vorticity[2] = vorticity_grad[1][0] - vorticity_grad[0][1];
            }

            const VectorizedArray<Number> flux =
              -normal * (convective + viscosity * curl_vorticity);

            pressure_values[q] = (mapping_info.quad_weights_z[q1] * area_element_xy * h_z) *
                                 mapping_info.quad_weights_z[qz] * flux;
          }
        }
      }
      else
      {
        std::array<unsigned int, n_lanes> shifted_data_indices;
        std::array<unsigned int, n_lanes> shifted_data_indices_gr;
        std::array<unsigned int, n_lanes> shifted_data_indices_norm;
        for(unsigned int v = 0; v < n_lanes; ++v)
        {
          const unsigned int cell_index =
            cells_at_dirichlet_boundary[cell].second[v] == numbers::invalid_unsigned_int ?
              cells_at_dirichlet_boundary[cell].second[n_lanes_filled - 1] :
              cells_at_dirichlet_boundary[cell].second[v];

          const unsigned int idx =
            mapping_info
              .face_mapping_data_index[cell_index / n_lanes][face][0][cell_index % n_lanes];
          shifted_data_indices[v]      = idx * 4;
          shifted_data_indices_gr[v]   = idx * 6;
          shifted_data_indices_norm[v] = idx * 2;
        }

        for(unsigned int q1 = 0; q1 < nn; ++q1)
        {
          Tensor<2, 2, VectorizedArray<Number>> jac_xy;
          vectorized_load_and_transpose(4,
                                        &mapping_info.face_jacobians_xy[q1][0][0],
                                        shifted_data_indices.data(),
                                        &jac_xy[0][0]);

          const VectorizedArray<Number> area_element_xy =
            std::sqrt(jac_xy[0][1 - face_direction] * jac_xy[0][1 - face_direction] +
                      jac_xy[1][1 - face_direction] * jac_xy[1][1 - face_direction]);

          const Tensor<2, 2, VectorizedArray<Number>> inv_jac_xy = transpose(invert(jac_xy));

          Tensor<1, dim, VectorizedArray<Number>> normal;
          vectorized_load_and_transpose(2,
                                        &mapping_info.face_normal_vector_xy[q1][0],
                                        shifted_data_indices_norm.data(),
                                        &normal[0]);

          for(unsigned int qz = 0, q = q1; qz < nn; ++qz, q += nn)
          {
            VectorizedArray<Number> vorticity_grad[3][3] = {{grads_face[1][0][q * dim],
                                                             grads_face[1][0][q * dim + 1],
                                                             grads_face[1][0][q * dim + 2]},
                                                            {grads_face[1][1][q * dim],
                                                             grads_face[1][1][q * dim + 1],
                                                             grads_face[1][1][q * dim + 2]},
                                                            {grads_face[1][2][q * dim],
                                                             grads_face[1][2][q * dim + 1],
                                                             grads_face[1][2][q * dim + 2]}};

            for(unsigned int d = 0; d < dim; ++d)
            {
              const VectorizedArray<Number> tmp = vorticity_grad[d][0];
              vorticity_grad[d][0] =
                inv_jac_xy[0][0] * tmp + inv_jac_xy[0][1] * vorticity_grad[d][1];
              vorticity_grad[d][1] =
                inv_jac_xy[1][0] * tmp + inv_jac_xy[1][1] * vorticity_grad[d][1];
              vorticity_grad[d][2] *= h_z_inverse;
            }
            Tensor<1, dim, VectorizedArray<Number>> curl_vorticity;
            if constexpr(dim == 2)
            {
              curl_vorticity[0] = vorticity_grad[0][1];
              curl_vorticity[1] = -vorticity_grad[0][0];
            }
            else
            {
              curl_vorticity[0] = vorticity_grad[2][1] - vorticity_grad[1][2];
              curl_vorticity[1] = vorticity_grad[0][2] - vorticity_grad[2][0];
              curl_vorticity[2] = vorticity_grad[1][0] - vorticity_grad[0][1];
            }

            const VectorizedArray<Number> flux = -normal * viscosity * curl_vorticity;

            pressure_values[q] = (mapping_info.quad_weights_z[q1] * area_element_xy * h_z) *
                                 mapping_info.quad_weights_z[qz] * flux;
          }
        }
      }
    }
    else
      AssertThrow(
        false, ExcNotImplemented("Face operations in extruded direction currently not supported"));

    // perform interpolation within face, utilize the fact that the pressure
    // space is the same as the tangential space in RT
    internal::FEEvaluationImplBasisChange<dealii::internal::evaluate_evenodd,
                                          dealii::internal::EvaluatorQuantity::value,
                                          dim - 1,
                                          degree,
                                          nn>::do_backward(1,
                                                           shape_data_pres,
                                                           false,
                                                           pressure_values,
                                                           pressure_values);

    constexpr unsigned int            data_per_face = Utilities::pow(degree, dim - 1);
    std::array<unsigned int, n_lanes> indices_of_face_data;
    for(unsigned int i = 0; i < n_lanes; ++i)
      indices_of_face_data[i] = (cell * n_lanes + i) * data_per_face;
    vectorized_transpose_and_store(
      false, data_per_face, pressure_values, indices_of_face_data.data(), pressure_rhs.begin());
  }

  template<int degree>
  void
  do_evaluate_field_on_cells(const VectorType &                        velocity_vector,
                             const std::array<unsigned int, n_lanes> & cell_indices,
                             AlignedVector<Tensor<1, dim, VectorizedArray<Number>>> & velocities,
                             const bool multiply_by_JxW) const
  {
    dealii::ndarray<unsigned int, 2 * dim + 1, n_lanes> my_dof_indices;
    for(unsigned int v = 0; v < n_lanes; ++v)
    {
      for(unsigned int i = 0; i < 2 * dim + 1; ++i)
        my_dof_indices[i][v] =
          this->dof_indices[cell_indices[v] / n_lanes][i][cell_indices[v] % n_lanes];
    }
    constexpr unsigned int n_q_points_1d = degree + 1;
    constexpr unsigned int n_points      = Utilities::pow(n_q_points_1d, dim);
    velocities.resize_fast(n_points);

    VectorizedArray<Number> quad_values[dim * n_points];
    read_cell_values<degree, n_q_points_1d>(velocity_vector.begin(), my_dof_indices, quad_values);

    constexpr unsigned int            n_q_points_2d = n_q_points_1d * n_q_points_1d;
    std::array<unsigned int, n_lanes> shifted_data_indices;
    for(unsigned int v = 0; v < n_lanes; ++v)
      shifted_data_indices[v] =
        mapping_info.mapping_data_index[cell_indices[v] / n_lanes][cell_indices[v] % n_lanes] * 4;

    const Number h_z_inverse = mapping_info.h_z_inverse;
    const Number h_z         = mapping_info.h_z;

    constexpr unsigned int nn = n_q_points_1d;

    for(unsigned int q1 = 0; q1 < n_q_points_2d; ++q1)
    {
      Tensor<2, 2, VectorizedArray<Number>> jac_xy;
      vectorized_load_and_transpose(4,
                                    &mapping_info.jacobians_xy[q1][0][0],
                                    shifted_data_indices.data(),
                                    &jac_xy[0][0]);
      const VectorizedArray<Number> inv_jac_det = Number(1.0) / (determinant(jac_xy));

      if constexpr(dim == 2)
      {
        const VectorizedArray<Number> val[2] = {quad_values[q1 * dim], quad_values[q1 * dim + 1]};
        if(multiply_by_JxW)
          for(unsigned int d = 0; d < 2; ++d)
            velocities[q1][d] =
              mapping_info.quad_weights_xy[q1] * (jac_xy[d][0] * val[0] + jac_xy[d][1] * val[1]);
        else
          for(unsigned int d = 0; d < 2; ++d)
            velocities[q1][d] = inv_jac_det * (jac_xy[d][0] * val[0] + jac_xy[d][1] * val[1]);
      }
      else // now to dim == 3
      {
        for(unsigned int qz = 0, q = q1; qz < nn; ++qz, q += n_q_points_2d)
        {
          const VectorizedArray<Number> val[3] = {quad_values[q * dim],
                                                  quad_values[q * dim + 1],
                                                  quad_values[q * dim + 2]};

          if(multiply_by_JxW)
          {
            for(unsigned int d = 0; d < 2; ++d)
              velocities[q][d] = mapping_info.quad_weights_xy[q1] *
                                 mapping_info.quad_weights_z[qz] *
                                 (jac_xy[d][0] * val[0] + jac_xy[d][1] * val[1]);
            velocities[q][2] =
              mapping_info.quad_weights_xy[q1] * mapping_info.quad_weights_z[qz] * h_z * val[2];
          }
          else
          {
            const VectorizedArray<Number> inv_jac_det_3d = inv_jac_det * h_z_inverse;
            for(unsigned int d = 0; d < 2; ++d)
              velocities[q][d] = inv_jac_det_3d * (jac_xy[d][0] * val[0] + jac_xy[d][1] * val[1]);
            velocities[q][2] = inv_jac_det * val[2];
          }
        }
      }
    }
  }

  unsigned int
  find_first_zero_bit(const std::uint64_t x)
  {
    std::uint64_t y = ~x;
    if(y == 0)
      return 64;

      // count number of trailing zero bits
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(y);
#else

    unsigned int pos = 0;
    while((y & 1) == 0)
    {
      y >>= 1;
      ++pos;
    }
    return pos;

#endif
  }

  void
  detect_dependencies_of_face_integrals()
  {
    Timer                      time;
    const Triangulation<dim> & tria = dof_handler->get_triangulation();

    std::vector<int>   touch_length(tria.n_raw_faces(), 0);
    const unsigned int n_cell_batches = dof_indices.size();
    for(unsigned int c = 0, count = 1; c < n_cell_batches; ++c)
      for(unsigned int v = 0; v < n_active_entries_per_cell_batch(c); ++v)
      {
        const auto cell = get_cell_iterator(c, v);
        for(unsigned int f = 0; f < 2 * dim; ++f)
        {
          const unsigned int face_idx = cell->face(f)->index();
          const unsigned int periodic_idx =
            (cell->at_boundary(f) && cell->has_periodic_neighbor(f)) ?
              cell->periodic_neighbor(f)->face(cell->periodic_neighbor_face_no(f))->index() :
              face_idx;
          if(touch_length[face_idx] == 0)
          {
            touch_length[face_idx]     = -count;
            touch_length[periodic_idx] = -count;
          }
          else
          {
            AssertThrow(touch_length[face_idx] < 0, ExcInternalError());
            touch_length[face_idx]     = count + touch_length[face_idx];
            touch_length[periodic_idx] = touch_length[face_idx];
          }
        }
        ++count;
      }

#if 0
    // Some debugging to check the distances of faces from first to last touch
    std::map<unsigned int, unsigned int> distances;
    for (const int a : touch_length)
      if (a > 0)
        {
          if (distances.find(a) == distances.end())
            distances[a] = 1;
          else
            ++distances[a];
        }
    std::cout << "faces: ";
    for (const auto it : distances)
      std::cout << it.first << "|" << it.second << "  ";
    std::cout << std::endl;
#endif

    face_flux_buffer_index.assign(dof_indices.size(), {});
    convective_face_flux_buffer_index.assign(dof_indices.size(), {});
    all_left_face_fluxes_from_buffer.assign(dof_indices.size(), {});
    const unsigned int n_data_per_face =
      Utilities::pow(mapping_info.quad_weights_z.size(), dim - 1) * 2 * dim;
    // velocity and convective divergence at the convective quadrature points
    const unsigned int n_data_per_face_convective =
      Utilities::pow(convective_mapping_info.quad_weights_z.size(), dim - 1) * (dim + 1);

    constexpr unsigned int     long_range_start = 16;
    std::vector<std::uint64_t> face_storage(1);
    std::vector<unsigned int>  face_index_in_storage(tria.n_raw_faces(), 0);
    unsigned int               size_of_storage = 1;
    for(unsigned int c = 0; c < n_cell_batches; ++c)
      for(unsigned int d = 0; d < dim; ++d)
      {
        for(unsigned int v = 0; v < n_active_entries_per_cell_batch(c); ++v)
        {
          const auto cell = get_cell_iterator(c, v);

          // For cells at the right boundary (including those adjacent to
          // remote MPI processes), set the index to 0 to indicate a sink
          // that will receive vectorized writes but not get picked up
          // again later. This is one single array and thus acceptable.
          if(cell->at_boundary(2 * d + 1) ||
             cell->neighbor(2 * d + 1)->subdomain_id() != cell->subdomain_id())
          {
            face_flux_buffer_index[c][2 * d + 1][v]            = 0;
            convective_face_flux_buffer_index[c][2 * d + 1][v] = 0;
            continue;
          }
          const unsigned int face_idx = cell->face(2 * d + 1)->index();

          const auto add_entry = [&](const unsigned int position) {
            const unsigned int entry_within_vector = position / 64;
            const unsigned int bit_within_entry    = position % 64;

            // flag current bit as 'occupied'
            face_storage[entry_within_vector] |= std::uint64_t(1) << bit_within_entry;
            face_index_in_storage[face_idx]         = position + 1;
            face_flux_buffer_index[c][2 * d + 1][v] = (position + 1) * n_data_per_face;
            convective_face_flux_buffer_index[c][2 * d + 1][v] =
              (position + 1) * n_data_per_face_convective;
            size_of_storage = std::max(size_of_storage, position + 2);
          };

          // Only short-distance faces are allowed in the first part of
          // the vector to ensure we have a high cache locality in some
          // part of the final data vector for the flux
          // buffer. Furthermore, the complexity of the linear search we
          // do in this array does not grow too large.
          if(touch_length[face_idx] < 40)
          {
            const unsigned int pos = find_first_zero_bit(face_storage[0]);
            if(pos < 64)
            {
              add_entry(pos);
              continue;
            }
          }
          else
          {
            // should only look in later part
            unsigned int pos = find_first_zero_bit(face_storage[0] >> long_range_start);
            pos += long_range_start;
            if(pos < 64)
            {
              add_entry(pos);
              continue;
            }
          }
          // if we did not find it in the first part, we need to perform a
          // linear search in the storage array, which we accelerate by
          // the bit-processing function
          bool added_this = false;
          for(unsigned int s = 1; s < face_storage.size(); ++s)
          {
            const unsigned int pos = find_first_zero_bit(face_storage[s]);
            if(pos < 64)
            {
              add_entry(pos + 64 * s);
              added_this = true;
              break;
            }
          }
          if(!added_this)
          {
            const unsigned int pos = face_storage.size() * 64;
            face_storage.emplace_back(1);
            face_index_in_storage[face_idx]         = pos + 1;
            face_flux_buffer_index[c][2 * d + 1][v] = (pos + 1) * n_data_per_face;
            convective_face_flux_buffer_index[c][2 * d + 1][v] =
              (pos + 1) * n_data_per_face_convective;
            size_of_storage += 1;
          }
        }
        bool face_missing = false;
        for(unsigned int v = 0; v < n_active_entries_per_cell_batch(c); ++v)
        {
          const auto         cell     = get_cell_iterator(c, v);
          const unsigned int face_idx = cell->face(2 * d)->index();
          if(face_index_in_storage[face_idx] > 0)
          {
            AssertIndexRange(face_index_in_storage[face_idx], size_of_storage);
            face_flux_buffer_index[c][2 * d][v] = face_index_in_storage[face_idx] * n_data_per_face;
            convective_face_flux_buffer_index[c][2 * d][v] =
              face_index_in_storage[face_idx] * n_data_per_face_convective;
            const unsigned int index_in_storage = (face_index_in_storage[face_idx] - 1) / 64;
            const unsigned int bit_index        = (face_index_in_storage[face_idx] - 1) % 64;
            face_storage[index_in_storage] &= ~(std::uint64_t(1) << bit_index);
          }
          else
            face_missing = true;
        }
        all_left_face_fluxes_from_buffer[c][d] = (face_missing == false);
      }
    face_flux_buffer.resize(n_data_per_face * size_of_storage);
    convective_face_flux_buffer.resize(n_data_per_face_convective * size_of_storage);

    unsigned int count_0 = 0, count_1 = 0;
    for(const auto & a : all_left_face_fluxes_from_buffer)
      for(const char b : a)
        if(b == 0)
          ++count_0;
        else
          ++count_1;

    // debug output to assess sanity of implementation
    if(false)
      Helper::print_time(static_cast<double>(count_1) / (count_0 + count_1),
                         "Face flux buffer efficiency",
                         MPI_COMM_WORLD);

#if 0
    std::cout << "Identified storage of length " << size_of_storage << " with "
              << face_flux_buffer.size() * sizeof(Number) << " bytes, quality: "
              << count_1 << " of " << (count_0 + count_1) << std::endl;

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "Time analyze dependencies: " << time.wall_time() << std::endl;
#endif
  }
};


} // namespace RTOperator
