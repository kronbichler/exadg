/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2025 by Martin Kronbichler, Shubham Goswami,
 *  Schussnig
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
#include <deal.II/fe/mapping.h>
#include <deal.II/matrix_free/matrix_free.h>


namespace Helper
{
inline void
print_time(const double        time,
           const std::string & name,
           const MPI_Comm      communicator,
           const double        total_time = 0.)
{
  dealii::Utilities::MPI::MinMaxAvg data = dealii::Utilities::MPI::min_max_avg(time, communicator);

  if(dealii::Utilities::MPI::this_mpi_process(communicator) == 0)
  {
    const unsigned int n_digits = static_cast<unsigned int>(
      std::ceil(std::log10(dealii::Utilities::MPI::n_mpi_processes(communicator))));
    std::cout << std::left << std::setw(29) << name << " " << std::setw(11) << data.min << " [p"
              << std::setw(n_digits) << data.min_index << "] " << std::setw(11) << data.avg << " "
              << std::setw(11) << data.max << " [p" << std::setw(n_digits) << data.max_index << "]";
    if(total_time > 0)
      std::cout << " " << data.avg * 100. / total_time << "%";
    std::cout << std::endl;
  }
}

// function to convert from a map, with keys associated to the buckets by
// which we sliced the index space, length chunk_size_zero_vector, and
// values equal to the slice index which are touched by the respective
// partition, to a "vectors-of-vectors" like data structure. Rather than
// using the vectors, we set up a sparsity-pattern like structure where
// one index specifies the start index (range_list_index), and the other
// the actual ranges (range_list).
inline void
convert_map_to_range_list(const unsigned int                                        n_partitions,
                          const unsigned int                                        chunk_size,
                          const std::map<unsigned int, std::vector<unsigned int>> & ranges_in,
                          std::vector<unsigned int> &                          range_list_index,
                          std::vector<std::pair<unsigned int, unsigned int>> & range_list,
                          const unsigned int                                   max_size)
{
  range_list_index.resize(n_partitions + 1);
  range_list_index[0] = 0;
  range_list.clear();
  for(unsigned int partition = 0; partition < n_partitions; ++partition)
  {
    auto it = ranges_in.find(partition);
    if(it != ranges_in.end())
    {
      for(unsigned int i = 0; i < it->second.size(); ++i)
      {
        const unsigned int first_i = i;
        while(i + 1 < it->second.size() && it->second[i + 1] == it->second[i] + 1)
          ++i;
        range_list.emplace_back(std::min(it->second[first_i] * chunk_size, max_size),
                                std::min((it->second[i] + 1) * chunk_size, max_size));
      }
      range_list_index[partition + 1] = range_list.size();
    }
    else
      range_list_index[partition + 1] = range_list_index[partition];
  }
}



template<std::size_t dim>
void
group_cells_by_left_neighbors(
  const std::array<int, dim> &                               lexicographic_index,
  const typename dealii::Triangulation<dim>::cell_iterator & cell,
  std::array<std::vector<std::pair<std::array<int, dim>,
                                   typename dealii::Triangulation<dim>::active_cell_iterator>>,
             8> &                                            cells_per_neighbor_value)
{
  if(cell->is_active() && cell->is_locally_owned())
  {
    unsigned int boundary_type = 0;
    for(unsigned int d = 0; d < dim; ++d)
      if(cell->at_boundary(2 * d) || cell->neighbor(2 * d)->subdomain_id() != cell->subdomain_id())
      {
        boundary_type += dealii::Utilities::pow(2, d);
      }
    cells_per_neighbor_value[boundary_type].emplace_back(lexicographic_index, cell);
  }
  else if(cell->has_children() && cell->n_children() == dealii::Utilities::pow(2, dim))
    for(unsigned int child = 0, d2 = 0; d2 < (dim > 2 ? 2 : 1); ++d2)
      for(unsigned int d1 = 0; d1 < (dim > 1 ? 2 : 1); ++d1)
        for(unsigned int d0 = 0; d0 < 2; ++d0, ++child)
        {
          std::array<int, dim> child_lexicographic;
          child_lexicographic[dim - 1] = lexicographic_index[dim - 1] * 2 + d0;
          if constexpr(dim > 1)
            child_lexicographic[dim - 2] = lexicographic_index[dim - 2] * 2 + d1;
          if constexpr(dim > 2)
            child_lexicographic[dim - 3] = lexicographic_index[dim - 3] * 2 + d2;
          group_cells_by_left_neighbors(child_lexicographic,
                                        cell->child(child),
                                        cells_per_neighbor_value);
        }
  else
    AssertThrow(cell->has_children() == false,
                dealii::ExcMessage("Only hypercube elements allowed for this code path"));
}



// Create categories of cells to ensure optimal use of the face flux buffer
// in the context of vectorization: we would like to place cells at the
// boundary or adjacent to remote processes (requiring different algorithms
// for the face data) in a lower category such that they get processed
// first. The assignment aims for having as few cells as possible in batches
// that are more expensive, i.e., to have cells placed together that have
// the same faces with the increased cost
template<int dim>
std::vector<unsigned int>
compute_vectorization_category(const dealii::Triangulation<dim> & tria)
{
  std::vector<unsigned int> cell_vectorization_category(tria.n_active_cells(),
                                                        dealii::numbers::invalid_unsigned_int);

  unsigned int n_owned_cells = 0;
  for(const auto & cell : tria.active_cell_iterators())
    if(cell->is_locally_owned())
      ++n_owned_cells;

  unsigned int category_counter = 0;

  // Store:
  //   - lexicographic coordinates (for ordering) - note that the z index
  //     comes first to allow standard sorting, this impacts the use of the
  //     index below
  //   - corresponding active cell iterator
  using CellLexEntry =
    std::pair<std::array<int, dim>, typename dealii::Triangulation<dim>::active_cell_iterator>;

  // Partition cells according to neighbor configuration.
  // Index meaning:
  //   0 = fully interior cells (all left neighbors treated in the same code,
  //       providing opportunity to use a flux buffer from right faces)
  //   1,2,... = cells touching specific subdomain boundary configurations or
  //             the physical boundary
  std::array<std::vector<CellLexEntry>, 8> cells_per_neighbor_value;

  // Limit the number of levels we use for the lexicographic traversal,
  // avoiding bad cache usage. For low degrees, it might actually make sense
  // to allow for one more level, but in general the utilization with 8^dim
  // cells is already very good in this case and improvements are rather
  // minor from allowing up to 16^dim cells.
  unsigned int blocking_level = 0;
  if(tria.n_levels() > 4)
    blocking_level = tria.n_levels() - 4;

  for(const auto & cell : tria.cell_iterators_on_level(blocking_level))
  {
    // Group cells by their left neighbor, and within each group create
    // a lexicographic ordering
    for(auto & local_vector : cells_per_neighbor_value)
      local_vector.clear();

    // Recursively group descendant cells by left-neighbor pattern and
    // collect lexicographic positions.
    group_cells_by_left_neighbors(std::array<int, dim>{}, cell, cells_per_neighbor_value);

    // Sort each group lexicographically
    for(auto & local_vector : cells_per_neighbor_value)
      std::sort(local_vector.begin(), local_vector.end());

    // First do xy base layer with z faces at boundary and the category of
    // z > 0, x=y=0
    for(unsigned int neighbor_id = 7; neighbor_id > 2; --neighbor_id)
      for(auto & entry : cells_per_neighbor_value[neighbor_id])
        cell_vectorization_category[entry.second->active_cell_index()] = category_counter++;

    // Then proceed layer by layer, including some blocking for cells
    // along the y/z faces and the interior to increase the data locality;
    // the SIMD width determines preferred category block size
    constexpr unsigned int n_lanes = dealii::VectorizedArray<float>::size();
    std::array<typename std::vector<CellLexEntry>::iterator, 3> lex_ptrs{
      {cells_per_neighbor_value[0].begin(),
       cells_per_neighbor_value[1].begin(),
       cells_per_neighbor_value[2].begin()}};
    const std::array<typename std::vector<CellLexEntry>::iterator, 3> lex_ends{
      {cells_per_neighbor_value[0].end(),
       cells_per_neighbor_value[1].end(),
       cells_per_neighbor_value[2].end()}};

    // First fill up the lanes with the previously accumulated info,
    // making sure that the remaining parts also fit within the available
    // lanes to ensure a good utilization
    auto assign_category = [&](auto & it) {
      cell_vectorization_category[it->second->active_cell_index()] = category_counter++;
      ++it;
    };

    // Before proceeding with the next category, fill up with cells whose
    // pattern should already be present in the same vectorization batch: we
    // either had category 3 or some previous one; in either case, we should
    // expect that categories 1 and 2 fit well; we first try to use cells of
    // those categories until they are divisible by the SIMD length
    // themselves, then we use some of the remaining cells
    while(category_counter % n_lanes != 0)
    {
      if(lex_ptrs[2] != lex_ends[2] && (lex_ends[2] - lex_ptrs[2]) % n_lanes > 0)
        assign_category(lex_ptrs[2]);
      else if(lex_ptrs[1] != lex_ends[1] && (lex_ends[1] - lex_ptrs[1]) % n_lanes > 0)
        assign_category(lex_ptrs[1]);
      else if(lex_ptrs[2] != lex_ends[2])
        assign_category(lex_ptrs[2]);
      else if(lex_ptrs[1] != lex_ends[1])
        assign_category(lex_ptrs[1]);
      else if(lex_ptrs[0] != lex_ends[0])
        assign_category(lex_ptrs[0]);
      else
        break;
    }

    // Then proceed layer by layer, always making sure that the interior
    // part with index 0 (= all faces have a left neighbor and can use the
    // flux buffer) comes after the other cells to not create bubbles
    // without 'left' neighbor data accessible in flux buffer
    while(lex_ptrs[0] != lex_ends[0])
    {
      int z_index = std::numeric_limits<int>::max();
      if((lex_ends[0] - lex_ptrs[0]) >= n_lanes)
        z_index = (lex_ptrs[0] + n_lanes)->first[0];
      while(lex_ptrs[2] != lex_ends[2] && lex_ptrs[2]->first[0] <= z_index)
        assign_category(lex_ptrs[2]);
      while(lex_ptrs[2] != lex_ends[2] && category_counter % n_lanes != 0)
        assign_category(lex_ptrs[2]);
      while(lex_ptrs[1] != lex_ends[1] && lex_ptrs[1]->first[0] <= z_index)
        assign_category(lex_ptrs[1]);
      while(lex_ptrs[1] != lex_ends[1] && category_counter % n_lanes != 0)
        assign_category(lex_ptrs[1]);
      do
        assign_category(lex_ptrs[0]);
      while(lex_ptrs[0] != lex_ends[0] && category_counter % n_lanes != 0);
    }
  }
  // Final consistency check:
  // every locally owned cell must have received exactly one category.
  AssertThrow(category_counter == n_owned_cells,
              dealii::ExcDimensionMismatch(category_counter, n_owned_cells));
  return cell_vectorization_category;
}

} // namespace Helper


namespace Extruded
{
using namespace dealii;
template<int dim, typename Number>
struct MappingInfo
{
  static constexpr unsigned int n_lanes = VectorizedArray<Number>::size();

  void
  reinit(const Mapping<dim> &                                         mapping,
         const Quadrature<1> &                                        quadrature_1d,
         const Triangulation<dim> &                                   triangulation,
         const std::vector<dealii::ndarray<unsigned int, n_lanes, 2>> cell_level_index)
  {
    underlying_quadrature = quadrature_1d;

    FloatingPointComparator<double> comparator(triangulation.begin()->diameter() * 1e-10);
    std::map<Point<2>, std::array<int, 3>, FloatingPointComparator<double>> unique_cells(
      comparator);

    std::vector<unsigned int> geometry_index(triangulation.n_active_cells(),
                                             numbers::invalid_unsigned_int);
    // collect possible compression of Jacobian data due to extrusion in z
    // direction by checking the position of the first cell vertex in the xy
    // plane. First check locally owned cells in exactly the order matrix-free
    // loops visit them, then for ghosts to get the complete face data also in
    // parallel computations
    for(unsigned int c = 0; c < cell_level_index.size(); ++c)
      for(unsigned int v = 0;
          v < n_lanes && cell_level_index[c][v][0] != numbers::invalid_unsigned_int;
          ++v)
      {
        const typename Triangulation<dim>::active_cell_iterator cell(&triangulation,
                                                                     cell_level_index[c][v][0],
                                                                     cell_level_index[c][v][1]);
        const auto vertices = mapping.get_vertices(cell);
        Point<2>   p(vertices[0][0], vertices[0][1]);
        const auto position = unique_cells.find(p);
        if(position == unique_cells.end())
        {
          geometry_index[cell->active_cell_index()] = unique_cells.size();
          unique_cells.insert(std::make_pair(
            p, std::array<int, 3>{{(int)unique_cells.size(), cell->level(), cell->index()}}));
        }
        else
          geometry_index[cell->active_cell_index()] = position->second[0];
      }
    for(const auto & cell : triangulation.active_cell_iterators())
      if(cell->is_ghost())
      {
        const auto vertices = mapping.get_vertices(cell);
        Point<2>   p(vertices[0][0], vertices[0][1]);
        const auto position = unique_cells.find(p);
        if(position == unique_cells.end())
        {
          geometry_index[cell->active_cell_index()] = unique_cells.size();
          unique_cells.insert(std::make_pair(
            p, std::array<int, 3>{{(int)unique_cells.size(), cell->level(), cell->index()}}));
        }
        else
          geometry_index[cell->active_cell_index()] = position->second[0];
      }

    const unsigned int n_q_points_1d = quadrature_1d.size();
    const unsigned int n_q_points_2d = Utilities::pow(n_q_points_1d, 2);
    mapping_data_index.resize(cell_level_index.size());
    face_mapping_data_index.resize(cell_level_index.size());
    for(unsigned int c = 0; c < cell_level_index.size(); ++c)
      for(unsigned int v = 0; v < n_lanes; ++v)
      {
        if(cell_level_index[c][v][0] == numbers::invalid_unsigned_int)
        {
          for(unsigned int f = 0; f < 4; ++f)
            for(unsigned int s = 0; s < 2; ++s)
              face_mapping_data_index[c][f][s][v] = face_mapping_data_index[c][f][s][0];
          continue;
        }
        const typename Triangulation<dim>::active_cell_iterator dcell(&triangulation,
                                                                      cell_level_index[c][v][0],
                                                                      cell_level_index[c][v][1]);
        mapping_data_index[c][v] = geometry_index[dcell->active_cell_index()] * n_q_points_2d;
        for(unsigned int f = 0; f < 4; ++f)
        {
          face_mapping_data_index[c][f][0][v] =
            (geometry_index[dcell->active_cell_index()] * 4 + f) * n_q_points_1d;
          const bool at_boundary           = dcell->at_boundary(f);
          const bool has_periodic_neighbor = at_boundary && dcell->has_periodic_neighbor(f);
          if(at_boundary == false || has_periodic_neighbor)
          {
            const auto neighbor =
              has_periodic_neighbor ? dcell->periodic_neighbor(f) : dcell->neighbor(f);
            const unsigned int neighbor_face_idx = has_periodic_neighbor ?
                                                     dcell->periodic_neighbor_face_no(f) :
                                                     dcell->neighbor_face_no(f);
            face_mapping_data_index[c][f][1][v] =
              (geometry_index[neighbor->active_cell_index()] * 4 + neighbor_face_idx) *
              n_q_points_1d;
          }
          else
            face_mapping_data_index[c][f][1][v] = face_mapping_data_index[c][f][0][v];
        }
      }

    quad_weights_z.resize(n_q_points_1d);
    for(unsigned int q = 0; q < n_q_points_1d; ++q)
      quad_weights_z[q] = quadrature_1d.weight(q);

    Quadrature<dim - 1> face_quadrature(quadrature_1d);
    Quadrature<2>       quadrature_2d(quadrature_1d);
    quad_weights_xy.resize(quadrature_2d.size());
    for(unsigned int q = 0; q < quadrature_2d.size(); ++q)
      quad_weights_xy[q] = quadrature_2d.weight(q);

    std::vector<Point<dim>> points(quadrature_2d.size());
    for(unsigned int i = 0; i < quadrature_2d.size(); ++i)
      for(unsigned int d = 0; d < 2; ++d)
        points[i][d] = quadrature_2d.point(i)[d];

    FE_Nothing<dim>   dummy_fe;
    FEValues<dim>     fe_values(mapping,
                            dummy_fe,
                            Quadrature<dim>(points),
                            update_jacobians | update_jacobian_grads | update_quadrature_points);
    FEFaceValues<dim> fe_face_values(mapping,
                                     dummy_fe,
                                     face_quadrature,
                                     update_quadrature_points | update_jacobians |
                                       update_JxW_values | update_normal_vectors |
                                       update_jacobian_grads);

    quadrature_points.resize(n_q_points_2d * unique_cells.size());
    jacobians_xy.resize(n_q_points_2d * unique_cells.size());
    inv_jacobians_xy.resize(n_q_points_2d * unique_cells.size());
    cell_JxW_xy.resize(n_q_points_2d * unique_cells.size());
    jacobian_grads.resize(n_q_points_2d * unique_cells.size());

    face_quadrature_points.resize(4 * n_q_points_1d * unique_cells.size());
    face_jacobians_xy.resize(4 * n_q_points_1d * unique_cells.size());
    face_jacobian_grads.resize(4 * n_q_points_1d * unique_cells.size());
    face_normal_vector_xy.resize(4 * n_q_points_1d * unique_cells.size());
    face_jxn_xy.resize(4 * n_q_points_1d * unique_cells.size());
    face_JxW_xy.resize(4 * n_q_points_1d * unique_cells.size());

    ip_penalty_factors.resize(unique_cells.size());
    for(const auto & [_, index] : unique_cells)
    {
      const typename Triangulation<dim>::cell_iterator cell(&triangulation, index[1], index[2]);
      fe_values.reinit(cell);
      AssertDimension(geometry_index[cell->active_cell_index()], index[0]);
      double cell_volume = 0;
      for(unsigned int q = 0; q < n_q_points_2d; ++q)
      {
        const DerivativeForm<1, dim, dim> jacobian = fe_values.jacobian(q);
        cell_volume += jacobian.determinant() * face_quadrature.weight(q);
        const DerivativeForm<1, dim, dim> inv_jacobian = jacobian.covariant_form();
        const unsigned int                data_idx     = index[0] * n_q_points_2d + q;
        cell_JxW_xy[data_idx] =
          (jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0]) *
          face_quadrature.weight(q);
        if(dim == 3)
        {
          h_z         = jacobian[2][2];
          h_z_inverse = 1. / h_z;
        }
        for(unsigned int d = 0; d < 2; ++d)
          for(unsigned int e = 0; e < 2; ++e)
          {
            jacobians_xy[data_idx][d][e]     = jacobian[d][e];
            inv_jacobians_xy[data_idx][d][e] = inv_jacobian[d][e];
          }
        for(unsigned int d = 0; d < 2; ++d)
          quadrature_points[data_idx][d] = fe_values.quadrature_point(q)[d];

        const auto jac_grad = fe_values.jacobian_grad(q);
        for(unsigned int d = 0; d < 2; ++d)
        {
          jacobian_grads[data_idx][0][d] = jac_grad[d][0][0];
          jacobian_grads[data_idx][1][d] = jac_grad[d][1][1];
          jacobian_grads[data_idx][2][d] = jac_grad[d][0][1];
        }
      }

      double surface_area = 0;
      for(unsigned int face = 0; face < 4; ++face)
      {
        fe_face_values.reinit(cell, face);
        double face_factor =
          (cell->at_boundary(face) && !cell->has_periodic_neighbor(face)) ? 1. : 0.5;
        for(unsigned int q = 0; q < face_quadrature.size(); ++q)
          surface_area += face_factor * fe_face_values.JxW(q);

        for(unsigned int qx = 0; qx < n_q_points_1d; ++qx)
        {
          // take switched coordinates xz on y faces into account
          const unsigned int q        = (face < 2 || dim == 2) ? qx : qx * n_q_points_1d;
          const unsigned int data_idx = (index[0] * 4 + face) * n_q_points_1d + qx;
          const auto         jac      = fe_face_values.jacobian(q);
          const auto         inv_jac  = jac.covariant_form();
          for(unsigned int d = 0; d < 2; ++d)
            face_quadrature_points[data_idx][d] = fe_face_values.quadrature_point(q)[d];
          for(unsigned int d = 0; d < 2; ++d)
            face_jxn_xy[data_idx][d] = inv_jac[0][d] * fe_face_values.normal_vector(q)[0] +
                                       inv_jac[1][d] * fe_face_values.normal_vector(q)[1];
          face_JxW_xy[data_idx] = std::sqrt(jac[0][1 - face / 2] * jac[0][1 - face / 2] +
                                            jac[1][1 - face / 2] * jac[1][1 - face / 2]) *
                                  quadrature_1d.weight(qx);
          for(unsigned int d = 0; d < 2; ++d)
            for(unsigned int e = 0; e < 2; ++e)
              face_jacobians_xy[data_idx][d][e] = jac[d][e];
          const auto jac_grad = fe_face_values.jacobian_grad(q);
          for(unsigned int d = 0; d < 2; ++d)
          {
            face_jacobian_grads[data_idx][0][d] = jac_grad[d][0][0];
            face_jacobian_grads[data_idx][1][d] = jac_grad[d][1][1];
            face_jacobian_grads[data_idx][2][d] = jac_grad[d][0][1];
            face_normal_vector_xy[data_idx][d]  = fe_face_values.normal_vector(q)[d];
          }
        }
      }
      // take the two faces in z direction into account; they are always in
      // periodic direction so do not check for boundary
      if(dim == 3)
        surface_area += 2 * 0.5 * cell_volume / h_z;

      ip_penalty_factors[index[0]] = surface_area / cell_volume;
    }
  }

  std::size_t
  memory_consumption() const
  {
    return MemoryConsumption::memory_consumption(mapping_data_index) +
           MemoryConsumption::memory_consumption(quadrature_points) +
           MemoryConsumption::memory_consumption(jacobians_xy) +
           MemoryConsumption::memory_consumption(inv_jacobians_xy) +
           MemoryConsumption::memory_consumption(jacobian_grads) +
           MemoryConsumption::memory_consumption(cell_JxW_xy) +
           MemoryConsumption::memory_consumption(face_mapping_data_index) +
           MemoryConsumption::memory_consumption(face_quadrature_points) +
           MemoryConsumption::memory_consumption(face_normal_vector_xy) +
           MemoryConsumption::memory_consumption(face_jacobians_xy) +
           MemoryConsumption::memory_consumption(face_jacobian_grads) +
           MemoryConsumption::memory_consumption(face_jxn_xy) +
           MemoryConsumption::memory_consumption(face_JxW_xy) +
           MemoryConsumption::memory_consumption(quad_weights_xy) +
           MemoryConsumption::memory_consumption(quad_weights_z) +
           MemoryConsumption::memory_consumption(ip_penalty_factors) + sizeof(this);
  }

  Number                                            h_z;
  Number                                            h_z_inverse;
  std::vector<std::array<unsigned int, n_lanes>>    mapping_data_index;
  AlignedVector<Point<2, Number>>                   quadrature_points;
  AlignedVector<Tensor<2, 2, Number>>               jacobians_xy;
  AlignedVector<Tensor<2, 2, Number>>               inv_jacobians_xy;
  AlignedVector<Number>                             cell_JxW_xy;
  AlignedVector<Tensor<1, 3, Tensor<1, 2, Number>>> jacobian_grads;
  std::vector<ndarray<unsigned int, 4, 2, n_lanes>> face_mapping_data_index;
  AlignedVector<Point<2, Number>>                   face_quadrature_points;
  AlignedVector<Tensor<1, 2, Number>>               face_normal_vector_xy;
  AlignedVector<Tensor<2, 2, Number>>               face_jacobians_xy;
  AlignedVector<Tensor<1, 3, Tensor<1, 2, Number>>> face_jacobian_grads;
  AlignedVector<Tensor<1, 2, Number>>               face_jxn_xy;
  AlignedVector<Number>                             face_JxW_xy;
  std::vector<Number>                               quad_weights_xy;
  std::vector<Number>                               quad_weights_z;
  std::vector<Number>                               ip_penalty_factors;
  Quadrature<1>                                     underlying_quadrature;
};

} // namespace Extruded
