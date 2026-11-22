/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2025 - 2026 by Martin Kronbichler, Shubham Goswami,
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
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <algorithm>
#include <map>
#include <set>
#include <tuple>

#ifdef __linux__
#  include <sched.h>
#endif


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



// Number of shared-memory domains (nodes) of the communicator
inline unsigned int
n_nodes(const MPI_Comm communicator)
{
  MPI_Comm  comm_node;
  const int rank = dealii::Utilities::MPI::this_mpi_process(communicator);
  int       ierr =
    MPI_Comm_split_type(communicator, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &comm_node);
  AssertThrowMPI(ierr);
  const unsigned int n_nodes =
    dealii::Utilities::MPI::sum(dealii::Utilities::MPI::this_mpi_process(comm_node) == 0 ? 1U : 0U,
                                communicator);
  ierr = MPI_Comm_free(&comm_node);
  AssertThrowMPI(ierr);
  return n_nodes;
}



// Like print_time(), but for the average of the time over the processes of
// each shared-memory domain (node), i.e., minimum, average and maximum over
// the nodes, with a process of the respective node in brackets. Nothing is
// printed if there is only one node.
inline void
print_time_per_node(const double        time,
                    const std::string & name,
                    const MPI_Comm      communicator,
                    const double        total_time = 0.)
{
  MPI_Comm  comm_node;
  const int rank = dealii::Utilities::MPI::this_mpi_process(communicator);
  int       ierr =
    MPI_Comm_split_type(communicator, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &comm_node);
  AssertThrowMPI(ierr);
  const double node_average = dealii::Utilities::MPI::sum(time, comm_node) /
                              dealii::Utilities::MPI::n_mpi_processes(comm_node);
  ierr = MPI_Comm_free(&comm_node);
  AssertThrowMPI(ierr);
  if(n_nodes(communicator) > 1)
    print_time(node_average, name, communicator, total_time);
}



// Print the distribution of the processes over the shared-memory domains
// (nodes) and the number of CPUs each process may run on according to its
// affinity mask, which shows how the processes are bound
inline void
print_parallel_layout(const MPI_Comm communicator, const std::string & note)
{
  MPI_Comm  comm_node;
  const int rank = dealii::Utilities::MPI::this_mpi_process(communicator);
  int       ierr =
    MPI_Comm_split_type(communicator, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &comm_node);
  AssertThrowMPI(ierr);
  const auto ranks_per_node = dealii::Utilities::MPI::min_max_avg(
    static_cast<double>(dealii::Utilities::MPI::n_mpi_processes(comm_node)), communicator);
  ierr = MPI_Comm_free(&comm_node);
  AssertThrowMPI(ierr);

  double n_cpus = 0;
#ifdef __linux__
  cpu_set_t cpu_set;
  CPU_ZERO(&cpu_set);
  if(sched_getaffinity(0, sizeof(cpu_set), &cpu_set) == 0)
    n_cpus = CPU_COUNT(&cpu_set);
#endif
  const auto cpus = dealii::Utilities::MPI::min_max_avg(n_cpus, communicator);

  const unsigned int nodes = n_nodes(communicator);
  if(rank == 0)
    std::cout << "Parallel layout: " << dealii::Utilities::MPI::n_mpi_processes(communicator)
              << " processes on " << nodes << " node(s), "
              << static_cast<unsigned int>(ranks_per_node.min) << "-"
              << static_cast<unsigned int>(ranks_per_node.max) << " per node, "
              << static_cast<unsigned int>(cpus.min) << "-" << static_cast<unsigned int>(cpus.max)
              << " CPUs in affinity mask per process; " << note << std::endl
              << std::endl;
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



// Our algorithm use the fluxes computed on the right face as input to the
// left faces of the neighboring cell through a flux buffer. However, this can
// only be done for faces that are locally owned and if the cell on the left
// has already computed its right faces. We therefore try to set up an order
// of cells that is beneficial according to this metric, which we achieve by a
// local lexicographic ordering of the cells.
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
    // query the MPI ranks set by deal.II to see whether the neighbor cell is
    // owned by another MPI process (given by hte subdomain_id of deal.II) or
    // if we are at the boundary. In those cases, we need to compute the
    // fluxes, so they are given separate type, which we then use to group
    // cells further down
    unsigned int boundary_type = 0;
    for(unsigned int d = 0; d < dim; ++d)
      if(cell->at_boundary(2 * d) || cell->neighbor(2 * d)->subdomain_id() != cell->subdomain_id())
      {
        boundary_type += dealii::Utilities::pow(2, d);
      }
    AssertIndexRange(boundary_type, cells_per_neighbor_value.size());
    cells_per_neighbor_value[boundary_type].emplace_back(lexicographic_index, cell);
  }
  else if(cell->has_children())
  {
    AssertThrow(cell->n_children() == dealii::Utilities::pow(2, dim) &&
                  cell->reference_cell().is_hyper_cube(),
                dealii::ExcMessage("Only hypercube elements allowed for this code path"));

    for(unsigned int child = 0; child < cell->n_children(); ++child)
    {
      std::array<int, dim> child_lexicographic;
      // retrieve the lexicographic index from the linear index of cell
      // children. Note that we switch the index to have the last index in
      // the first index array because we later want to sort data
      // lexicographically

      for(unsigned int d = 0; d < dim; ++d)
      {
        // compute bitmask along current direction
        const unsigned int bit           = (child >> d) & 1;
        child_lexicographic[dim - 1 - d] = lexicographic_index[dim - 1 - d] * 2 + bit;
      }

      group_cells_by_left_neighbors(child_lexicographic,
                                    cell->child(child),
                                    cells_per_neighbor_value);
    }
  }
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

  // Limit the number of levels of cells that will be traversed
  // lexicographically. This is because lexicographic ordering in the limit
  // case leads to worse cache usage than a hierarchical (Morton-like)
  // ordering, and the reason we do this is like the cache blocking strategies
  // for finite differences, namely to switch to loop tiling. Using this
  // setting will slightly reduce the ratio of faces that can use the flux
  // buffer, but in comparison the cost is lower. For low degrees, it might
  // make sense to allow for one more level, but in general the flux buffer
  // utilization with 8^dim cells is already very good in this case and
  // improvements are rather minor from allowing up to 16^dim cells.
  // With few cells per process, blocks of 4^dim cells are better: on the
  // periodic hill with 768-1024 cells per process (k=5, 12-16 processes on
  // AMD Zen 5), they made the Chebyshev iteration 1-9% and the products
  // with 'before_loop' and 'after_loop' 3-10% faster, as the source vector
  // is reused after 4^(dim-1) instead of 8^(dim-1) cells, at the price of
  // 2% (double) or 5% (float) fewer faces using the flux buffer. With 6144
  // cells per process, both sizes were within 2%.
  const unsigned int block_levels   = n_owned_cells < 4096 ? 3 : 4;
  unsigned int       blocking_level = 0;
  if(tria.n_levels() > block_levels)
    blocking_level = tria.n_levels() - block_levels;

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
    // making sure that the remaining parts also fits within the available
    // lanes to ensure a good utilization.
    auto assign_category = [&](auto & it) {
      cell_vectorization_category[it->second->active_cell_index()] = category_counter++;
      ++it;
    };

    // Before proceeding with the next category, fill up with cells whose
    // pattern should already be present in the same vectorization batch: we
    // either had category 3 or some previous one; in either case, we should
    // expect that categories 1 and 2 fit well; we first try to use cells of
    // those categories until they are divisible by the SIMD length
    // themselves, then we use some of the remaining cells.
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
        z_index = (lex_ptrs[0] + n_lanes - 1)->first[0];
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

    // Cells of categories 1 and 2 left when the block has no more interior
    // cells, e.g. in blocks of few cells at the boundary to other processes
    for(unsigned int i = 1; i < 3; ++i)
      while(lex_ptrs[i] != lex_ends[i])
        assign_category(lex_ptrs[i]);
  }
  // Final consistency check:
  // every locally owned cell must have received exactly one category.
  AssertThrow(category_counter == n_owned_cells,
              dealii::ExcDimensionMismatch(category_counter, n_owned_cells));
  return cell_vectorization_category;
}



// Renumber the locally owned DoFs of a Raviart-Thomas element for the loop
// of RTOperator::RaviartThomasOperator, which processes the cells in the
// order given by 'cell_vectorization_category' (as computed by
// compute_vectorization_category(), with a unique category per locally owned
// cell). The DoFs are numbered by entities (faces, cell interiors), which
// need to be contiguous in the current numbering, as is the case after
// dealii::DoFRenumbering::matrix_free_data_locality(). The numbering is:
//   1. the entities of the cells with a face to a cell of another process,
//      which are read by other processes and thus pass through the function
//      'before_loop' of the matrix-vector product ahead of the cell loop,
//   2. the entities of the other cells,
//   3. the DoFs on faces to cells of other processes, which are also sent
//      to other processes as ghost values,
//   4. the DoFs constrained by 'constraints'.
// Within 1 and 2, the entities are numbered in the order of the last write
// in the cell loop, such that the ranges of the vector completed by the loop
// (passed to 'after_loop') are contiguous, and the entries ahead of the loop
// form a single range. On the periodic hill (12-16 processes), this made the
// Chebyshev iteration 4-10% and the products with 'before_loop' and
// 'after_loop' of SolverCG 1-6% faster than the numbering of
// dealii::DoFRenumbering::matrix_free_data_locality(), which uses the default
// order of the cells in dealii::MatrixFree.
//
// The constraints are used to identify the DoFs on periodic faces with the
// ones on the other side and the constrained DoFs; they need to be
// recomputed after the renumbering.
template<int dim, typename Number>
void
renumber_dofs_in_cell_loop_order(dealii::DoFHandler<dim> &                 dof_handler,
                                 const dealii::AffineConstraints<Number> & constraints,
                                 const std::vector<unsigned int> & cell_vectorization_category)
{
  using CellIterator                     = typename dealii::DoFHandler<dim>::cell_iterator;
  const dealii::IndexSet & owned         = dof_handler.locally_owned_dofs();
  const auto &             fe            = dof_handler.get_fe();
  const unsigned int       dofs_per_face = fe.n_dofs_per_face();
  const unsigned int       dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int       invalid       = dealii::numbers::invalid_unsigned_int;
  AssertThrow(cell_vectorization_category.size() ==
                dof_handler.get_triangulation().n_active_cells(),
              dealii::ExcDimensionMismatch(cell_vectorization_category.size(),
                                           dof_handler.get_triangulation().n_active_cells()));

  // first DoF of each of the 2*dim+1 entities, after resolving the
  // identity constraints of periodic faces, and the number of DoFs
  std::vector<dealii::types::global_dof_index> dof_indices(dofs_per_cell);
  const auto                                   get_entities = [&](const CellIterator & cell) {
    cell->get_dof_indices(dof_indices);
    std::array<std::pair<dealii::types::global_dof_index, unsigned int>, 2 * dim + 1> entities;
    for(unsigned int e = 0; e < 2 * dim + 1; ++e)
    {
      const unsigned int n_dofs =
        e < 2 * dim ? dofs_per_face : dofs_per_cell - 2 * dim * dofs_per_face;
      for(unsigned int i = 1; i < n_dofs; ++i)
        AssertThrow(dof_indices[e * dofs_per_face + i] == dof_indices[e * dofs_per_face] + i,
                    dealii::ExcMessage("The DoFs of each entity need to be contiguous"));
      dealii::types::global_dof_index first = dof_indices[e * dofs_per_face];
      const auto                      line = constraints.get_constraint_entries(first);
      if(line != nullptr && line->size() == 1)
        first = (*line)[0].first;
      entities[e] = {first, n_dofs};
    }
    return entities;
  };
  const auto get_neighbor = [](const CellIterator & cell, const unsigned int f) {
    if(cell->at_boundary(f) && cell->has_periodic_neighbor(f))
      return cell->periodic_neighbor(f);
    else if(!cell->at_boundary(f))
      return cell->neighbor(f);
    else
      return CellIterator();
  };

  // locally owned cells in the order of the loop
  std::vector<std::pair<unsigned int, CellIterator>> cells;
  for(const auto & cell : dof_handler.active_cell_iterators())
    if(cell->is_locally_owned())
      cells.emplace_back(cell_vectorization_category[cell->active_cell_index()], cell);
  std::sort(cells.begin(), cells.end(), [](const auto & a, const auto & b) {
    return a.first < b.first;
  });

  // DoFs on faces to remote cells, entities of cells next to remote cells,
  // and the number of cells writing into each entity
  std::vector<unsigned char>                              is_remote(owned.n_elements(), 0);
  std::set<dealii::types::global_dof_index>               surface_entities;
  std::map<dealii::types::global_dof_index, unsigned int> n_writers;
  for(const auto & [category, cell] : cells)
  {
    const auto entities   = get_entities(cell);
    bool       is_surface = false;
    for(unsigned int f = 0; f < 2 * dim; ++f)
    {
      const auto neighbor = get_neighbor(cell, f);
      if(neighbor.state() == dealii::IteratorState::valid && !neighbor->is_locally_owned())
      {
        is_surface = true;
        if(owned.is_element(entities[f].first))
          for(unsigned int i = 0; i < entities[f].second; ++i)
            is_remote[owned.index_within_set(entities[f].first + i)] = 1;
      }
    }
    for(const auto & entity : entities)
    {
      ++n_writers[entity.first];
      if(is_surface)
        surface_entities.insert(entity.first);
    }
  }

  std::vector<unsigned int> new_numbers(owned.n_elements(), invalid);
  std::vector<unsigned int> interior_order, remote_order;
  unsigned int              counter = 0;
  for(const auto & [category, cell] : cells)
    for(const auto & entity : get_entities(cell))
    {
      if(--n_writers[entity.first] > 0 || !owned.is_element(entity.first) ||
         constraints.is_constrained(entity.first))
        continue;
      const unsigned int local = owned.index_within_set(entity.first);
      AssertThrow(new_numbers[local] == invalid, dealii::ExcInternalError());
      for(unsigned int i = local; i < local + entity.second; ++i)
        if(is_remote[i])
        {
          remote_order.push_back(i);
          new_numbers[i] = invalid - 1;
        }
        else if(surface_entities.count(entity.first) == 0)
        {
          interior_order.push_back(i);
          new_numbers[i] = invalid - 1;
        }
        else
          new_numbers[i] = counter++;
    }
  for(const unsigned int i : interior_order)
    new_numbers[i] = counter++;
  for(const unsigned int i : remote_order)
    new_numbers[i] = counter++;
  for(unsigned int & n : new_numbers)
    if(n == invalid)
      n = counter++;
  AssertThrow(counter == owned.n_elements(), dealii::ExcInternalError());

  std::vector<dealii::types::global_dof_index> new_global(owned.n_elements());
  for(unsigned int i = 0; i < owned.n_elements(); ++i)
    new_global[i] = owned.nth_index_in_set(new_numbers[i]);
  dof_handler.renumber_dofs(new_global);
}



// Renumber the locally owned DoFs of a discontinuous element for the cell
// loop of LaplaceOperator::LaplaceOperatorDG, which processes the cells in
// the order given by 'cell_vectorization_category' (with a unique category
// per locally owned cell, see compute_vectorization_category()). The DoFs
// of each cell need to be contiguous, as after DoFHandler::distribute_dofs()
// for discontinuous elements, and stay so. As for the velocity in
// renumber_dofs_in_cell_loop_order(), the cells with a face to a cell of
// another process come first, as their data passes through 'before_loop'
// ahead of the cell loop, followed by the other cells, both in the order of
// the cell loop. With the same categories as for the velocity, this also
// gives the cell order of the coupling terms of
// RTOperator::RaviartThomasOperator with the pressure.
template<int dim>
void
renumber_dg_dofs_in_cell_loop_order(dealii::DoFHandler<dim> &         dof_handler,
                                    const std::vector<unsigned int> & cell_vectorization_category)
{
  using CellIterator                     = typename dealii::DoFHandler<dim>::active_cell_iterator;
  const dealii::IndexSet & owned         = dof_handler.locally_owned_dofs();
  const unsigned int       dofs_per_cell = dof_handler.get_fe().n_dofs_per_cell();
  AssertThrow(dof_handler.get_fe().n_dofs_per_face() == 0,
              dealii::ExcMessage("Expected a discontinuous element"));
  AssertThrow(cell_vectorization_category.size() ==
                dof_handler.get_triangulation().n_active_cells(),
              dealii::ExcDimensionMismatch(cell_vectorization_category.size(),
                                           dof_handler.get_triangulation().n_active_cells()));

  // locally owned cells in the order of the loop, the cells next to remote
  // cells first
  std::vector<std::tuple<bool, unsigned int, CellIterator>> cells;
  for(const auto & cell : dof_handler.active_cell_iterators())
    if(cell->is_locally_owned())
    {
      bool is_interior = true;
      for(unsigned int f = 0; f < 2 * dim; ++f)
      {
        const bool has_periodic = cell->at_boundary(f) && cell->has_periodic_neighbor(f);
        if((!cell->at_boundary(f) && !cell->neighbor(f)->is_locally_owned()) ||
           (has_periodic && !cell->periodic_neighbor(f)->is_locally_owned()))
          is_interior = false;
      }
      cells.emplace_back(is_interior, cell_vectorization_category[cell->active_cell_index()], cell);
    }
  std::sort(cells.begin(), cells.end(), [](const auto & a, const auto & b) {
    return std::get<0>(a) < std::get<0>(b) ||
           (std::get<0>(a) == std::get<0>(b) && std::get<1>(a) < std::get<1>(b));
  });

  std::vector<dealii::types::global_dof_index> dof_indices(dofs_per_cell);
  std::vector<dealii::types::global_dof_index> new_global(owned.n_elements(),
                                                          dealii::numbers::invalid_dof_index);
  unsigned int                                 counter = 0;
  for(const auto & entry : cells)
  {
    std::get<2>(entry)->get_dof_indices(dof_indices);
    for(unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      AssertThrow(dof_indices[i] == dof_indices[0] + i,
                  dealii::ExcMessage("The DoFs of each cell need to be contiguous"));
      new_global[owned.index_within_set(dof_indices[i])] = owned.nth_index_in_set(counter++);
    }
  }
  AssertThrow(counter == owned.n_elements(), dealii::ExcInternalError());
  dof_handler.renumber_dofs(new_global);
}

} // namespace Helper
