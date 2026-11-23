/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2026 by the ExaDG authors
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
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#pragma once

// Orders of the cells in the loops of the RT operator, given as categories
// of MatrixFree::AdditionalData::cell_vectorization_category with
// non-strict categories: the cells are sorted by category and filled into
// the cell batches in that order.

#include <deal.II/distributed/tria.h>

#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/extruded_mapping_info.h>

#include <algorithm>
#include <array>
#include <string>
#include <vector>

namespace CellOrderings
{
using namespace dealii;

inline std::string
list_names()
{
  return "default, current, hierarchic, blocked<L>, surface<L> (L = levels of\n"
         "                          the blocks traversed lexicographically, current: 3 below 4096 cells\n"
         "                          per process, else 4)";
}

// Hierarchical order of the cells as given by the iteration over the
// active cells (the Morton order of p4est within each coarse cell)
template<int dim>
std::vector<unsigned int>
compute_hierarchic(const Triangulation<dim> & tria)
{
  std::vector<unsigned int> categories(tria.n_active_cells(), numbers::invalid_unsigned_int);
  unsigned int              counter = 0;
  for(const auto & cell : tria.active_cell_iterators())
    if(cell->is_locally_owned())
      categories[cell->active_cell_index()] = counter++;
  return categories;
}

// Variant of Helper::group_cells_by_left_neighbors() for the cells in a
// subset, with a general criterion whether the flux buffer cannot provide
// the data of the left face in direction d
template<int dim, typename InSet, typename LeftUnavailable>
void
group_cells(
  const std::array<int, dim> &                       lexicographic_index,
  const typename Triangulation<dim>::cell_iterator & cell,
  std::array<
    std::vector<std::pair<std::array<int, dim>, typename Triangulation<dim>::active_cell_iterator>>,
    8> &                  cells_per_neighbor_value,
  const InSet &           in_set,
  const LeftUnavailable & left_unavailable)
{
  if(cell->is_active())
  {
    if(cell->is_locally_owned() && in_set(cell))
    {
      unsigned int type = 0;
      for(unsigned int d = 0; d < dim; ++d)
        if(left_unavailable(cell, d))
          type += 1U << d;
      cells_per_neighbor_value[type].emplace_back(lexicographic_index, cell);
    }
  }
  else
    for(unsigned int child = 0; child < cell->n_children(); ++child)
    {
      std::array<int, dim> child_lexicographic;
      for(unsigned int d = 0; d < dim; ++d)
        child_lexicographic[dim - 1 - d] =
          lexicographic_index[dim - 1 - d] * 2 + ((child >> d) & 1);
      group_cells<dim>(child_lexicographic,
                       cell->child(child),
                       cells_per_neighbor_value,
                       in_set,
                       left_unavailable);
    }
}

// Generalization of Helper::compute_vectorization_category() with the
// number of levels of the blocks as a parameter (4 there; the blocks
// traversed in lexicographic order are the cells on level n_levels -
// block_levels), restricted to the cells in a subset, continuing the
// numbering at 'category_counter'
template<int dim, typename InSet, typename LeftUnavailable>
void
compute_blocked(const Triangulation<dim> &  tria,
                const unsigned int          block_levels,
                const InSet &               in_set,
                const LeftUnavailable &     left_unavailable,
                unsigned int &              category_counter,
                std::vector<unsigned int> & cell_vectorization_category)
{
  using CellLexEntry =
    std::pair<std::array<int, dim>, typename Triangulation<dim>::active_cell_iterator>;
  std::array<std::vector<CellLexEntry>, 8> cells_per_neighbor_value;

  unsigned int blocking_level = 0;
  if(tria.n_levels() > block_levels)
    blocking_level = tria.n_levels() - block_levels;

  constexpr unsigned int n_lanes = VectorizedArray<float>::size();
  for(const auto & cell : tria.cell_iterators_on_level(blocking_level))
  {
    for(auto & local_vector : cells_per_neighbor_value)
      local_vector.clear();
    group_cells<dim>(
      std::array<int, dim>{}, cell, cells_per_neighbor_value, in_set, left_unavailable);
    for(auto & local_vector : cells_per_neighbor_value)
      std::sort(local_vector.begin(), local_vector.end());

    for(unsigned int neighbor_id = 7; neighbor_id > 2; --neighbor_id)
      for(auto & entry : cells_per_neighbor_value[neighbor_id])
        cell_vectorization_category[entry.second->active_cell_index()] = category_counter++;

    std::array<typename std::vector<CellLexEntry>::iterator, 3> lex_ptrs{
      {cells_per_neighbor_value[0].begin(),
       cells_per_neighbor_value[1].begin(),
       cells_per_neighbor_value[2].begin()}};
    const std::array<typename std::vector<CellLexEntry>::iterator, 3> lex_ends{
      {cells_per_neighbor_value[0].end(),
       cells_per_neighbor_value[1].end(),
       cells_per_neighbor_value[2].end()}};
    auto assign_category = [&](auto & it) {
      cell_vectorization_category[it->second->active_cell_index()] = category_counter++;
      ++it;
    };
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
    // cells of types 1 and 2 left when there are no more cells of type 0
    for(unsigned int i = 1; i < 3; ++i)
      while(lex_ptrs[i] != lex_ends[i])
        assign_category(lex_ptrs[i]);
  }
}

template<int dim>
typename Triangulation<dim>::cell_iterator
get_neighbor(const typename Triangulation<dim>::cell_iterator & cell, const unsigned int f)
{
  if(cell->at_boundary(f) && cell->has_periodic_neighbor(f))
    return cell->periodic_neighbor(f);
  else if(!cell->at_boundary(f))
    return cell->neighbor(f);
  else
    return typename Triangulation<dim>::cell_iterator();
}

// Cells with a face to a cell of another process, whose data is read by
// other processes and therefore passes through 'before_loop' ahead of the
// cell loop
template<int dim>
bool
is_surface(const typename Triangulation<dim>::cell_iterator & cell)
{
  for(unsigned int f = 0; f < 2 * dim; ++f)
  {
    const auto neighbor = get_neighbor<dim>(cell, f);
    if(neighbor.state() == IteratorState::valid && !neighbor->is_locally_owned())
      return true;
  }
  return false;
}

// Criterion of the RT operator for the flux buffer of the left face
template<int dim>
bool
left_at_boundary_or_remote(const typename Triangulation<dim>::cell_iterator & cell,
                           const unsigned int                                 d)
{
  return cell->at_boundary(2 * d) || cell->neighbor(2 * d)->subdomain_id() != cell->subdomain_id();
}

template<int dim>
std::vector<unsigned int>
compute_categories(const std::string & name, const Triangulation<dim> & tria)
{
  if(name == "default")
    return {};
  else if(name == "current")
    return Helper::compute_vectorization_category(tria);
  else if(name == "hierarchic")
    return compute_hierarchic(tria);
  std::vector<unsigned int> categories(tria.n_active_cells(), numbers::invalid_unsigned_int);
  unsigned int              counter = 0;
  using CellIterator                = typename Triangulation<dim>::cell_iterator;
  if(name.rfind("blocked", 0) == 0)
    compute_blocked(
      tria,
      std::stoi(name.substr(7)),
      [](const CellIterator &) { return true; },
      left_at_boundary_or_remote<dim>,
      counter,
      categories);
  else if(name.rfind("surface", 0) == 0)
  {
    // first the cells at the surface to other processes, where the cells
    // of the interior are not yet available for the flux buffer, then the
    // interior cells
    const unsigned int levels = std::stoi(name.substr(7));
    compute_blocked(
      tria,
      levels,
      is_surface<dim>,
      [](const CellIterator & cell, const unsigned int d) {
        return left_at_boundary_or_remote<dim>(cell, d) || !is_surface<dim>(cell->neighbor(2 * d));
      },
      counter,
      categories);
    compute_blocked(
      tria,
      levels,
      [](const CellIterator & cell) { return !is_surface<dim>(cell); },
      left_at_boundary_or_remote<dim>,
      counter,
      categories);
  }
  else
    AssertThrow(false, ExcMessage("Unknown order of cells " + name));

  unsigned int n_owned = 0;
  for(const auto & cell : tria.active_cell_iterators())
    if(cell->is_locally_owned())
      ++n_owned;
  AssertThrow(counter == n_owned, ExcMessage("Not all cells assigned by " + name));
  return categories;
}

} // namespace CellOrderings
