/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2026 by Martin Kronbichler, Shubham Goswami,
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

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <queue>
#include <type_traits>
#include <vector>

namespace ExaDG
{
/**
 * Infrastructure for a matrix-vector product that accumulates the result in
 * a small scratch array instead of a global vector, for use in operators
 * with a cell loop over cell batches. The result is consumed by a function
 * 'after_loop(begin, end, values)' once all contributions to the vector
 * entries [begin, end) have been collected, where 'values' points to the
 * result for entry 'begin'. Both begin and end are multiples of chunk_size
 * except for the end of the locally owned range, and the pointer has the
 * same alignment as the vector entry 'begin', i.e., 64-byte aligned for
 * entries at multiples of chunk_size.
 *
 * The operator describes the writes of the cell loop by entities, i.e.,
 * contiguous ranges of vector entries written by a cell batch as a whole
 * (e.g. the vertices, lines, quads and hexes of continuous elements, the
 * faces and interior of Raviart-Thomas elements, or the cells of DG
 * elements), identified by their first index. In the cell loop, the operator
 * writes to the scratch array with the entity indices translated by
 * scratch_index(). Entries of the locally owned range that are not part of
 * any entity are zero in the result.
 *
 * The scratch array is organized as follows. An entity spanning more than
 * one chunk gets a slot in a pool that is only alive between the first and
 * the last write into the entity, so the pool stays small. Within a slot,
 * the entries are placed with the same alignment with respect to chunk_size
 * as in the vector. Chunks whose entities finish in the same cell batch and
 * that are contiguous in the pool are read directly from the slots of the
 * entities. All other chunks, including those with entities contained in a
 * single chunk, get a staging slot of one chunk: the entities contained in
 * the chunk are written there directly, while the other entities copy their
 * part of the chunk after their last write. The chunks receiving
 * contributions from other processes in the compress step (and the chunks
 * of entities overlapping with those) get a slot for the whole loop after
 * the pool, followed by the ghost range of the vector.
 */
template<typename Number>
class ScratchMatrixVectorProduct
{
public:
  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

  // Granularity of the ranges passed to the function 'after_loop'. As the
  // vectors are 64-byte aligned, all accesses to the vectors and the scratch
  // array in the vector updates are aligned for float and double with up to
  // AVX-512.
  static constexpr unsigned int chunk_size = 16;

  // Function adding an entity with the given first index and length,
  // called for all entities written by a cell batch
  using EntityFunction = std::function<void(const unsigned int, const unsigned int)>;

  // Minimal interface of a vector on the scratch array as used by
  // functions writing into vectors via local_element() and begin()
  struct View
  {
    using value_type = Number;

    Number &
    local_element(const unsigned int index) const
    {
      return data[index];
    }

    Number *
    begin() const
    {
      return data;
    }

    Number * data;
  };

  /**
   * Compute the layout of the scratch array. The function
   * 'for_each_entity(cell, add)' calls 'add(index, length)' for all
   * entities written by the cell batch 'cell', with locally owned or ghost
   * indices of the vector described by 'partitioner'; invalid indices are
   * skipped. If 'compress_ghosts' is true, the contributions to ghost
   * entities are sent to the owner in a compress step after the cell
   * loop. Otherwise, no ghost entities may be written.
   */
  void
  reinit(const std::shared_ptr<const dealii::Utilities::MPI::Partitioner> &      partitioner,
         const unsigned int                                                      n_cell_batches,
         const std::function<void(const unsigned int, const EntityFunction &)> & for_each_entity,
         const bool                                                              compress_ghosts);

  // Index into the scratch array for an entity given by its first index in
  // the vector, available between reinit() and release_setup_data(). Returns
  // an invalid index for indices that are not the start of an entity.
  unsigned int
  scratch_index(const unsigned int index) const
  {
    if(index == dealii::numbers::invalid_unsigned_int)
      return index;
    else if(index < n_owned)
    {
      AssertIndexRange(index, entity_offset.size());
      return entity_offset[index];
    }
    else
      return layout.ghost_start + (index - n_owned);
  }

  // Release the data only needed for scratch_index()
  void
  release_setup_data()
  {
    entity_offset.clear();
    entity_offset.shrink_to_fit();
  }

  // Zero the parts of the scratch array alive throughout the loop, to be
  // called before cell_loop(), e.g. while the ghost exchange is in flight
  void
  prepare_loop() const
  {
    Number * scratch = scratch_data.data();
    Number * mpi_dst = compress_vector.begin();
    for(const auto & [dof, offset, length] : layout.mpi_list)
    {
      std::memset(scratch + offset, 0, length * sizeof(Number));
      if(compress_ghosts)
        std::memset(mpi_dst + dof, 0, length * sizeof(Number));
    }
    std::memset(scratch + layout.ghost_start,
                0,
                (layout.size - layout.ghost_start) * sizeof(Number));
  }

  /**
   * Run through the cell batches in order, calling
   * 'cell_operation(cell, scratch)' that adds the contributions of the cell
   * batch into the scratch array at the positions given by scratch_index(),
   * and pass the finished ranges to 'after_loop'. If given, the function
   * 'gap_operation(begin, end, values)' sets the result for the ranges of
   * entries not part of any entity, which are zero otherwise.
   */
  template<typename CellOperation, typename AfterLoop, typename GapOperation = std::nullptr_t>
  void
  cell_loop(const CellOperation & cell_operation,
            const AfterLoop &     after_loop,
            const GapOperation &  gap_operation = nullptr) const
  {
    Number *           scratch        = scratch_data.data();
    const unsigned int n_cell_batches = layout.zero_list_index.size() - 1;
    for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
    {
      for(unsigned int r = layout.zero_list_index[cell]; r < layout.zero_list_index[cell + 1]; ++r)
        std::memset(scratch + layout.zero_list[r].first,
                    0,
                    layout.zero_list[r].second * sizeof(Number));

      cell_operation(cell, scratch);

      for(unsigned int r = layout.copy_list_index[cell]; r < layout.copy_list_index[cell + 1]; ++r)
      {
        const auto & [source, destination, length] = layout.copy_list[r];
        std::copy(scratch + source, scratch + source + length, scratch + destination);
      }

      if constexpr(!std::is_same_v<GapOperation, std::nullptr_t>)
        for(unsigned int r = layout.gap_list_index[cell]; r < layout.gap_list_index[cell + 1]; ++r)
        {
          const auto & [dof, offset, length] = layout.gap_list[r];
          gap_operation(dof, dof + length, scratch + offset);
        }

      for(unsigned int r = layout.finalize_list_index[cell];
          r < layout.finalize_list_index[cell + 1];
          ++r)
      {
        const auto & [dof, offset, length] = layout.finalize_list[r];
        after_loop(dof, dof + length, scratch + offset);
      }
    }
  }

  /**
   * Send the contributions to ghost entries to their owners, if enabled,
   * and pass the remaining ranges to 'after_loop'. The function
   * 'work_during_compress' is run while the messages are in flight. The
   * function 'gap_operation' is the same as for cell_loop().
   */
  template<typename AfterLoop, typename GapOperation = std::nullptr_t>
  void
  finish_loop(const AfterLoop &             after_loop,
              const std::function<void()> & work_during_compress,
              const GapOperation &          gap_operation = nullptr) const
  {
    Number * scratch = scratch_data.data();
    if(compress_ghosts)
    {
      Number * mpi_dst = compress_vector.begin();
      std::copy(scratch + layout.ghost_start, scratch + layout.size, mpi_dst + n_owned);
      compress_vector.compress_start(0, dealii::VectorOperation::add);
      if(work_during_compress)
        work_during_compress();
      compress_vector.compress_finish(dealii::VectorOperation::add);

      // add the local contributions to the ones received from the other
      // processes
      for(const auto & [dof, offset, length] : layout.mpi_list)
        for(unsigned int i = 0; i < length; ++i)
          mpi_dst[dof + i] += scratch[offset + i];
      if constexpr(!std::is_same_v<GapOperation, std::nullptr_t>)
        for(const auto & [dof, offset, length] : layout.mpi_gap_list)
          gap_operation(dof, dof + length, mpi_dst + dof);
      for(const auto & [dof, offset, length] : layout.mpi_list)
        after_loop(dof, dof + length, mpi_dst + dof);
    }
    else
    {
      if(work_during_compress)
        work_during_compress();
      if constexpr(!std::is_same_v<GapOperation, std::nullptr_t>)
        for(const auto & [dof, offset, length] : layout.mpi_gap_list)
          gap_operation(dof, dof + length, scratch + offset);
      for(const auto & [dof, offset, length] : layout.mpi_list)
        after_loop(dof, dof + length, scratch + offset);
    }
  }

private:
  // Layout of the scratch array, see the class description
  struct Layout
  {
    // ranges [first, first + second) of the scratch array to be zeroed
    // before the cell batch with the given index
    std::vector<unsigned int>                          zero_list_index;
    std::vector<std::pair<unsigned int, unsigned int>> zero_list;

    // ranges of entries in the slots of entities copied into the staging
    // slots of chunks after the cell batch with the given index, stored as
    // source offset, destination offset and length
    std::vector<unsigned int>                copy_list_index;
    std::vector<std::array<unsigned int, 3>> copy_list;

    // ranges of entries finalized after the cell batch with the given
    // index, stored as dof index, scratch offset and length
    std::vector<unsigned int>                finalize_list_index;
    std::vector<std::array<unsigned int, 3>> finalize_list;

    // ranges of consecutive chunks in the part of the scratch array alive
    // throughout the loop, finalized after the loop, stored as dof index,
    // scratch offset and length
    std::vector<std::array<unsigned int, 3>> mpi_list;

    // ranges of entries not part of any entity, set after the cell batch
    // with the given index or after the loop, stored as dof index, scratch
    // offset and length
    std::vector<unsigned int>                gap_list_index;
    std::vector<std::array<unsigned int, 3>> gap_list;
    std::vector<std::array<unsigned int, 3>> mpi_gap_list;

    unsigned int pool_size   = 0;
    unsigned int mpi_size    = 0;
    unsigned int ghost_start = 0;
    unsigned int size        = 0;
  };

  unsigned int                          n_owned         = 0;
  bool                                  compress_ghosts = false;
  Layout                                layout;
  std::vector<unsigned int>             entity_offset;
  mutable dealii::AlignedVector<Number> scratch_data;
  mutable VectorType                    compress_vector;
};



template<typename Number>
void
ScratchMatrixVectorProduct<Number>::reinit(
  const std::shared_ptr<const dealii::Utilities::MPI::Partitioner> &      partitioner,
  const unsigned int                                                      n_cell_batches,
  const std::function<void(const unsigned int, const EntityFunction &)> & for_each_entity,
  const bool                                                              compress_ghosts)
{
  using dealii::ExcInternalError;
  using dealii::ExcMessage;
  constexpr unsigned int chunk   = chunk_size;
  constexpr unsigned int invalid = dealii::numbers::invalid_unsigned_int;

  this->compress_ghosts       = compress_ghosts;
  n_owned                     = partitioner->locally_owned_size();
  const unsigned int n_ghost  = compress_ghosts ? partitioner->n_ghost_indices() : 0;
  const unsigned int n_chunks = (n_owned + chunk - 1) / chunk;

  // first and last write of each entity, indexed by its first index
  std::vector<unsigned int> first_write(n_owned, invalid), last_write(n_owned, invalid);
  std::vector<unsigned int> entity_length(n_owned, 0);
  std::vector<unsigned int> entities;
  for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
    for_each_entity(cell, [&](const unsigned int idx, const unsigned int length) {
      if(idx == invalid)
        return;
      if(idx >= n_owned)
      {
        AssertThrow(compress_ghosts, ExcMessage("Write into ghost entity without compress"));
        return;
      }
      if(first_write[idx] == invalid)
      {
        AssertThrow(length > 0 && idx + length <= n_owned, ExcInternalError());
        first_write[idx]   = cell;
        entity_length[idx] = length;
        entities.push_back(idx);
      }
      AssertThrow(entity_length[idx] == length, ExcMessage("Inconsistent entity length"));
      last_write[idx] = cell;
    });

  // Merge entities adjacent in the vector with the same first and last
  // write into a single entity, e.g. the cells of a cell batch for DG
  // elements or the interiors of the cells of a cell batch for continuous
  // elements. This avoids splitting the chunks shared by those entities
  // between different slots.
  std::vector<std::pair<unsigned int, unsigned int>> merged_entities;
  std::vector<unsigned int>                          entities_by_index(entities);
  std::sort(entities_by_index.begin(), entities_by_index.end());
  {
    unsigned int parent = invalid;
    for(const unsigned int idx : entities_by_index)
      if(parent != invalid && parent + entity_length[parent] == idx &&
         first_write[parent] == first_write[idx] && last_write[parent] == last_write[idx])
      {
        entity_length[parent] += entity_length[idx];
        merged_entities.emplace_back(idx, parent);
      }
      else
        parent = idx;
    for(const auto & [idx, parent] : merged_entities)
      first_write[idx] = invalid;
    const auto is_merged = [&](const unsigned int idx) { return first_write[idx] == invalid; };
    entities.erase(std::remove_if(entities.begin(), entities.end(), is_merged), entities.end());
    entities_by_index.erase(std::remove_if(entities_by_index.begin(),
                                           entities_by_index.end(),
                                           is_merged),
                            entities_by_index.end());
  }

  // entities overlapping with each chunk, in the order of the vector, and
  // entries not covered by any entity
  const auto first_chunk = [&](const unsigned int idx) { return idx / chunk; };
  const auto last_chunk  = [&](const unsigned int idx) {
    return (idx + entity_length[idx] - 1) / chunk;
  };
  const auto is_contained = [&](const unsigned int idx) {
    return first_chunk(idx) == last_chunk(idx);
  };
  std::vector<unsigned int> chunk_entities_index(n_chunks + 1, 0);
  for(const unsigned int idx : entities_by_index)
    for(unsigned int c = first_chunk(idx); c <= last_chunk(idx); ++c)
      ++chunk_entities_index[c + 1];
  for(unsigned int c = 0; c < n_chunks; ++c)
    chunk_entities_index[c + 1] += chunk_entities_index[c];
  std::vector<unsigned int> chunk_entities(chunk_entities_index.back());
  {
    std::vector<unsigned int> position(chunk_entities_index.begin(),
                                       chunk_entities_index.end() - 1);
    for(const unsigned int idx : entities_by_index)
      for(unsigned int c = first_chunk(idx); c <= last_chunk(idx); ++c)
        chunk_entities[position[c]++] = idx;
  }
  // ranges of entries not covered by any entity, split at the chunks
  std::vector<unsigned char>                         chunk_has_gap(n_chunks, 0);
  std::vector<std::pair<unsigned int, unsigned int>> gaps;
  {
    const auto add_gap = [&](const unsigned int begin, const unsigned int end) {
      for(unsigned int i = begin; i < end; i = std::min(end, (i / chunk + 1) * chunk))
      {
        chunk_has_gap[i / chunk] = 1;
        gaps.emplace_back(i, std::min(end, (i / chunk + 1) * chunk));
      }
    };
    unsigned int covered_until = 0;
    for(const unsigned int idx : entities_by_index)
    {
      AssertThrow(idx >= covered_until, ExcMessage("Overlapping entities"));
      add_gap(covered_until, idx);
      covered_until = idx + entity_length[idx];
    }
    add_gap(covered_until, n_owned);
  }

  // Chunks with entries receiving contributions from other processes are
  // finalized after the compress step, as are chunks without any entity.
  // Entities overlapping with those chunks get a slot in the part of the
  // scratch array alive throughout the loop, so all other chunks
  // overlapping with those entities are also finalized after the
  // loop. Repeat until no more chunks are added.
  std::vector<unsigned char> chunk_is_mpi(n_chunks, 0), is_mpi(n_owned, 0);
  if(compress_ghosts)
    for(const auto & range : partitioner->import_indices())
      for(unsigned int i = range.first; i < range.second; ++i)
        chunk_is_mpi[i / chunk] = 1;
  for(unsigned int c = 0; c < n_chunks; ++c)
    if(chunk_entities_index[c + 1] == chunk_entities_index[c])
      chunk_is_mpi[c] = 1;
  for(bool changed = true; changed;)
  {
    changed = false;
    for(const unsigned int idx : entities_by_index)
    {
      bool overlaps_mpi = false;
      for(unsigned int c = first_chunk(idx); c <= last_chunk(idx); ++c)
        overlaps_mpi |= chunk_is_mpi[c];
      if(overlaps_mpi && !is_mpi[idx])
      {
        is_mpi[idx] = 1;
        for(unsigned int c = first_chunk(idx); c <= last_chunk(idx); ++c)
          if(!chunk_is_mpi[c])
          {
            chunk_is_mpi[c] = 1;
            changed         = true;
          }
      }
    }
  }

  // Time of finalization of each chunk, and the time at which a staging
  // slot of the chunk is needed: the first write into an entity contained
  // in the chunk, or the first last write of the entities spanning several
  // chunks, which copy their part into the staging slot after their last
  // write. Chunks with entities finishing in different cell batches, with
  // contained entities or with entries not covered by any entity need
  // staging.
  std::vector<unsigned int>  chunk_finalize(n_chunks, 0), chunk_staging_start(n_chunks, invalid);
  std::vector<unsigned char> needs_staging(n_chunks, 0);
  for(unsigned int c = 0; c < n_chunks; ++c)
    if(!chunk_is_mpi[c])
    {
      unsigned int first_done = invalid;
      bool         contained  = false;
      for(unsigned int j = chunk_entities_index[c]; j < chunk_entities_index[c + 1]; ++j)
      {
        const unsigned int idx = chunk_entities[j];
        chunk_finalize[c]      = std::max(chunk_finalize[c], last_write[idx]);
        if(is_contained(idx))
        {
          contained              = true;
          chunk_staging_start[c] = std::min(chunk_staging_start[c], first_write[idx]);
        }
        else
        {
          first_done             = std::min(first_done, last_write[idx]);
          chunk_staging_start[c] = std::min(chunk_staging_start[c], last_write[idx]);
        }
      }
      needs_staging[c] = contained || chunk_has_gap[c] || first_done != chunk_finalize[c];
    }

  // number of blocks of a slot that keeps the alignment of the entity
  const auto n_blocks = [&](const unsigned int idx) {
    return (idx % chunk + entity_length[idx] + chunk - 1) / chunk;
  };

  // Allocate the slots of the pool in the order of the cell loop, where the
  // entities are already sorted by their first write. We keep a list of
  // free ranges and pick the first one large enough. This tends to keep the
  // slots of entities processed together contiguous and reuses the slots of
  // the previous cell batches that are likely still in cache. Whether a
  // chunk without contained entities needs staging depends on the placement
  // of the entities, so we first place the entities assuming that only the
  // chunks identified above need staging,
  // and then add staging for the chunks found to be non-contiguous.
  entity_offset.assign(n_owned, invalid);
  std::vector<unsigned int> staging(n_chunks, invalid);
  const auto                allocate = [&]() {
    std::vector<std::vector<unsigned int>> staging_by_cell(n_cell_batches);
    for(unsigned int c = 0; c < n_chunks; ++c)
      if(needs_staging[c])
        staging_by_cell[chunk_staging_start[c]].push_back(c);

    // release queue of (time, start, length) and free ranges of the pool
    // as map from start to length, with adjacent free ranges coalesced
    std::priority_queue<std::array<unsigned int, 3>,
                        std::vector<std::array<unsigned int, 3>>,
                        std::greater<>>
                                         release_queue;
    std::map<unsigned int, unsigned int> free_ranges;
    unsigned int                         pool_size = 0;
    const auto release = [&](const unsigned int start, const unsigned int length) {
      auto next = free_ranges.emplace(start, length).first;
      if(next != free_ranges.begin())
      {
        auto previous = std::prev(next);
        if(previous->first + previous->second == start)
        {
          previous->second += next->second;
          free_ranges.erase(next);
          next = previous;
        }
      }
      auto after = std::next(next);
      if(after != free_ranges.end() && next->first + next->second == after->first)
      {
        next->second += after->second;
        free_ranges.erase(after);
      }
    };
    // first fit at the lowest offset, growing the pool at the end if no
    // free range is large enough
    const auto get_slot = [&](const unsigned int cell, const unsigned int blocks) {
      while(!release_queue.empty() && release_queue.top()[0] < cell)
      {
        release(release_queue.top()[1], release_queue.top()[2]);
        release_queue.pop();
      }
      const unsigned int length = blocks * chunk;
      for(auto range = free_ranges.begin(); range != free_ranges.end(); ++range)
        if(range->second >= length || range->first + range->second == pool_size)
        {
          const unsigned int start = range->first;
          if(range->second < length)
            pool_size = start + length;
          else if(range->second > length)
            free_ranges.emplace(start + length, range->second - length);
          free_ranges.erase(range);
          return start;
        }
      pool_size += length;
      return pool_size - length;
    };

    auto next_entity = entities.begin();
    for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
    {
      for(; next_entity != entities.end() && first_write[*next_entity] == cell; ++next_entity)
        if(!is_mpi[*next_entity] && !is_contained(*next_entity))
        {
          const unsigned int idx   = *next_entity;
          const unsigned int start = get_slot(cell, n_blocks(idx));
          entity_offset[idx]       = start + idx % chunk;
          release_queue.push({{last_write[idx], start, n_blocks(idx) * chunk}});
        }
      for(const unsigned int c : staging_by_cell[cell])
      {
        staging[c] = get_slot(cell, 1);
        release_queue.push({{chunk_finalize[c], staging[c], chunk}});
      }
    }
    return pool_size;
  };
  // As additional staging slots can change the placement of the entities
  // allocated later, repeat until all chunks without staging are
  // contiguous; the set of chunks with staging only grows, so this
  // terminates.
  const auto is_contiguous = [&](const unsigned int c) {
    bool contiguous = true;
    for(unsigned int j = chunk_entities_index[c] + 1; j < chunk_entities_index[c + 1]; ++j)
      contiguous &= (entity_offset[chunk_entities[j]] - chunk_entities[j] ==
                     entity_offset[chunk_entities[j - 1]] - chunk_entities[j - 1]);
    return contiguous;
  };
  layout = Layout();
  for(bool added_staging = true; added_staging;)
  {
    layout.pool_size = allocate();
    added_staging    = false;
    for(unsigned int c = 0; c < n_chunks; ++c)
      if(!chunk_is_mpi[c] && !needs_staging[c] && !is_contiguous(c))
      {
        needs_staging[c]       = 1;
        chunk_staging_start[c] = chunk_finalize[c];
        for(unsigned int j = chunk_entities_index[c]; j < chunk_entities_index[c + 1]; ++j)
          chunk_staging_start[c] = std::min(chunk_staging_start[c], last_write[chunk_entities[j]]);
        added_staging = true;
      }
  }

  // entities contained in a chunk are placed in its staging slot
  for(const unsigned int idx : entities)
    if(!is_mpi[idx] && is_contained(idx))
      entity_offset[idx] = staging[first_chunk(idx)] + idx % chunk;

  // MPI chunks after the pool in the order of the vector, so that the
  // entities overlapping with several of them are contiguous
  std::vector<unsigned int> mpi_offset(n_chunks, invalid);
  unsigned int              position = layout.pool_size;
  for(unsigned int c = 0; c < n_chunks; ++c)
    if(chunk_is_mpi[c])
    {
      mpi_offset[c]            = position;
      const unsigned int begin = c * chunk;
      const unsigned int end   = std::min(begin + chunk, n_owned);
      auto &             list  = layout.mpi_list;
      if(!list.empty() && list.back()[0] + list.back()[2] == begin)
        list.back()[2] += end - begin;
      else
        list.push_back({{begin, position, end - begin}});
      position += chunk;
    }
  for(const unsigned int idx : entities)
    if(is_mpi[idx])
      entity_offset[idx] = mpi_offset[first_chunk(idx)] + idx % chunk;
  for(const auto & [idx, parent] : merged_entities)
    entity_offset[idx] = entity_offset[parent] + (idx - parent);
  layout.mpi_size    = position - layout.pool_size;
  layout.ghost_start = position;
  layout.size        = position + n_ghost;
  AssertThrow(layout.size < invalid, ExcMessage("Scratch array too large"));

  // ranges to zero before the cell batches, merging adjacent ranges
  std::vector<std::vector<std::pair<unsigned int, unsigned int>>> zero_ranges(n_cell_batches);
  for(const unsigned int idx : entities)
    if(!is_mpi[idx] && !is_contained(idx))
      zero_ranges[first_write[idx]].emplace_back(entity_offset[idx], entity_length[idx]);
  for(unsigned int c = 0; c < n_chunks; ++c)
    if(needs_staging[c])
      zero_ranges[chunk_staging_start[c]].emplace_back(staging[c], chunk);
  layout.zero_list_index.resize(n_cell_batches + 1);
  for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
  {
    layout.zero_list_index[cell] = layout.zero_list.size();
    std::sort(zero_ranges[cell].begin(), zero_ranges[cell].end());
    for(const auto & range : zero_ranges[cell])
      if(layout.zero_list.size() > layout.zero_list_index[cell] &&
         layout.zero_list.back().first + layout.zero_list.back().second >= range.first)
        layout.zero_list.back().second =
          std::max(layout.zero_list.back().first + layout.zero_list.back().second,
                   range.first + range.second) -
          layout.zero_list.back().first;
      else
        layout.zero_list.push_back(range);
  }
  layout.zero_list_index[n_cell_batches] = layout.zero_list.size();

  // copies into the staging slots after the cell batches, merged when
  // contiguous in both places
  std::vector<std::vector<std::array<unsigned int, 3>>> copies(n_cell_batches);
  for(unsigned int c = 0; c < n_chunks; ++c)
    if(needs_staging[c])
      for(unsigned int j = chunk_entities_index[c]; j < chunk_entities_index[c + 1]; ++j)
      {
        const unsigned int idx = chunk_entities[j];
        if(is_contained(idx))
          continue;
        const unsigned int piece_begin = std::max(c * chunk, idx);
        const unsigned int piece_end   = std::min((c + 1) * chunk, idx + entity_length[idx]);
        copies[last_write[idx]].push_back({{entity_offset[idx] + (piece_begin - idx),
                                            staging[c] + (piece_begin - c * chunk),
                                            piece_end - piece_begin}});
      }
  layout.copy_list_index.resize(n_cell_batches + 1);
  for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
  {
    layout.copy_list_index[cell] = layout.copy_list.size();
    std::sort(copies[cell].begin(), copies[cell].end());
    for(const auto & copy : copies[cell])
      if(layout.copy_list.size() > layout.copy_list_index[cell] &&
         layout.copy_list.back()[0] + layout.copy_list.back()[2] == copy[0] &&
         layout.copy_list.back()[1] + layout.copy_list.back()[2] == copy[1])
        layout.copy_list.back()[2] += copy[2];
      else
        layout.copy_list.push_back(copy);
  }
  layout.copy_list_index[n_cell_batches] = layout.copy_list.size();

  // ranges to finalize after the cell batches, merging consecutive chunks
  // that are contiguous in the scratch array
  std::vector<std::vector<unsigned int>> finalize_chunks(n_cell_batches);
  for(unsigned int c = 0; c < n_chunks; ++c)
    if(!chunk_is_mpi[c])
      finalize_chunks[chunk_finalize[c]].push_back(c);
  layout.finalize_list_index.resize(n_cell_batches + 1);
  for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
  {
    layout.finalize_list_index[cell] = layout.finalize_list.size();
    for(const unsigned int c : finalize_chunks[cell])
    {
      const unsigned int begin = c * chunk;
      const unsigned int end   = std::min(begin + chunk, n_owned);
      const unsigned int idx   = chunk_entities[chunk_entities_index[c]];
      const unsigned int scratch_offset =
        needs_staging[c] ? staging[c] : entity_offset[idx] + (begin - idx);
      auto & list = layout.finalize_list;
      if(list.size() > layout.finalize_list_index[cell] &&
         list.back()[0] + list.back()[2] == begin &&
         list.back()[1] + list.back()[2] == scratch_offset)
        list.back()[2] += end - begin;
      else
        list.push_back({{begin, scratch_offset, end - begin}});
    }
  }
  layout.finalize_list_index[n_cell_batches] = layout.finalize_list.size();

  // gaps, set right before the finalization of their chunk
  std::vector<std::vector<std::array<unsigned int, 3>>> gaps_by_cell(n_cell_batches);
  for(const auto & [begin, end] : gaps)
  {
    const unsigned int c = begin / chunk;
    if(chunk_is_mpi[c])
      layout.mpi_gap_list.push_back({{begin, mpi_offset[c] + begin % chunk, end - begin}});
    else
      gaps_by_cell[chunk_finalize[c]].push_back({{begin, staging[c] + begin % chunk, end - begin}});
  }
  layout.gap_list_index.resize(n_cell_batches + 1);
  for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
  {
    layout.gap_list_index[cell] = layout.gap_list.size();
    layout.gap_list.insert(layout.gap_list.end(),
                           gaps_by_cell[cell].begin(),
                           gaps_by_cell[cell].end());
  }
  layout.gap_list_index[n_cell_batches] = layout.gap_list.size();

  scratch_data.resize_fast(layout.size);
  if(compress_ghosts)
    compress_vector.reinit(partitioner);
  else
    compress_vector.reinit(0);
}

} // namespace ExaDG
