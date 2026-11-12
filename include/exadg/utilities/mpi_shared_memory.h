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

#ifndef EXADG_UTILITIES_MPI_SHARED_MEMORY_H_
#define EXADG_UTILITIES_MPI_SHARED_MEMORY_H_

#include <deal.II/base/memory_space_data.h>
#include <deal.II/base/mpi.h>

#include <memory>
#include <vector>

namespace mpi_shared_memory
{
/**
 * Allocate an array of `expected_size` entries in an MPI-3 shared-memory
 * window over `communicator_shared` (collective over that communicator).
 * `data.values` holds the local array, `data.values_sm[i]` a read view of the
 * array of rank `i` of `communicator_shared`. The window is freed when the
 * last copy of `data.values_sm_ptr` goes away, which is again collective.
 */
template<typename Number>
void
allocate_shared_recv_data(
  dealii::MemorySpace::MemorySpaceData<Number, dealii::MemorySpace::Host> & data,
  const unsigned int                                                        expected_size,
  const MPI_Comm                                                            communicator_shared)
{
  const unsigned int n_shared_ranks = dealii::Utilities::MPI::n_mpi_processes(communicator_shared);
  const unsigned int rank_shared    = dealii::Utilities::MPI::this_mpi_process(communicator_shared);

  MPI_Win  mpi_window;
  Number * data_this;

  std::vector<Number *> others(n_shared_ranks);

  MPI_Info info;
  int      ierr = MPI_Info_create(&info);
  AssertThrowMPI(ierr);

  ierr = MPI_Info_set(info, "alloc_shared_noncontig", "true");
  AssertThrowMPI(ierr);

  const std::size_t align_by = 64;

  std::size_t allocated_bytes =
    ((expected_size * sizeof(Number) + align_by) / sizeof(Number)) * sizeof(Number);

  ierr = MPI_Win_allocate_shared(
    allocated_bytes, sizeof(Number), info, communicator_shared, &data_this, &mpi_window);
  AssertThrowMPI(ierr);

  for(unsigned int i = 0; i < n_shared_ranks; ++i)
  {
    int        disp_unit;
    MPI_Aint   segment_size;
    const auto ierr = MPI_Win_shared_query(mpi_window, i, &segment_size, &disp_unit, &others[i]);
    AssertDimension(disp_unit, sizeof(Number));
    AssertThrowMPI(ierr);
  }

  Number * ptr_unaligned = others[rank_shared];
  Number * ptr_aligned   = ptr_unaligned;

  AssertThrow(std::align(align_by,
                         expected_size * sizeof(Number),
                         reinterpret_cast<void *&>(ptr_aligned),
                         allocated_bytes) != nullptr,
              dealii::ExcNotImplemented());

  unsigned int              n_align_local = ptr_aligned - ptr_unaligned;
  std::vector<unsigned int> n_align_sm(n_shared_ranks);

  ierr = MPI_Allgather(
    &n_align_local, 1, MPI_UNSIGNED, n_align_sm.data(), 1, MPI_UNSIGNED, communicator_shared);
  AssertThrowMPI(ierr);

  for(unsigned int i = 0; i < n_shared_ranks; ++i)
    others[i] += n_align_sm[i];

  std::vector<unsigned int> allocated_sizes(n_shared_ranks);

  ierr = MPI_Allgather(
    &expected_size, 1, MPI_UNSIGNED, allocated_sizes.data(), 1, MPI_UNSIGNED, communicator_shared);
  AssertThrowMPI(ierr);

  data.values_sm.resize(n_shared_ranks);
  for(unsigned int i = 0; i < n_shared_ranks; ++i)
    data.values_sm[i] = dealii::ArrayView<const Number>(others[i], allocated_sizes[i]);

  data.values = Kokkos::View<Number *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
    ptr_aligned, expected_size);

  // Kokkos will not free the memory because the memory is
  // unmanaged. Instead we use a shared pointer to take care of
  // that.
  data.values_sm_ptr = {ptr_aligned, [mpi_window](Number *) mutable {
                          // note: we are creating here a copy of
                          // the window other approaches led to
                          // segmentation faults
                          const auto ierr = MPI_Win_free(&mpi_window);
                          AssertThrowMPI(ierr);
                        }};
}
} // namespace mpi_shared_memory

#endif /* EXADG_UTILITIES_MPI_SHARED_MEMORY_H_ */
