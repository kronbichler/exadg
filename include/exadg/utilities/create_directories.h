/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
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

#ifndef EXADG_UTILITIES_CREATE_DIRECTORIES_H_
#define EXADG_UTILITIES_CREATE_DIRECTORIES_H_

// C/C++
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <optional>

// deal.II
#include <deal.II/base/mpi.h>

namespace ExaDG
{
/**
 * Creates directories if not already existing. An MPI barrier ensures that
 * directories have been created for all processes when completing this function.
 */
inline void
create_directories(std::string const & directory, MPI_Comm const & mpi_comm)
{
  if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    std::filesystem::create_directories(directory);

  MPI_Barrier(mpi_comm);
}

/**
 * Scans `base_directory` for immediate subdirectories whose name consists of
 * `prefix` followed by a non-negative integer (e.g. "case_span_3" for
 * `prefix = "case_span_"`), and returns the largest such integer found. Returns
 * an empty optional if `base_directory` does not exist or no matching
 * subdirectory is found. The scan is performed on MPI rank 0 only and the
 * result is broadcast to all ranks, so that all processes agree on the result
 * even though the filesystem is only queried once.
 */
inline std::optional<unsigned int>
find_last_indexed_subdirectory(std::string const & base_directory,
                               std::string const & prefix,
                               MPI_Comm const &    mpi_comm)
{
  int last_index = -1; // -1 encodes "not found" for the broadcast below

  if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
  {
    if(std::filesystem::exists(base_directory))
    {
      for(auto const & entry : std::filesystem::directory_iterator(base_directory))
      {
        if(not entry.is_directory())
          continue;

        std::string const name = entry.path().filename().string();
        if(name.size() <= prefix.size() or name.compare(0, prefix.size(), prefix) != 0)
          continue;

        std::string const suffix = name.substr(prefix.size());
        if(suffix.empty() or not std::all_of(suffix.begin(), suffix.end(), [](unsigned char c) {
             return std::isdigit(c) != 0;
           }))
          continue;

        last_index = std::max(last_index, std::stoi(suffix));
      }
    }
  }

  last_index = dealii::Utilities::MPI::broadcast(mpi_comm, last_index, 0);

  if(last_index < 0)
    return {};

  return static_cast<unsigned int>(last_index);
}

} // namespace ExaDG

#endif /* EXADG_UTILITIES_CREATE_DIRECTORIES_H_ */
