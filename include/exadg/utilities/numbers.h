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

#ifndef EXADG_UTILITIES_NUMBERS_H_
#define EXADG_UTILITIES_NUMBERS_H_

// deal.II
#include <deal.II/base/mpi.h>
#include <deal.II/base/numbers.h>

// C/C++
#include <cmath>
#include <iomanip>
#include <limits>

namespace ExaDG
{
namespace types
{
using time_step = unsigned int;
}

namespace numbers
{
types::time_step const invalid_timestep = std::numeric_limits<unsigned int>::max();
types::time_step const steady_timestep  = std::numeric_limits<unsigned int>::max() - 1;
} // namespace numbers

namespace Utilities
{
inline bool
is_unsteady_timestep(types::time_step const timestep)
{
  return (timestep != numbers::steady_timestep);
}
inline bool
is_valid_timestep(types::time_step const timestep)
{
  return (timestep != numbers::invalid_timestep);
}

struct StatisticalQuantity
{
  StatisticalQuantity()
    : n_samples(0),
      min_value(std::numeric_limits<double>::infinity()),
      min_index(0),
      max_value(-min_value),
      max_index(0),
      sum_value(0.),
      n_samples_log(0),
      sum_logarithms(0.)
  {
  }

  void
  add_sample(const double value)
  {
    if(value < min_value)
    {
      min_value = value;
      min_index = n_samples;
    }
    if(value > max_value)
    {
      max_value = value;
      max_index = n_samples;
    }
    sum_value += value;
    ++n_samples;

    if(value > 0)
    {
      sum_logarithms += std::log(value);
      ++n_samples_log;
    }
  }

  double
  get_mean() const
  {
    if(n_samples > 0)
      return sum_value / static_cast<double>(n_samples);
    else
      return std::numeric_limits<double>::quiet_NaN();
  }

  double
  get_geometric_mean() const
  {
    if(n_samples_log > 0)
      return std::exp(sum_logarithms / static_cast<double>(n_samples_log));
    else
      return std::numeric_limits<double>::quiet_NaN();
  }

  template<typename StreamType>
  void
  print_statistics(StreamType &        stream,
                   const std::string & name,
                   const unsigned int  width,
                   MPI_Comm const *    mpi_comm = nullptr) const
  {
    double      min_print  = n_samples > 0 ? min_value : std::numeric_limits<double>::quiet_NaN();
    double      max_print  = n_samples > 0 ? max_value : std::numeric_limits<double>::quiet_NaN();
    double      mean_print = get_mean();
    double      geometric_mean_print = get_geometric_mean();
    std::size_t max_index_print      = max_index;
    std::size_t min_index_print      = min_index;
    if(mpi_comm != nullptr)
    {
      // Consider the maximum over all participating ranks.
      min_print            = dealii::Utilities::MPI::max(min_print, *mpi_comm);
      max_print            = dealii::Utilities::MPI::max(max_print, *mpi_comm);
      mean_print           = dealii::Utilities::MPI::max(mean_print, *mpi_comm);
      geometric_mean_print = dealii::Utilities::MPI::max(geometric_mean_print, *mpi_comm);

      // The index information is not communicated.
      max_index_print = dealii::numbers::invalid_size_type;
      min_index_print = dealii::numbers::invalid_size_type;
    }

    stream << "  " << std::left << std::setw(width) << name << " " << std::right << std::setw(8)
           << std::setprecision(2) << min_print << " (idx" << std::setw(6) << min_index_print
           << ")  " << std::setw(8) << geometric_mean_print << "  " << std::setw(8) << mean_print
           << "  (idx" << std::setw(6) << max_index_print << ") " << std::setw(8) << max_print
           << std::endl;
  }

private:
  std::size_t n_samples;
  double      min_value;
  std::size_t min_index;
  double      max_value;
  std::size_t max_index;
  double      sum_value;

  std::size_t n_samples_log;
  double      sum_logarithms;
};
} // namespace Utilities
} // namespace ExaDG

#endif /*EXADG_UTILITIES_NUMBERS_H_*/
