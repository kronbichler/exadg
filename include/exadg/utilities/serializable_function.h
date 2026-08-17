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

#ifndef EXADG_UTILITIES_SERIALIZABLE_FUNCTION_H_
#define EXADG_UTILITIES_SERIALIZABLE_FUNCTION_H_

// C/C++
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

// deal.II
#include <deal.II/base/function.h>

namespace ExaDG
{
template<int dim, typename Number = double>
class SerializableFunction : public dealii::Function<dim, Number>
{
public:
  SerializableFunction(const unsigned int n_components = 1)
    : dealii::Function<dim, Number>(n_components)
  {
  }

  SerializableFunction(const dealii::Function<dim> & function)
    : dealii::Function<dim, Number>(function)
  {
  }

  /**
   * These functions define the interface for the added functionality to
   * serialize and deserialize the parameters of derived classes.
   */

  // Add data to be serialized to the archive.
  virtual void
  write_restart_data(boost::archive::text_oarchive & archive) const = 0;

  // Read serialized data from the archive. Note that the information is always
  // read on rank 0 and then broadcast to all ranks by the additional function
  // `broadcast_function_parameters()`
  virtual void
  read_restart_data(boost::archive::text_iarchive & archive) = 0;

  // Broadcast the de-/serialized parameters from `root_rank` to all others.
  virtual void
  broadcast_function_parameters(const MPI_Comm mpi_comm, const unsigned int root_rank = 0) = 0;
};

} // namespace ExaDG

#endif /* EXADG_UTILITIES_SERIALIZABLE_FUNCTION_H_ */
