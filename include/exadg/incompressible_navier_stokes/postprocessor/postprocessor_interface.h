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

#ifndef EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_INTERFACE_H_
#define EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_INTERFACE_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/utilities/numbers.h>

namespace ExaDG
{
namespace IncNS
{
template<typename Number>
class PostProcessorInterface
{
protected:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

public:
  virtual ~PostProcessorInterface()
  {
  }

  /*
   * Return whether do_postprocessing() reads the velocity vector passed to it
   * in this time step. Time integrators that do not keep that vector up to
   * date in every time step only fill it when this function returns true.
   * The default returns true. Derived classes that evaluate the velocity
   * themselves, in addition to the tools of a base class, must take this
   * into account when overriding the function.
   */
  virtual bool
  needs_velocity_vector(double const /*          time */,
                        types::time_step const /*time_step_number*/) const
  {
    return true;
  }

  virtual bool
  needs_constraints_distributed(double const /*          time */,
                                types::time_step const /*time_step_number*/) const
  {
    return true;
  }

  /*
   * This function has to be called to apply the postprocessing tools.
   */
  virtual void
  do_postprocessing(VectorType const &     velocity,
                    VectorType const &     pressure,
                    double const           time             = 0.0,
                    types::time_step const time_step_number = numbers::steady_timestep) = 0;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_INTERFACE_H_ */
