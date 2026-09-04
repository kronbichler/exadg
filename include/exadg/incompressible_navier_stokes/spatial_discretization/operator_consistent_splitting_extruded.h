/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2025 by the ExaDG authors
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

#ifndef EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATOR_CONSISTENT_SPLITTING_EXTRUDED_H_
#define EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATOR_CONSISTENT_SPLITTING_EXTRUDED_H_

// base class
#include <exadg/incompressible_navier_stokes/spatial_discretization/curl_compute.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_consistent_splitting.h>

namespace RTOperator
{
template<int, typename>
class RaviartThomasOperator;
}

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number = double>
class OperatorConsistentSplittingExtruded : public OperatorConsistentSplitting<dim, Number>
{
public:
  /*
   * Constructor.
   */
  OperatorConsistentSplittingExtruded(
    std::shared_ptr<Grid<dim> const>                      grid,
    std::shared_ptr<dealii::Mapping<dim> const>           mapping,
    std::shared_ptr<MultigridMappings<dim, Number>> const multigrid_mappings,
    std::shared_ptr<BoundaryDescriptor<dim> const>        boundary_descriptor,
    std::shared_ptr<FieldFunctions<dim> const>            field_functions,
    Parameters const &                                    parameters,
    std::string const &                                   field,
    MPI_Comm const &                                      mpi_comm)
    : OperatorConsistentSplitting<dim, Number>(grid,
                                               mapping,
                                               multigrid_mappings,
                                               boundary_descriptor,
                                               field_functions,
                                               parameters,
                                               field,
                                               mpi_comm)
  {
  }

  std::shared_ptr<const RTOperator::RaviartThomasOperator<dim, Number>> momentum_operator;
  dealii::LinearAlgebra::distributed::Vector<Number> const *            velocity_vector;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATOR_CONSISTENT_SPLITTING_H_ \
        */
