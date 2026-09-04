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

#ifndef EXADG_TIME_INTEGRATION_RESTART_DATA_H_
#define EXADG_TIME_INTEGRATION_RESTART_DATA_H_

// C/C++
#include <limits>

// deal.II
#include <deal.II/base/conditional_ostream.h>

// ExaDG
#include <exadg/grid/grid_data.h>
#include <exadg/incompressible_navier_stokes/user_interface/enum_types.h>
#include <exadg/solvers_and_preconditioners/solvers/solver_data.h>
#include <exadg/utilities/numbers.h>
#include <exadg/utilities/print_functions.h>
#include <exadg/utilities/serializable_function.h>

namespace ExaDG
{
template<int dim>
struct DeserializationParameters
{
  DeserializationParameters()
    : degree(dealii::numbers::invalid_unsigned_int),
      degree_u(dealii::numbers::invalid_unsigned_int),
      degree_p(dealii::numbers::invalid_unsigned_int),
      mapping_degree(dealii::numbers::invalid_unsigned_int),
      consider_mapping_write(false),
      triangulation_type(TriangulationType::Serial),
      spatial_discretization(IncNS::SpatialDiscretization::Undefined)
  {
  }

  void
  print(dealii::ConditionalOStream const & pcout) const
  {
    pcout << "  Deserialization parameters:" << std::endl;
    print_parameter(pcout, "Polynomial degree `degree`", degree);
    print_parameter(pcout, "Polynomial degree `degree_u`", degree_u);
    print_parameter(pcout, "Polynomial degree `degree_p`", degree_p);
    print_parameter(pcout, "Mapping degree", mapping_degree);
    print_parameter(pcout, "Consider mapping", consider_mapping_write);
    print_parameter(pcout, "Triangulation type", triangulation_type);
    print_parameter(pcout, "Spatial discretization", spatial_discretization);
  }

  void
  broadcast(const MPI_Comm comm, const unsigned int root_rank = 0)
  {
    std::array<unsigned int, 7> parameters{{degree,
                                            degree_u,
                                            degree_p,
                                            mapping_degree,
                                            consider_mapping_write,
                                            static_cast<unsigned int>(triangulation_type),
                                            static_cast<unsigned int>(spatial_discretization)}};
    parameters     = dealii::Utilities::MPI::broadcast(comm, parameters, root_rank);
    degree         = parameters[0];
    degree_u       = parameters[1];
    degree_p       = parameters[2];
    mapping_degree = parameters[3];
    AssertThrow(parameters[4] == 0 || parameters[4] == 1,
                dealii::ExcInternalError("Invalid value " + std::to_string(parameters[4]) +
                                         "of bool"));
    consider_mapping_write = parameters[4];
    triangulation_type     = TriangulationType(parameters[5]);
    spatial_discretization = IncNS::SpatialDiscretization(parameters[6]);

    AssertThrow(serializable_functions.size() ==
                  dealii::Utilities::MPI::max(serializable_functions.size(), comm),
                dealii::ExcDimensionMismatch(
                  serializable_functions.size(),
                  dealii::Utilities::MPI::max(serializable_functions.size(), comm)));
    for(std::shared_ptr<SerializableFunction<dim>> & function : serializable_functions)
      function->broadcast_function_parameters(comm, root_rank);
  }

  // Polynomial degrees of the finite elements used at serialization.
  unsigned int degree;
  unsigned int degree_u;
  unsigned int degree_p;

  // Polynomial degree of the mapping used at serialization.
  unsigned int mapping_degree;

  // The mapping is stored during serialization as a displacement vector.
  bool consider_mapping_write;

  // Triangulation type used at serialization.
  TriangulationType triangulation_type;

  // Spatial discretization used at serialization. Relevant for incompressible Navier-Stokes only.
  IncNS::SpatialDiscretization spatial_discretization;

  // Functions including possible state
  std::vector<std::shared_ptr<SerializableFunction<dim>>> serializable_functions;
};

struct RestartData
{
  RestartData()
    : write_restart(false),
      write_vectors_to_vtu(false),
      n_snapshots_keep(2),
      interval_time(std::numeric_limits<double>::max()),
      interval_time_start(std::numeric_limits<double>::lowest()),
      interval_time_end(std::numeric_limits<double>::max()),
      interval_wall_time(std::numeric_limits<double>::max()),
      interval_time_steps(std::numeric_limits<unsigned int>::max()),
      directory_coarse_triangulation("./output/"),
      directory_read("./output/"),
      directory_write("./output/"),
      filename("restart"),
      interval_counter(1),
      interval_counter_wall_time(1),
      discretization_identical(false),
      consider_mapping_write(false),
      consider_mapping_read_source(false),
      consider_mapping_read_target(false),
      consider_restart_time_in_mesh_movement_function(true),
      rpe_rtree_level(0),
      rpe_tolerance_unit_cell(1e-12),
      rpe_enforce_unique_mapping(false),
      solver_data(SolverData(1e3, 1e-20, 1e-6, LinearSolver::CG))
  {
  }

  void
  print(dealii::ConditionalOStream const & pcout) const
  {
    pcout << "  Restart:" << std::endl;
    print_parameter(pcout, "Write restart", write_restart);

    if(write_restart == true)
    {
      print_parameter(pcout, "Write the restart vectors to vtu", write_vectors_to_vtu);
      print_parameter(pcout, "Interval physical time", interval_time);
      print_parameter(pcout, "Interval physical time window start", interval_time_start);
      print_parameter(pcout, "Interval physical time window end", interval_time_end);
      print_parameter(pcout, "Interval wall time", interval_wall_time);
      print_parameter(pcout, "Interval time steps", interval_time_steps);
      print_parameter(pcout, "Directory coarse triangulation", directory_coarse_triangulation);
      print_parameter(pcout, "Directory for reading data", directory_read);
      print_parameter(pcout, "Directory for writing data", directory_write);
      print_parameter(pcout, "Filename", filename);
    }
  }

  bool
  do_restart(double const           wall_time,
             double const           time,
             types::time_step const time_step_number,
             bool const             reset_counter) const
  {
    // After a restart, the counter is reset to 1, but time = current_time - start time != 0 after a
    // restart. Hence, we have to explicitly reset the counter in that case. There is nothing to do
    // if the restart is controlled by the wall time or the time_step_number because these
    // variables are reinitialized after a restart anyway.
    if(reset_counter)
      interval_counter += static_cast<unsigned int>(time * (1.0 + 1.e-12) / interval_time);

    bool const trigger_restart_by_wall_time =
      wall_time > interval_wall_time * interval_counter_wall_time;
    if(trigger_restart_by_wall_time)
      ++interval_counter_wall_time;

    bool const trigger_restart_base = trigger_restart_by_wall_time or
                                      time > interval_time * interval_counter or
                                      time_step_number > interval_time_steps * interval_counter;

    // Additionally use the physical time window.
    bool const trigger_restart =
      trigger_restart_base and time >= interval_time_start and time <= interval_time_end;

    if(trigger_restart)
      ++interval_counter;

    return trigger_restart;
  }

  bool
  requires_dof_handler_mapping(bool const restarted_simulation) const
  {
    bool const reading_requires_dof_handler_mapping =
      restarted_simulation and consider_mapping_read_source;
    bool const writing_requires_dof_handler_mapping = write_restart and consider_mapping_write;

    return reading_requires_dof_handler_mapping or writing_requires_dof_handler_mapping;
  }

  bool
  requires_mapping_reset(bool const restarted_simulation = true) const
  {
    if(discretization_identical or not restarted_simulation)
    {
      // If we do not restart, we do not need to reset the mapppings. Also, mappings can be safely
      // ignored if we simply assign the vectors read from snapshot data.
      return false;
    }

    return not consider_mapping_read_target;
  }

  bool write_restart;

  // Write intermediate vectors in the current configuration to vtu.
  bool write_vectors_to_vtu;

  // Number of snapshots to keep
  unsigned int n_snapshots_keep;

  // physical time interval between serializations
  double interval_time;

  // physical time interval in which serialization is enabled
  double interval_time_start;
  double interval_time_end;

  // wall time in seconds (= hours * 3600)
  double interval_wall_time;

  // number of time steps after which to write restart
  unsigned int interval_time_steps;

  // Directories for restart files can be chosen independently for the coarse triangulation, the
  // snapshot to be read at start, and the snapshot data to be stored during the simulation.
  std::string directory_coarse_triangulation;
  std::string directory_read;
  std::string directory_write;

  // filename for restart files
  std::string filename;

  // counter needed do decide when to write restart regarding physical time
  mutable unsigned int interval_counter;

  // counter needed do decide when to write restart regarding elapsed wall time
  mutable unsigned int interval_counter_wall_time;

  // The discretization used when writing the restart data was identical to the current one.
  // Note that this includes the finite element, uniform and adaptive refinement, and the
  // `TriangulationType`, *but* one might consider a different number of MPI ranks for
  // `dealii::parallel::distributed::Triangulation` without the need for the otherwise
  // necessary global projection.
  bool discretization_identical;

  // Attach the mapping as a displacement vector when *writing* the restart data.
  bool consider_mapping_write;

  /**
   * The following options are only effective for the grid-to-grid projection, when
   * `discretization_identical == false`
   * These options toggle storing and reading the mapping via a displacement vector.
   * Mismatching parameters might lead to undesired configurations, use with care.
   */

  // Reconstruct the mapping for the serialized grid (`source`) in the grid-to-grid projection at
  // restart. The checkpointed data might carry a mapping, which is used if
  // `consider_mapping_read_source == true`. The grid used at restart can be considered as defined
  // in the application or reset to perform the grid-to-grid projection on an unmapped grid.
  bool consider_mapping_read_source;
  bool consider_mapping_read_target;

  // When creating a mapping function via `create_mesh_movement_function()`, use the `start_time` or
  // the `time` serialized to evaluate the mapping at restart.
  bool consider_restart_time_in_mesh_movement_function;

  // Parameters for `dealii::Utilities::MPI::RemotePointEvaluation<dim>::RemotePointEvaluation`
  // used for grid-to-grid projection.
  unsigned int rpe_rtree_level;
  double       rpe_tolerance_unit_cell;
  bool         rpe_enforce_unique_mapping;

  // Settings for Krylov solver used in case grid-to-grid projection is necessary.
  SolverData solver_data;
};

} // namespace ExaDG

#endif /* EXADG_TIME_INTEGRATION_RESTART_DATA_H_ */
