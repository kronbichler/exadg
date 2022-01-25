/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2022 by the ExaDG authors
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

#ifndef INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_DRIVER_SOLID_H_
#define INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_DRIVER_SOLID_H_

// application
#include <exadg/fluid_structure_interaction/user_interface/application_base.h>
#include <exadg/fluid_structure_interaction_precice/interface_coupling.h>

// utilities
#include <exadg/functions_and_boundary_conditions/verify_boundary_conditions.h>
#include <exadg/matrix_free/matrix_free_data.h>
#include <exadg/utilities/print_general_infos.h>

// grid
#include <exadg/grid/grid_motion_elasticity.h>
#include <exadg/grid/grid_motion_poisson.h>
#include <exadg/grid/mapping_degree.h>
#include <exadg/poisson/spatial_discretization/operator.h>

// Structure
#include <exadg/structure/spatial_discretization/operator.h>
#include <exadg/structure/time_integration/time_int_gen_alpha.h>

namespace ExaDG
{
namespace FSI
{
using namespace dealii;

template<int dim, typename Number>
class DriverSolid : public Driver<dim, Number>
{
private:
  using VectorType = typename LinearAlgebra::distributed::Vector<Number>;

public:
  DriverSolid(std::string const & input_file, MPI_Comm const & comm, bool const is_test)
    : Driver<dim, Number>(input_file, comm, is_test)
  {
  }

  void
  setup(std::shared_ptr<ApplicationBase<dim, Number>> application,
        unsigned int const                            degree_fluid,
        unsigned int const                            degree_structure,
        unsigned int const                            refine_space_fluid,
        unsigned int const                            refine_space_structure)
  {
    (void)degree_fluid;
    (void)refine_space_fluid;

    print_exadg_header(this->pcout);
    this->pcout << "Setting up fluid-structure interaction solver:" << std::endl;

    if(!(this->is_test))
    {
      print_dealii_info(this->pcout);
      print_matrixfree_info<Number>(this->pcout);
    }
    print_MPI_info(this->pcout, this->mpi_comm);

    this->application = application;

    /**************************************** STRUCTURE *****************************************/
    {
      this->application->set_parameters_structure(degree_structure);
      this->application->get_parameters_structure().check();
      // Some FSI specific Asserts
      AssertThrow(this->application->get_parameters_structure().pull_back_traction == true,
                  ExcMessage("Invalid parameter in context of fluid-structure interaction."));
      this->application->get_parameters_structure().print(this->pcout,
                                                          "List of parameters for structure:");

      // grid
      GridData structure_grid_data;
      structure_grid_data.triangulation_type =
        this->application->get_parameters_structure().triangulation_type;
      structure_grid_data.n_refine_global = refine_space_structure;
      structure_grid_data.mapping_degree =
        get_mapping_degree(this->application->get_parameters_structure().mapping,
                           this->application->get_parameters_structure().degree);

      structure_grid = this->application->create_grid_structure(structure_grid_data);
      print_grid_info(this->pcout, *structure_grid);

      // boundary conditions
      this->application->set_boundary_descriptor_structure();
      verify_boundary_conditions(*this->application->get_boundary_descriptor_structure(),
                                 *structure_grid);

      // material_descriptor
      this->application->set_material_descriptor_structure();

      // field functions
      this->application->set_field_functions_structure();

      // setup spatial operator
      structure_operator = std::make_shared<Structure::Operator<dim, Number>>(
        structure_grid,
        this->application->get_boundary_descriptor_structure(),
        this->application->get_field_functions_structure(),
        this->application->get_material_descriptor_structure(),
        this->application->get_parameters_structure(),
        "elasticity",
        this->mpi_comm);

      // initialize matrix_free
      structure_matrix_free_data = std::make_shared<MatrixFreeData<dim, Number>>();
      structure_matrix_free_data->append(structure_operator);

      structure_matrix_free = std::make_shared<MatrixFree<dim, Number>>();
      structure_matrix_free->reinit(*structure_grid->mapping,
                                    structure_matrix_free_data->get_dof_handler_vector(),
                                    structure_matrix_free_data->get_constraint_vector(),
                                    structure_matrix_free_data->get_quadrature_vector(),
                                    structure_matrix_free_data->data);

      structure_operator->setup(structure_matrix_free, structure_matrix_free_data);

      // initialize postprocessor
      structure_postprocessor = this->application->create_postprocessor_structure();
      structure_postprocessor->setup(structure_operator->get_dof_handler(),
                                     *structure_grid->mapping);
    }

    /**************************************** STRUCTURE *****************************************/


    /*********************************** INTERFACE COUPLING *************************************/

    this->precice = std::make_shared<Adapter::Adapter<dim, dim, VectorType>>(
      this->precice_parameters,
      application->get_boundary_descriptor_structure()->neumann_mortar_bc.begin()->first,
      structure_matrix_free,
      structure_operator->get_dof_index(),
      0 /*dummy*/,
      /*unused(dummy)*/ false);


    // structure to ALE
    // structure to fluid

    // fluid to structure
    {
      std::vector<unsigned int> quad_indices;
      quad_indices.emplace_back(structure_operator->get_quad_index());

      // VectorType stress_fluid;
      communicator = std::make_shared<InterfaceCoupling<dim, dim, Number>>(this->precice);
      VectorType displacement_structure;
      structure_operator->initialize_dof_vector(displacement_structure);
      communicator->setup(structure_matrix_free,
                          structure_operator->get_dof_index(),
                          quad_indices,
                          this->application->get_boundary_descriptor_structure()->neumann_mortar_bc,
                          displacement_structure);
    }

    /*********************************** INTERFACE COUPLING *************************************/


    /**************************** SETUP TIME INTEGRATORS AND SOLVERS ****************************/

    // Structure
    {
      // initialize time integrator
      structure_time_integrator = std::make_shared<Structure::TimeIntGenAlpha<dim, Number>>(
        structure_operator,
        structure_postprocessor,
        0 /* refine_time */,
        this->application->get_parameters_structure(),
        this->mpi_comm,
        this->is_test);

      structure_time_integrator->setup(
        this->application->get_parameters_structure().restarted_simulation);

      structure_operator->setup_solver();
    }

    /**************************** SETUP TIME INTEGRATORS AND SOLVERS ****************************/
  }

  void
  solve() const final
  {
    Assert(this->application->get_parameters_fluid().adaptive_time_stepping == false,
           ExcNotImplemented());
    VectorType d_old;
    structure_operator->initialize_dof_vector(d_old);
    structure_time_integrator->advance_one_timestep_pre_solve(true);

    // preCICE dictates when the time loop is finished
    while(this->precice->is_coupling_ongoing())
    {
      structure_time_integrator->advance_one_timestep_pre_solve(
        this->precice->is_time_window_complete());

      this->precice->save_current_state_if_required(
        [&]() { d_old = structure_time_integrator->get_displacement_np(); });

      // update stress boundary condition for solid
      coupling_fluid_to_structure();

      // solve structural problem
      // store_solution needs to be true for compatibility
      structure_time_integrator->advance_one_timestep_partitioned_solve(false, true);
      // send displacement data to ale
      coupling_structure_to_ale(structure_time_integrator->get_displacement_np(),
                                structure_time_integrator->get_time_step_size());

      // send velocity boundary condition for fluid
      // coupling_structure_to_fluid(structure_time_integrator->get_velocity_np(),
      //                             structure_time_integrator->get_time_step_size());

      this->precice->advance(structure_time_integrator->get_time_step_size());

      // Needs to be called before the swaps in post_solve
      this->precice->reload_old_state_if_required(
        [&]() { structure_time_integrator->set_displacement(d_old); });

      if(this->precice->is_time_window_complete())
        structure_time_integrator->advance_one_timestep_post_solve();
    }
  }

  void
  print_performance_results(double const total_time) const override
  {
    this->pcout
      << std::endl
      << "_________________________________________________________________________________"
      << std::endl
      << std::endl;

    this->pcout << "Performance results for fluid-structure interaction solver:" << std::endl;

    this->pcout << std::endl << "Structure:" << std::endl;
    structure_time_integrator->print_iterations();

    // wall times
    this->pcout << std::endl << "Wall times:" << std::endl;

    this->timer_tree.insert({"FSI"}, total_time);

    this->timer_tree.insert({"FSI"}, structure_time_integrator->get_timings(), "Structure");

    this->pcout << std::endl << "Timings for level 1:" << std::endl;
    this->timer_tree.print_level(this->pcout, 1);

    this->pcout << std::endl << "Timings for level 2:" << std::endl;
    this->timer_tree.print_level(this->pcout, 2);

    // Throughput in DoFs/s per time step per core
    types::global_dof_index DoFs = structure_operator->get_number_of_dofs();

    unsigned int const N_mpi_processes = Utilities::MPI::n_mpi_processes(this->mpi_comm);

    Utilities::MPI::MinMaxAvg total_time_data =
      Utilities::MPI::min_max_avg(total_time, this->mpi_comm);
    double const total_time_avg = total_time_data.avg;

    unsigned int N_time_steps = structure_time_integrator->get_number_of_time_steps();

    print_throughput_unsteady(this->pcout, DoFs, total_time_avg, N_time_steps, N_mpi_processes);

    // computational costs in CPUh
    print_costs(this->pcout, total_time_avg, N_mpi_processes);

    this->pcout
      << "_________________________________________________________________________________"
      << std::endl
      << std::endl;
  }

private:
  void
  coupling_structure_to_ale(VectorType const & displacement_structure,
                            const double       time_step_size) const
  {
    this->precice->write_data(displacement_structure, time_step_size);
  }

  void
  coupling_structure_to_fluid(bool const) const
  {
  }

  void
  coupling_fluid_to_structure() const
  {
    communicator->read();
  }

  // grid
  std::shared_ptr<Grid<dim, Number>> structure_grid;

  // matrix-free
  std::shared_ptr<MatrixFreeData<dim, Number>> structure_matrix_free_data;
  std::shared_ptr<MatrixFree<dim, Number>>     structure_matrix_free;

  // spatial discretization
  std::shared_ptr<Structure::Operator<dim, Number>> structure_operator;

  // temporal discretization
  std::shared_ptr<Structure::TimeIntGenAlpha<dim, Number>> structure_time_integrator;

  // postprocessor
  std::shared_ptr<Structure::PostProcessor<dim, Number>> structure_postprocessor;

  /**************************************** STRUCTURE *****************************************/


  /******************************* FLUID - STRUCTURE - INTERFACE ******************************/

  std::shared_ptr<InterfaceCoupling<dim, dim, Number>> communicator;

  /******************************* FLUID - STRUCTURE - INTERFACE ******************************/
};

} // namespace FSI
} // namespace ExaDG


#endif /* INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_DRIVER_H_ */
