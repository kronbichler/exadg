/*
 * WriteOutputNavierStokes.h
 *
 *  Created on: Oct 13, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_WRITEOUTPUTNAVIERSTOKES_H_
#define INCLUDE_WRITEOUTPUTNAVIERSTOKES_H_

#include <deal.II/numerics/data_out.h>

template<int dim>
void write_output_navier_stokes(OutputDataNavierStokes const                &output_data,
                                DoFHandler<dim> const                       &dof_handler_velocity,
                                DoFHandler<dim> const                       &dof_handler_pressure,
                                Mapping<dim> const                          &mapping,
                                parallel::distributed::Vector<double> const &velocity,
                                parallel::distributed::Vector<double> const &pressure,
                                parallel::distributed::Vector<double> const &vorticity,
                                parallel::distributed::Vector<double> const &divergence,
                                unsigned int const                          output_counter)
{
  DataOut<dim> data_out;
  std::vector<std::string> velocity_names (dim, "velocity");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    velocity_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
  data_out.add_data_vector (dof_handler_velocity, velocity, velocity_names, velocity_component_interpretation);

  std::vector<std::string> vorticity_names (dim, "vorticity");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    vorticity_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
  data_out.add_data_vector (dof_handler_velocity, vorticity, vorticity_names, vorticity_component_interpretation);

  pressure.update_ghost_values();
  data_out.add_data_vector (dof_handler_pressure,pressure, "p");

  if(output_data.compute_divergence == true)
  {
    std::vector<std::string> divergence_names (dim, "divergence");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      divergence_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    data_out.add_data_vector (dof_handler_velocity, divergence, divergence_names, divergence_component_interpretation);
  }

  std::ostringstream filename;
  filename << "output/"
           << output_data.output_prefix
           << "_Proc"
           << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
           << "_"
           << output_counter
           << ".vtu";

  data_out.build_patches (mapping, output_data.number_of_patches, DataOut<dim>::curved_inner_cells);

  std::ofstream output (filename.str().c_str());
  data_out.write_vtu (output);

  if ( Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::vector<std::string> filenames;
    for (unsigned int i=0;i<Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);++i)
    {
      std::ostringstream filename;
      filename << output_data.output_prefix
               << "_Proc"
               << i
               << "_"
               << output_counter
               << ".vtu";

      filenames.push_back(filename.str().c_str());
    }
    std::string master_name = "output/" + output_data.output_prefix + "_" + Utilities::int_to_string(output_counter) + ".pvtu";
    std::ofstream master_output (master_name.c_str());
    data_out.write_pvtu_record (master_output, filenames);
  }
}

template<int dim>
class OutputGenerator
{
public:
  OutputGenerator()
    :
    output_counter(0)
  {}

  void setup(DoFHandler<dim> const        &dof_handler_velocity_in,
             DoFHandler<dim> const        &dof_handler_pressure_in,
             Mapping<dim> const           &mapping_in,
             OutputDataNavierStokes const &output_data_in)
  {
    dof_handler_velocity = &dof_handler_velocity_in;
    dof_handler_pressure = &dof_handler_pressure_in;
    mapping = &mapping_in;
    output_data = output_data_in;

    // reset output counter
    output_counter = output_data.output_counter_start;
  }

  void write_output(parallel::distributed::Vector<double> const &velocity,
                    parallel::distributed::Vector<double> const &pressure,
                    parallel::distributed::Vector<double> const &vorticity,
                    parallel::distributed::Vector<double> const &divergence,
                    double const                                &time,
                    int const                                   &time_step_number)
  {
    const double EPSILON = 1.0e-10; // small number which is much smaller than the time step size

    if(output_data.write_output == true)
    {
      if(time_step_number >= 0) // unsteady problem
      {
        if(time > (output_data.output_start_time + output_counter*output_data.output_interval_time - EPSILON))
        {
          ConditionalOStream pcout(std::cout,
              Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
          pcout << std::endl << "OUTPUT << Write data at time t = "
                << std::scientific << std::setprecision(4) << time << std::endl;

          write_output_navier_stokes<dim>(output_data,
                                          *dof_handler_velocity,
                                          *dof_handler_pressure,
                                          *mapping,
                                          velocity,
                                          pressure,
                                          vorticity,
                                          divergence,
                                          output_counter);

          ++output_counter;
        }
      }
      else // steady problem (time_step_number = -1)
      {
        ConditionalOStream pcout(std::cout,
            Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
        pcout << std::endl << "OUTPUT << Write "
              << (output_counter == 0 ? "initial" : "solution") << " data"
              << std::endl;

        write_output_navier_stokes<dim>(output_data,
                                        *dof_handler_velocity,
                                        *dof_handler_pressure,
                                        *mapping,
                                        velocity,
                                        pressure,
                                        vorticity,
                                        divergence,
                                        output_counter);

        ++output_counter;
      }
    }
  }

private:
  unsigned int output_counter;

  SmartPointer< DoFHandler<dim> const > dof_handler_velocity;
  SmartPointer< DoFHandler<dim> const > dof_handler_pressure;
  SmartPointer< Mapping<dim> const > mapping;
  OutputDataNavierStokes output_data;
};


#endif /* INCLUDE_WRITEOUTPUTNAVIERSTOKES_H_ */
