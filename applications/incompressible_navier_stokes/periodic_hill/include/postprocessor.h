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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_PERIODIC_HILL_POSTPROCESSOR_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_PERIODIC_HILL_POSTPROCESSOR_H_

// ExaDG
#include <exadg/incompressible_navier_stokes/postprocessor/line_plot_calculation_statistics_homogeneous.h>
#include <exadg/incompressible_navier_stokes/postprocessor/mean_velocity_calculator.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_consistent_splitting_extruded.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/momentum_operator_rt.h>

#include "flow_rate_controller.h"

namespace ExaDG
{
namespace IncNS
{
template<int dim>
struct MyPostProcessorData
{
  PostProcessorData<dim>          pp_data;
  MeanVelocityCalculatorData<dim> mean_velocity_data;
  LinePlotDataStatistics<dim>     line_plot_data;
};

template<int dim, typename Number>
class MyPostProcessor : public PostProcessor<dim, Number>
{
public:
  typedef PostProcessor<dim, Number> Base;

  typedef typename Base::VectorType VectorType;

  typedef typename Base::Operator Operator;

  MyPostProcessor(MyPostProcessorData<dim> const & my_pp_data_in,
                  MPI_Comm const &                 mpi_comm,
                  double const                     length_in,
                  FlowRateController &             flow_rate_controller_in)
    : Base(my_pp_data_in.pp_data, mpi_comm),
      my_pp_data(my_pp_data_in),
      length(length_in),
      flow_rate_controller(flow_rate_controller_in),
      consistent_splitting_operator(nullptr),
      clear_files(true)
  {
  }

  ~MyPostProcessor()
  {
    write_flow_rate_output(-1., -1., true);
  }

  void
  setup(Operator const & pde_operator) final
  {
    // call setup function of base class
    Base::setup(pde_operator);

    consistent_splitting_operator =
      dynamic_cast<const OperatorConsistentSplittingExtruded<dim, Number> *>(&pde_operator);
    if(not consistent_splitting_operator)
    {
      // calculation of mean velocity
      mean_velocity_calculator.reset(
        new MeanVelocityCalculator<dim, Number>(pde_operator.get_matrix_free(),
                                                pde_operator.get_dof_index_velocity(),
                                                pde_operator.get_quad_index_velocity_standard(),
                                                my_pp_data.mean_velocity_data,
                                                this->mpi_comm));
    }

    // evaluation of characteristic quantities along lines
    line_plot_calculator_statistics.reset(
      new LinePlotCalculatorStatisticsHomogeneous<dim, Number>(pde_operator.get_dof_handler_u(),
                                                               pde_operator.get_dof_handler_p(),
                                                               *pde_operator.get_mapping(),
                                                               this->mpi_comm));

    if(consistent_splitting_operator)
      line_plot_calculator_statistics->setup(my_pp_data.line_plot_data,
                                             *consistent_splitting_operator->momentum_operator);
    else
      line_plot_calculator_statistics->setup(my_pp_data.line_plot_data);
  }

  bool
  needs_constraints_distributed(double const           time,
                                types::time_step const time_step_number) const override
  {
    // if we have the consistent splitting operator, we do not need to take
    // care of constraints as we use the matrix-free operator directly
    return consistent_splitting_operator == nullptr ||
           this->output_generator.time_control.check_for_evaluation(time, time_step_number);
  }

  void
  do_postprocessing(VectorType const &     velocity,
                    VectorType const &     pressure,
                    double const           time,
                    types::time_step const time_step_number) final
  {
    Base::do_postprocessing(velocity, pressure, time, time_step_number);

    if(my_pp_data.mean_velocity_data.calculate == true)
    {
      // calculation of flow rate
      double const flow_rate =
        consistent_splitting_operator ?
          compute_velocity_integral() / length :
          mean_velocity_calculator->calculate_flow_rate_volume(velocity, time, length);

      // if done manually in our function rather than
      // mean_velocity_calculator, need to write the file here
      if(consistent_splitting_operator)
        write_flow_rate_output(flow_rate, time, false);

      // update body force
      flow_rate_controller.update_body_force(flow_rate, time);
    }

    // line plot statistics
    if(line_plot_calculator_statistics->time_control_statistics.time_control.needs_evaluation(
         time, time_step_number))
    {
      double time_step_size = 0.0;
      if(previous_time >= 0.0)
        time_step_size = time - previous_time;
      previous_time = time;

      if(time_step_size > 0.0)
      {
        if(consistent_splitting_operator)
          line_plot_calculator_statistics->evaluate(*consistent_splitting_operator->velocity_vector,
                                                    pressure,
                                                    time_step_size);
        else
          line_plot_calculator_statistics->evaluate(velocity, pressure, time_step_size);
      }
    }

    if(line_plot_calculator_statistics->time_control_statistics.write_preliminary_results(
         time, time_step_number))
    {
      line_plot_calculator_statistics->write_output();
    }
  }

private:
  // postprocessor data supplemented with data required for periodic hill
  MyPostProcessorData<dim> my_pp_data;

  // calculate flow rate in precursor domain so that the flow rate can be
  // dynamically adjusted by a flow rate controller.
  std::shared_ptr<MeanVelocityCalculator<dim, Number>> mean_velocity_calculator;

  double const length;

  FlowRateController & flow_rate_controller;

  // line plot statistics with averaging in homogeneous direction
  std::shared_ptr<LinePlotCalculatorStatisticsHomogeneous<dim, Number>>
    line_plot_calculator_statistics;

  const OperatorConsistentSplittingExtruded<dim, Number> * consistent_splitting_operator;

  mutable bool                                   clear_files;
  mutable std::vector<std::pair<double, double>> accumulated_flow_rate_results;

  mutable double previous_time = -1.0;

  Number
  compute_velocity_integral()
  {
    AssertThrow(consistent_splitting_operator != nullptr, dealii::ExcNotImplemented());
    AssertThrow(consistent_splitting_operator->momentum_operator.get() != nullptr,
                dealii::ExcNotImplemented());
    const auto & momentum_op = *consistent_splitting_operator->momentum_operator;
    std::array<unsigned int, dealii::VectorizedArray<Number>::size()>              indices;
    dealii::AlignedVector<dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>> velocities;
    Number                                                                         sum = 0;
    consistent_splitting_operator->velocity_vector->update_ghost_values();
    for(unsigned int batch = 0; batch < momentum_op.get_cell_level_index().size(); ++batch)
    {
      for(unsigned int v = 0; v < indices.size(); ++v)
        if(momentum_op.get_cell_level_index()[batch][v][0] != dealii::numbers::invalid_unsigned_int)
          indices[v] = batch * dealii::VectorizedArray<Number>::size() + v;
        else
          indices[v] = batch * dealii::VectorizedArray<Number>::size();
      momentum_op.evaluate_field(*consistent_splitting_operator->velocity_vector,
                                 indices,
                                 velocities,
                                 true);
      dealii::VectorizedArray<Number> cell_sum = 0;
      for(const auto & velocity : velocities)
        cell_sum += velocity[0];
      for(unsigned int v = 0; v < indices.size(); ++v)
        if(momentum_op.get_cell_level_index()[batch][v][0] != dealii::numbers::invalid_unsigned_int)
          sum += cell_sum[v];
    }
    consistent_splitting_operator->velocity_vector->zero_out_ghost_values();
    return dealii::Utilities::MPI::sum(
      sum, consistent_splitting_operator->velocity_vector->get_mpi_communicator());
  }

  void
  write_flow_rate_output(Number const & value, double const & time, bool const force_write) const
  {
    // write output file
    if(consistent_splitting_operator and my_pp_data.mean_velocity_data.write_to_file == true and
       dealii::Utilities::MPI::this_mpi_process(this->mpi_comm) == 0)
    {
      if(force_write || accumulated_flow_rate_results.size() >
                          my_pp_data.line_plot_data.time_control_data_statistics
                            .write_preliminary_results_every_nth_time_step)
      {
        std::string filename =
          my_pp_data.mean_velocity_data.directory + my_pp_data.mean_velocity_data.filename;

        std::ofstream f;
        if(clear_files == true)
        {
          f.open(filename.c_str(), std::ios::trunc);
          f << std::endl << "# Time                Mean velocity [m/s]" << std::endl;

          clear_files = false;
        }
        else
        {
          f.open(filename.c_str(), std::ios::app);
        }

        unsigned int precision = 12;
        for(std::pair<double, double> const & data : accumulated_flow_rate_results)
          f << std::scientific << std::setprecision(precision) << std::setw(precision + 8)
            << data.first << std::setw(precision + 8) << data.second << std::endl;
        accumulated_flow_rate_results.clear();
      }

      accumulated_flow_rate_results.emplace_back(time, value);
    }
  }
};

} // namespace IncNS
} // namespace ExaDG


#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_PERIODIC_HILL_POSTPROCESSOR_H_ */
