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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_PERIODIC_HILL_FLOW_RATE_CONTROLLER_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_PERIODIC_HILL_FLOW_RATE_CONTROLLER_H_

namespace ExaDG
{
namespace IncNS
{
class FlowRateController
{
public:
  FlowRateController(double const bulk_velocity,
                     double const target_flow_rate,
                     double const H,
                     double const start_time,
                     bool const   assert_non_matching_parameters_at_restart = false)
    : bulk_velocity(bulk_velocity),
      target_flow_rate(target_flow_rate),
      length_scale(H),
      f(0.0), // f(t=t_0) = f_0
      f_damping(0.0),
      f_proportional(0.0),
      time_old(start_time),
      flow_rate(0.0),
      flow_rate_old(0.0),
      assert_non_matching_parameters_at_restart(assert_non_matching_parameters_at_restart)
  {
  }

  double
  get_body_force() const
  {
    return f + f_damping + f_proportional;
  }

  void
  update_body_force(double const flow_rate_in, double const time)
  {
    flow_rate              = flow_rate_in;
    double const time_step = time - time_old;

    // use an PI-controller with damping (D) to asymptotically reach the desired target flow rate

    // dimensional analysis: [k_I] = 1/(m^2 s^2) -> k_I = const * u_b^2 / H^4
    double const C_I = 100.0;
    double const k_I = C_I * std::pow(bulk_velocity, 2) / std::pow(length_scale, 4);
    f += k_I * (target_flow_rate - flow_rate) * time_step;

    // the time step size is 0 when this function is called the first time
    if(time_step > 0)
    {
      // dimensional analysis: [k_D] = 1/(m^2) -> k_D = const / H^2
      double const C_D = 0.1;
      double const k_D = C_D / std::pow(length_scale, 2.0);
      f_damping        = -k_D * (flow_rate - flow_rate_old) / time_step;
    }

    // proportional term (P) to damp the oscillation of the I-controller
    // dimensional analysis: [k_P] = 1/(m^2 s) -> k_P = const * u_b / H^3
    double const C_P = 2.0;
    double const k_P = C_P * bulk_velocity / std::pow(length_scale, 3);
    f_proportional   = k_P * (target_flow_rate - flow_rate);

    flow_rate_old = flow_rate;
    time_old      = time;
  }

  std::array<double, 8>
  get_parameters_for_serialization(bool const print_parameters = false) const
  {
    // Store complete state of the `FlowRateController`.
    std::array<double, 8> parameters{{f,
                                      f_damping,
                                      time_old,
                                      flow_rate,
                                      flow_rate_old,
                                      bulk_velocity,
                                      target_flow_rate,
                                      length_scale}};

    if(print_parameters)
    {
      print_parameters_for_all_ranks();
    }

    return parameters;
  }

  void
  set_parameters_from_serialization(const std::array<double, 8> & parameters,
                                    bool const                    print_parameters = false)
  {
    f             = parameters[0];
    f_damping     = parameters[1];
    time_old      = parameters[2];
    flow_rate     = parameters[3];
    flow_rate_old = parameters[4];

    // Check if the serialized data matches with the given parameters.
    if(assert_non_matching_parameters_at_restart)
    {
      const double bulk_velocity_old    = parameters[5];
      const double target_flow_rate_old = parameters[6];
      const double length_scale_old     = parameters[7];

      AssertThrow(std::abs(bulk_velocity_old - bulk_velocity) < 1e-20,
                  dealii::ExcMessage("The `bulk_velocity` parameter provided at restart "
                                     "does not match the value serialized"));
      AssertThrow(std::abs(target_flow_rate_old - target_flow_rate) < 1e-20,
                  dealii::ExcMessage("The `bulk_velocity` parameter provided at restart "
                                     "does not match the value serialized"));
      AssertThrow(std::abs(length_scale_old - length_scale) < 1e-20,
                  dealii::ExcMessage("The `bulk_velocity` parameter provided at restart "
                                     "does not match the value serialized"));
    }

    if(print_parameters)
    {
      print_parameters_for_all_ranks();
    }
  }

private:
  void
  print_parameters_for_all_ranks() const
  {
    unsigned int const mpi_process = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    for(unsigned int i = 0; i < dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++i)
    {
      if(i == mpi_process)
      {
        // clang-format off
        std::cout << "    FlowRateController parameters: (MPI process "
                  << std::to_string(mpi_process) << ")\n" << std::scientific << std::setprecision(4)
                  << "      f                = " << f << "\n"
                  << "      f_damping        = " << f_damping << "\n"
                  << "      f_proportional   = " << f_proportional << "\n"
                  << "      time_old         = " << time_old << "\n"
                  << "      flow_rate        = " << flow_rate << "\n"
                  << "      flow_rate_old    = " << flow_rate_old << "\n"
                  << "      bulk_velocity    = " << bulk_velocity << "\n"
                  << "      target_flow_rate = " << target_flow_rate << "\n"
                  << "      length_scale     = " << length_scale << "\n";
        // clang-format on
      }
    }
  }

  double const bulk_velocity, target_flow_rate, length_scale;

  double f;
  double f_damping;
  double f_proportional;

  double time_old;

  double flow_rate;
  double flow_rate_old;

  bool const assert_non_matching_parameters_at_restart;
};

} // namespace IncNS
} // namespace ExaDG


#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_PERIODIC_HILL_FLOW_RATE_CONTROLLER_H_ \
        */
