/*
 * incompressible_flow_with_transport.cc
 *
 *  Created on: Nov 6, 2018
 *      Author: fehn
 */

// deal.II
#include <deal.II/base/revision.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_tools.h>

// CONVECTION-DIFFUSION

// postprocessor
#include "convection_diffusion/postprocessor/postprocessor.h"

// spatial discretization
#include "convection_diffusion/spatial_discretization/dg_convection_diffusion_operation.h"

// time integration
#include "convection_diffusion/time_integration/time_int_bdf.h"
#include "convection_diffusion/time_integration/time_int_explicit_runge_kutta.h"

// user interface, etc.
#include "convection_diffusion/user_interface/analytical_solution.h"
#include "convection_diffusion/user_interface/boundary_descriptor.h"
#include "convection_diffusion/user_interface/field_functions.h"
#include "convection_diffusion/user_interface/input_parameters.h"
#include "functionalities/print_functions.h"
#include "functionalities/print_general_infos.h"

// NAVIER-STOKES

// postprocessor
#include "../include/incompressible_navier_stokes/postprocessor/postprocessor.h"

// spatial discretization
#include "../include/incompressible_navier_stokes/spatial_discretization/dg_navier_stokes_coupled_solver.h"
#include "../include/incompressible_navier_stokes/spatial_discretization/dg_navier_stokes_dual_splitting.h"
#include "../include/incompressible_navier_stokes/spatial_discretization/dg_navier_stokes_pressure_correction.h"

#include "../include/incompressible_navier_stokes/interface_space_time/operator.h"

// temporal discretization
#include "../include/incompressible_navier_stokes/time_integration/time_int_bdf_coupled_solver.h"
#include "../include/incompressible_navier_stokes/time_integration/time_int_bdf_dual_splitting.h"
#include "../include/incompressible_navier_stokes/time_integration/time_int_bdf_navier_stokes.h"
#include "../include/incompressible_navier_stokes/time_integration/time_int_bdf_pressure_correction.h"

// Parameters, BCs, etc.
#include "../include/incompressible_navier_stokes/user_interface/analytical_solution.h"
#include "../include/incompressible_navier_stokes/user_interface/boundary_descriptor.h"
#include "../include/incompressible_navier_stokes/user_interface/field_functions.h"
#include "../include/incompressible_navier_stokes/user_interface/input_parameters.h"

#include "../include/functionalities/print_general_infos.h"

using namespace dealii;

// select the test case
//#include "incompressible_flow_with_transport_test_cases/cavity.h"
#include "incompressible_flow_with_transport_test_cases/lung.h"


template<int dim, int degree_u, int degree_p, int degree_s, typename Number = double>
class Problem
{
public:
  Problem(unsigned int const refine_steps_space,
          unsigned int const refine_steps_time = 0,
          unsigned int const n_scalars         = 1);

  void
  setup(bool const do_restart);

  void
  solve() const;

  void
  analyze_computing_times() const;

private:
  // GENERAL (FLUID + TRANSPORT)
  void
  print_header() const;

  void
  run_timeloop() const;

  void
  set_start_time() const;

  void
  synchronize_time_step_size() const;

  double
  analyze_computing_times_fluid(double const overall_time) const;

  void
  analyze_iterations_fluid() const;

  double
  analyze_computing_times_transport(double const overall_time) const;

  void
  analyze_iterations_transport() const;

  ConditionalOStream pcout;

  std::shared_ptr<parallel::Triangulation<dim>> triangulation;
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_faces;

  // refinements in space or time
  unsigned int const n_refine_space, n_refine_time;

  // number of scalar quantities
  unsigned int const n_scalars;

  bool use_adaptive_time_stepping;

  // INCOMPRESSIBLE NAVIER-STOKES
  std::shared_ptr<IncNS::FieldFunctions<dim>>      fluid_field_functions;
  std::shared_ptr<IncNS::BoundaryDescriptorU<dim>> fluid_boundary_descriptor_velocity;
  std::shared_ptr<IncNS::BoundaryDescriptorP<dim>> fluid_boundary_descriptor_pressure;
  std::shared_ptr<IncNS::AnalyticalSolution<dim>>  fluid_analytical_solution;

  IncNS::InputParameters<dim> fluid_param;

  typedef IncNS::DGNavierStokesBase<dim, degree_u, degree_p, Number>          DGBase;
  typedef IncNS::DGNavierStokesCoupled<dim, degree_u, degree_p, Number>       DGCoupled;
  typedef IncNS::DGNavierStokesDualSplitting<dim, degree_u, degree_p, Number> DGDualSplitting;
  typedef IncNS::DGNavierStokesPressureCorrection<dim, degree_u, degree_p, Number>
    DGPressureCorrection;

  std::shared_ptr<DGBase> navier_stokes_operation;

  typedef IncNS::PostProcessorBase<dim, degree_u, degree_p, Number> Postprocessor;

  std::shared_ptr<Postprocessor> fluid_postprocessor;

  typedef IncNS::TimeIntBDF<dim, Number>                   TimeInt;
  typedef IncNS::TimeIntBDFCoupled<dim, Number>            TimeIntCoupled;
  typedef IncNS::TimeIntBDFDualSplitting<dim, Number>      TimeIntDualSplitting;
  typedef IncNS::TimeIntBDFPressureCorrection<dim, Number> TimeIntPressureCorrection;

  std::shared_ptr<TimeInt> fluid_time_integrator;

  // SCALAR TRANSPORT
  std::vector<ConvDiff::InputParameters> scalar_param;

  std::vector<std::shared_ptr<ConvDiff::FieldFunctions<dim>>>     scalar_field_functions;
  std::vector<std::shared_ptr<ConvDiff::BoundaryDescriptor<dim>>> scalar_boundary_descriptor;

  std::vector<std::shared_ptr<ConvDiff::AnalyticalSolution<dim>>> scalar_analytical_solution;

  typedef ConvDiff::DGOperation<dim, degree_s, Number> ConvDiffOperator;
  std::vector<std::shared_ptr<ConvDiffOperator>>       conv_diff_operator;

  std::vector<std::shared_ptr<ConvDiff::PostProcessor<dim, degree_s>>> scalar_postprocessor;

  std::vector<std::shared_ptr<TimeIntBase>> scalar_time_integrator;

  /*
   * Computation time (wall clock time).
   */
  Timer          timer;
  mutable double overall_time;
  double         setup_time;

  unsigned int const length = 15;
};

template<int dim, int degree_u, int degree_p, int degree_s, typename Number>
Problem<dim, degree_u, degree_p, degree_s, Number>::Problem(unsigned int const refine_steps_space,
                                                            unsigned int const refine_steps_time,
                                                            unsigned int const n_scalars)
  : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    n_refine_space(refine_steps_space),
    n_refine_time(refine_steps_time),
    n_scalars(n_scalars),
    use_adaptive_time_stepping(false),
    overall_time(0.0),
    setup_time(0.0)
{
  scalar_param.resize(n_scalars);
  scalar_field_functions.resize(n_scalars);
  scalar_boundary_descriptor.resize(n_scalars);
  scalar_analytical_solution.resize(n_scalars);

  conv_diff_operator.resize(n_scalars);
  scalar_postprocessor.resize(n_scalars);
  scalar_time_integrator.resize(n_scalars);
}

template<int dim, int degree_u, int degree_p, int degree_s, typename Number>
void
Problem<dim, degree_u, degree_p, degree_s, Number>::print_header() const
{
  // clang-format off
  pcout << std::endl << std::endl << std::endl
  << "_________________________________________________________________________________" << std::endl
  << "                                                                                 " << std::endl
  << "                High-order discontinuous Galerkin solver for the                 " << std::endl
  << "                unsteady, incompressible Navier-Stokes equations                 " << std::endl
  << "                             with scalar transport.                              " << std::endl
  << "_________________________________________________________________________________" << std::endl
  << std::endl;
  // clang-format on
}

template<int dim, int degree_u, int degree_p, int degree_s, typename Number>
void
Problem<dim, degree_u, degree_p, degree_s, Number>::setup(bool const do_restart)
{
  timer.restart();

  print_header();
  print_MPI_info(pcout);

  // parameters (fluid + scalar)
  fluid_param.set_input_parameters();
  fluid_param.check_input_parameters();

  if(fluid_param.print_input_parameters == true)
    fluid_param.print(pcout);

  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    scalar_param[i].set_input_parameters(i);
    scalar_param[i].check_input_parameters();
    AssertThrow(scalar_param[i].problem_type == ConvDiff::ProblemType::Unsteady,
                ExcMessage("ProblemType must be unsteady!"));

    if(scalar_param[i].print_input_parameters)
      scalar_param[i].print(pcout);
  }

  // triangulation
  if(fluid_param.triangulation_type == IncNS::TriangulationType::Distributed)
  {
    for(unsigned int i = 0; i < n_scalars; ++i)
    {
      AssertThrow(scalar_param[i].triangulation_type == ConvDiff::TriangulationType::Distributed,
                  ExcMessage(
                    "Parameter triangulation_type is different for fluid field and scalar field"));
    }

    triangulation.reset(new parallel::distributed::Triangulation<dim>(
      MPI_COMM_WORLD,
      dealii::Triangulation<dim>::none,
      parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy));
  }
  else if(fluid_param.triangulation_type == IncNS::TriangulationType::FullyDistributed)
  {
    for(unsigned int i = 0; i < n_scalars; ++i)
    {
      AssertThrow(
        scalar_param[i].triangulation_type == ConvDiff::TriangulationType::FullyDistributed,
        ExcMessage("Parameter triangulation_type is different for fluid field and scalar field"));
    }

    triangulation.reset(new parallel::fullydistributed::Triangulation<dim>(MPI_COMM_WORLD));
  }
  else
  {
    AssertThrow(false, ExcMessage("Invalid parameter triangulation_type."));
  }

  create_grid_and_set_boundary_ids(triangulation, n_refine_space, periodic_faces);

  print_grid_data(pcout, n_refine_space, *triangulation);

  // FLUID

  // boundary conditions
  fluid_boundary_descriptor_velocity.reset(new IncNS::BoundaryDescriptorU<dim>());
  fluid_boundary_descriptor_pressure.reset(new IncNS::BoundaryDescriptorP<dim>());

  IncNS::set_boundary_conditions(fluid_boundary_descriptor_velocity,
                                 fluid_boundary_descriptor_pressure);

  // field functions
  fluid_field_functions.reset(new IncNS::FieldFunctions<dim>());
  IncNS::set_field_functions(fluid_field_functions);

  // analytical solution
  fluid_analytical_solution.reset(new IncNS::AnalyticalSolution<dim>());
  IncNS::set_analytical_solution(fluid_analytical_solution);

  AssertThrow(fluid_param.solver_type == IncNS::SolverType::Unsteady,
              ExcMessage("This is an unsteady solver. Check input parameters."));

  // initialize postprocessor
  fluid_postprocessor =
    IncNS::construct_postprocessor<dim, degree_u, degree_p, Number>(fluid_param);

  // initialize navier_stokes_operation
  if(this->fluid_param.temporal_discretization == IncNS::TemporalDiscretization::BDFCoupledSolution)
  {
    std::shared_ptr<DGCoupled> navier_stokes_operation_coupled;

    navier_stokes_operation_coupled.reset(
      new DGCoupled(*triangulation, fluid_param, fluid_postprocessor));

    navier_stokes_operation = navier_stokes_operation_coupled;

    fluid_time_integrator.reset(new TimeIntCoupled(navier_stokes_operation_coupled,
                                                   navier_stokes_operation_coupled,
                                                   fluid_param,
                                                   n_refine_time));
  }
  else if(this->fluid_param.temporal_discretization ==
          IncNS::TemporalDiscretization::BDFDualSplittingScheme)
  {
    std::shared_ptr<DGDualSplitting> navier_stokes_operation_dual_splitting;

    navier_stokes_operation_dual_splitting.reset(
      new DGDualSplitting(*triangulation, fluid_param, fluid_postprocessor));

    navier_stokes_operation = navier_stokes_operation_dual_splitting;

    fluid_time_integrator.reset(new TimeIntDualSplitting(navier_stokes_operation_dual_splitting,
                                                         navier_stokes_operation_dual_splitting,
                                                         fluid_param,
                                                         n_refine_time));
  }
  else if(this->fluid_param.temporal_discretization ==
          IncNS::TemporalDiscretization::BDFPressureCorrection)
  {
    std::shared_ptr<DGPressureCorrection> navier_stokes_operation_pressure_correction;

    navier_stokes_operation_pressure_correction.reset(
      new DGPressureCorrection(*triangulation, fluid_param, fluid_postprocessor));

    navier_stokes_operation = navier_stokes_operation_pressure_correction;

    fluid_time_integrator.reset(
      new TimeIntPressureCorrection(navier_stokes_operation_pressure_correction,
                                    navier_stokes_operation_pressure_correction,
                                    fluid_param,
                                    n_refine_time));
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  AssertThrow(navier_stokes_operation.get() != 0, ExcMessage("Not initialized."));
  navier_stokes_operation->setup(periodic_faces,
                                 fluid_boundary_descriptor_velocity,
                                 fluid_boundary_descriptor_pressure,
                                 fluid_field_functions,
                                 fluid_analytical_solution);

  // setup time integrator before calling setup_solvers
  // (this is necessary since the setup of the solvers
  // depends on quantities such as the time_step_size or gamma0!!!)
  fluid_time_integrator->setup(do_restart);

  navier_stokes_operation->setup_solvers(
    fluid_time_integrator->get_scaling_factor_time_derivative_term());

  // SCALAR TRANSPORT
  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    // boundary conditions
    scalar_boundary_descriptor[i].reset(new ConvDiff::BoundaryDescriptor<dim>());

    ConvDiff::set_boundary_conditions(scalar_boundary_descriptor[i], i);

    // field functions
    scalar_field_functions[i].reset(new ConvDiff::FieldFunctions<dim>());
    ConvDiff::set_field_functions(scalar_field_functions[i], i);

    // analytical solution
    scalar_analytical_solution[i].reset(new ConvDiff::AnalyticalSolution<dim>());
    ConvDiff::set_analytical_solution(scalar_analytical_solution[i], i);

    // initialize postprocessor
    scalar_postprocessor[i].reset(new ConvDiff::PostProcessor<dim, degree_s>());

    // initialize convection diffusion operation
    conv_diff_operator[i].reset(new ConvDiff::DGOperation<dim, degree_s, Number>(
      *triangulation, scalar_param[i], scalar_postprocessor[i]));

    // initialize time integrator
    if(scalar_param[i].temporal_discretization == ConvDiff::TemporalDiscretization::ExplRK)
    {
      scalar_time_integrator[i].reset(
        new ConvDiff::TimeIntExplRK<Number>(conv_diff_operator[i], scalar_param[i], n_refine_time));
    }
    else if(scalar_param[i].temporal_discretization == ConvDiff::TemporalDiscretization::BDF)
    {
      scalar_time_integrator[i].reset(
        new ConvDiff::TimeIntBDF<Number>(conv_diff_operator[i], scalar_param[i], n_refine_time));
    }
    else
    {
      AssertThrow(scalar_param[i].temporal_discretization ==
                      ConvDiff::TemporalDiscretization::ExplRK ||
                    scalar_param[i].temporal_discretization ==
                      ConvDiff::TemporalDiscretization::BDF,
                  ExcMessage("Specified time integration scheme is not implemented!"));
    }

    if(fluid_param.adaptive_time_stepping == true)
    {
      AssertThrow(
        scalar_param[i].adaptive_time_stepping == true,
        ExcMessage(
          "Adaptive time stepping has to be used for both fluid and scalar transport solvers."));

      use_adaptive_time_stepping = true;
    }

    // The parameter start_with_low_order has to be true.
    // This is due to the fact that the setup function of the time integrator initializes
    // the solution at previous time instants t_0 - dt, t_0 - 2*dt, ... in case of
    // start_with_low_order == false. However, the combined time step size
    // is not known at this point since we have to first communicate the time step size
    // in order to find the minimum time step size. Hence, the easiest way to avoid these kind of
    // inconsistencies is to preclude the case start_with_low_order == false.
    AssertThrow(fluid_param.start_with_low_order == true &&
                  scalar_param[i].start_with_low_order == true,
                ExcMessage("start_with_low_order has to be true for this solver."));

    conv_diff_operator[i]->setup(periodic_faces,
                                 scalar_boundary_descriptor[i],
                                 scalar_field_functions[i],
                                 scalar_analytical_solution[i]);

    scalar_time_integrator[i]->setup(do_restart);

    if(scalar_param[i].temporal_discretization == ConvDiff::TemporalDiscretization::BDF)
    {
      std::shared_ptr<ConvDiff::TimeIntBDF<Number>> scalar_time_integrator_BDF =
        std::dynamic_pointer_cast<ConvDiff::TimeIntBDF<Number>>(scalar_time_integrator[i]);

      conv_diff_operator[i]->setup_solver(
        scalar_time_integrator_BDF->get_scaling_factor_time_derivative_term());
    }
  }

  setup_time = timer.wall_time();
}

template<int dim, int degree_u, int degree_p, int degree_s, typename Number>
void
Problem<dim, degree_u, degree_p, degree_s, Number>::set_start_time() const
{
  // Setup time integrator and get time step size
  double const fluid_time = fluid_time_integrator->get_time();

  double time = fluid_time;

  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    double const scalar_time = scalar_time_integrator[i]->get_time();
    time                     = std::min(time, scalar_time);
  }

  // Set the same start time for both solvers

  // fluid
  fluid_time_integrator->reset_time(time);

  // scalar transport
  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    scalar_time_integrator[i]->reset_time(time);
  }
}

template<int dim, int degree_u, int degree_p, int degree_s, typename Number>
void
Problem<dim, degree_u, degree_p, degree_s, Number>::synchronize_time_step_size() const
{
  double const EPSILON = 1.e-10;

  // Setup time integrator and get time step size
  double time_step_size_fluid = std::numeric_limits<double>::max();

  // fluid
  if(fluid_time_integrator->get_time() > fluid_param.start_time - EPSILON)
    time_step_size_fluid = fluid_time_integrator->get_time_step_size();

  double time_step_size = time_step_size_fluid;

  // scalar transport
  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    double time_step_size_scalar = std::numeric_limits<double>::max();
    if(scalar_time_integrator[i]->get_time() > scalar_param[i].start_time - EPSILON)
      time_step_size_scalar = scalar_time_integrator[i]->get_time_step_size();

    time_step_size = std::min(time_step_size, time_step_size_scalar);
  }

  if(use_adaptive_time_stepping == false)
  {
    // decrease time_step in order to exactly hit end_time
    time_step_size = (fluid_param.end_time - fluid_param.start_time) /
                     (1 + int((fluid_param.end_time - fluid_param.start_time) / time_step_size));

    pcout << std::endl << "Combined time step size dt = " << time_step_size << std::endl;
  }

  // Set the same time step size for both solvers

  // fluid
  fluid_time_integrator->set_time_step_size(time_step_size);

  // scalar transport
  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    scalar_time_integrator[i]->set_time_step_size(time_step_size);
  }
}

template<int dim, int degree_u, int degree_p, int degree_s, typename Number>
void
Problem<dim, degree_u, degree_p, degree_s, Number>::run_timeloop() const
{
  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    AssertThrow(scalar_param[i].type_velocity_field == ConvDiff::TypeVelocityField::Numerical,
                ExcMessage("Type of velocity field has to be TypeVelocityField::Numerical."));
  }

  bool              finished_fluid = false;
  std::vector<bool> finished_scalar(n_scalars, false);
  bool              finished_all_scalars = false;

  set_start_time();

  synchronize_time_step_size();

  while(!finished_fluid || !finished_all_scalars)
  {
    // We need to communicate between fluid solver and scalar transport solver, i.e.,
    // ask the fluid solver for the velocity field and hand it over to the scalar transport solver

    for(unsigned int i = 0; i < n_scalars; ++i)
    {
      if(scalar_param[i].treatment_of_convective_term ==
         ConvDiff::TreatmentOfConvectiveTerm::Explicit)
      {
        // get velocity at time t_{n}
        conv_diff_operator[i]->set_velocity(fluid_time_integrator->get_velocity());
      }
      else if(scalar_param[i].treatment_of_convective_term ==
              ConvDiff::TreatmentOfConvectiveTerm::ExplicitOIF)
      {
        std::vector<LinearAlgebra::distributed::Vector<Number> const *> velocities;
        std::vector<double>                                             times;

        fluid_time_integrator->get_velocities_and_times(velocities, times);

        conv_diff_operator[i]->set_velocities_and_times(velocities, times);
      }
    }

    // fluid: advance one time step
    finished_fluid = fluid_time_integrator->advance_one_timestep(!finished_fluid);

    // in case of an implicit treatment we first have to solve the fluid before sending the
    // velocity field to the scalar convection-diffusion solver
    for(unsigned int i = 0; i < n_scalars; ++i)
    {
      if(scalar_param[i].treatment_of_convective_term ==
         ConvDiff::TreatmentOfConvectiveTerm::Implicit)
      {
        // get velocity at time t_{n+1}
        conv_diff_operator[i]->set_velocity(fluid_time_integrator->get_velocity());
      }

      // scalar transport: advance one time step
      finished_scalar[i] = scalar_time_integrator[i]->advance_one_timestep(!finished_scalar[i]);
    }

    if(use_adaptive_time_stepping == true)
    {
      // Both solvers have already calculated the new, adaptive time step size individually in
      // function advance_one_timestep(). Here, we only have to synchronize the time step size.
      synchronize_time_step_size();
    }

    // all scalars finished?
    finished_all_scalars = true;
    for(unsigned int i = 0; i < n_scalars; ++i)
    {
      if(finished_scalar[i] == false)
        finished_all_scalars = false;
    }
  }
}

template<int dim, int degree_u, int degree_p, int degree_s, typename Number>
void
Problem<dim, degree_u, degree_p, degree_s, Number>::solve() const
{
  run_timeloop();

  overall_time += this->timer.wall_time();
}

template<int dim, int degree_u, int degree_p, int degree_s, typename Number>
double
Problem<dim, degree_u, degree_p, degree_s, Number>::analyze_computing_times_fluid(
  double const overall_time_avg) const
{
  this->pcout << std::endl << "Incompressible Navier-Stokes solver:" << std::endl;

  // wall times
  std::vector<std::string> names;
  std::vector<double>      computing_times;

  if(fluid_param.solver_type == IncNS::SolverType::Unsteady)
  {
    this->fluid_time_integrator->get_wall_times(names, computing_times);
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  double sum_of_substeps = 0.0;
  for(unsigned int i = 0; i < computing_times.size(); ++i)
  {
    Utilities::MPI::MinMaxAvg data =
      Utilities::MPI::min_max_avg(computing_times[i], MPI_COMM_WORLD);
    this->pcout << "  " << std::setw(length) << std::left << names[i] << std::setprecision(2)
                << std::scientific << std::setw(10) << std::right << data.avg << " s  "
                << std::setprecision(2) << std::fixed << std::setw(6) << std::right
                << data.avg / overall_time_avg * 100 << " %" << std::endl;

    sum_of_substeps += data.avg;
  }

  return sum_of_substeps;
}

template<int dim, int degree_u, int degree_p, int degree_s, typename Number>
void
Problem<dim, degree_u, degree_p, degree_s, Number>::analyze_iterations_fluid() const
{
  this->pcout << std::endl << "Incompressible Navier-Stokes solver:" << std::endl;

  // Iterations
  if(fluid_param.solver_type == IncNS::SolverType::Unsteady)
  {
    std::vector<std::string> names;
    std::vector<double>      iterations;

    this->fluid_time_integrator->get_iterations(names, iterations);

    for(unsigned int i = 0; i < iterations.size(); ++i)
    {
      this->pcout << "  " << std::setw(length + 2) << std::left << names[i] << std::fixed
                  << std::setprecision(2) << std::right << std::setw(6) << iterations[i]
                  << std::endl;
    }
  }
}

template<int dim, int degree_u, int degree_p, int degree_s, typename Number>
double
Problem<dim, degree_u, degree_p, degree_s, Number>::analyze_computing_times_transport(
  double const overall_time_avg) const
{
  double wall_time_summed_over_all_scalars = 0.0;

  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    this->pcout << std::endl << "Convection-diffusion solver for scalar " << i << ":" << std::endl;

    // wall times
    std::vector<std::string> names;
    std::vector<double>      computing_times;

    if(scalar_param[i].problem_type == ConvDiff::ProblemType::Unsteady)
    {
      this->scalar_time_integrator[i]->get_wall_times(names, computing_times);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    double sum_of_substeps = 0.0;
    for(unsigned int i = 0; i < computing_times.size(); ++i)
    {
      Utilities::MPI::MinMaxAvg data =
        Utilities::MPI::min_max_avg(computing_times[i], MPI_COMM_WORLD);
      this->pcout << "  " << std::setw(length) << std::left << names[i] << std::setprecision(2)
                  << std::scientific << std::setw(10) << std::right << data.avg << " s  "
                  << std::setprecision(2) << std::fixed << std::setw(6) << std::right
                  << data.avg / overall_time_avg * 100 << " %" << std::endl;

      sum_of_substeps += data.avg;
    }

    wall_time_summed_over_all_scalars += sum_of_substeps;
  }

  return wall_time_summed_over_all_scalars;
}

template<int dim, int degree_u, int degree_p, int degree_s, typename Number>
void
Problem<dim, degree_u, degree_p, degree_s, Number>::analyze_iterations_transport() const
{
  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    this->pcout << std::endl << "Convection-diffusion solver for scalar " << i << ":" << std::endl;

    // Iterations are only relevant for BDF time integrator
    if(scalar_param[i].temporal_discretization == ConvDiff::TemporalDiscretization::BDF)
    {
      // Iterations
      if(scalar_param[i].problem_type == ConvDiff::ProblemType::Unsteady)
      {
        std::vector<std::string> names;
        std::vector<double>      iterations;

        std::shared_ptr<ConvDiff::TimeIntBDF<Number>> time_integrator_bdf =
          std::dynamic_pointer_cast<ConvDiff::TimeIntBDF<Number>>(scalar_time_integrator[i]);
        time_integrator_bdf->get_iterations(names, iterations);

        for(unsigned int i = 0; i < iterations.size(); ++i)
        {
          this->pcout << "  " << std::setw(length + 2) << std::left << names[i] << std::fixed
                      << std::setprecision(2) << std::right << std::setw(6) << iterations[i]
                      << std::endl;
        }
      }
    }
  }
}

template<int dim, int degree_u, int degree_p, int degree_s, typename Number>
void
Problem<dim, degree_u, degree_p, degree_s, Number>::analyze_computing_times() const
{
  this->pcout << std::endl
              << "_________________________________________________________________________________"
              << std::endl
              << std::endl;

  // Iterations
  this->pcout << std::endl << "Average number of iterations:" << std::endl;

  analyze_iterations_fluid();
  analyze_iterations_transport();

  // Wall times

  this->pcout << std::endl << "Wall times:" << std::endl;
  Utilities::MPI::MinMaxAvg overall_time_data =
    Utilities::MPI::min_max_avg(overall_time, MPI_COMM_WORLD);
  double const overall_time_avg = overall_time_data.avg;

  double const time_fluid_avg  = analyze_computing_times_fluid(overall_time_avg);
  double const time_scalar_avg = analyze_computing_times_transport(overall_time_avg);

  this->pcout << std::endl;

  Utilities::MPI::MinMaxAvg setup_time_data =
    Utilities::MPI::min_max_avg(setup_time, MPI_COMM_WORLD);
  double const setup_time_avg = setup_time_data.avg;
  this->pcout << "  " << std::setw(length) << std::left << "Setup" << std::setprecision(2)
              << std::scientific << std::setw(10) << std::right << setup_time_avg << " s  "
              << std::setprecision(2) << std::fixed << std::setw(6) << std::right
              << setup_time_avg / overall_time_avg * 100 << " %" << std::endl;

  double const other = overall_time_avg - time_fluid_avg - time_scalar_avg - setup_time_avg;
  this->pcout << "  " << std::setw(length) << std::left << "Other" << std::setprecision(2)
              << std::scientific << std::setw(10) << std::right << other << " s  "
              << std::setprecision(2) << std::fixed << std::setw(6) << std::right
              << other / overall_time_avg * 100 << " %" << std::endl;

  this->pcout << "  " << std::setw(length) << std::left << "Overall" << std::setprecision(2)
              << std::scientific << std::setw(10) << std::right << overall_time_avg << " s  "
              << std::setprecision(2) << std::fixed << std::setw(6) << std::right
              << overall_time_avg / overall_time_avg * 100 << " %" << std::endl;

  // computational costs in CPUh
  unsigned int N_mpi_processes = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  this->pcout << std::endl
              << "Computational costs (fluid + transport, including setup + postprocessing):"
              << std::endl
              << "  Number of MPI processes = " << N_mpi_processes << std::endl
              << "  Wall time               = " << std::scientific << std::setprecision(2)
              << overall_time_avg << " s" << std::endl
              << "  Computational costs     = " << std::scientific << std::setprecision(2)
              << overall_time_avg * (double)N_mpi_processes / 3600.0 << " CPUh" << std::endl;

  // Throughput in DoFs/s per time step per core
  unsigned int DoFs = this->navier_stokes_operation->get_number_of_dofs();

  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    DoFs += this->conv_diff_operator[i]->get_number_of_dofs();
  }

  if(fluid_param.solver_type == IncNS::SolverType::Unsteady)
  {
    unsigned int N_time_steps      = this->fluid_time_integrator->get_number_of_time_steps();
    double const time_per_timestep = overall_time_avg / (double)N_time_steps;
    this->pcout << std::endl
                << "Throughput per time step (fluid + transport, including setup + postprocessing):"
                << std::endl
                << "  Degrees of freedom      = " << DoFs << std::endl
                << "  Wall time               = " << std::scientific << std::setprecision(2)
                << overall_time_avg << " s" << std::endl
                << "  Time steps              = " << std::left << N_time_steps << std::endl
                << "  Wall time per time step = " << std::scientific << std::setprecision(2)
                << time_per_timestep << " s" << std::endl
                << "  Throughput              = " << std::scientific << std::setprecision(2)
                << DoFs / (time_per_timestep * N_mpi_processes) << " DoFs/s/core" << std::endl;
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }


  this->pcout << "_________________________________________________________________________________"
              << std::endl
              << std::endl;
}

int
main(int argc, char ** argv)
{
  try
  {
    Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::cout << "deal.II git version " << DEAL_II_GIT_SHORTREV << " on branch "
                << DEAL_II_GIT_BRANCH << std::endl
                << std::endl;
    }

    deallog.depth_console(0);

    bool do_restart = false;
    if(argc > 1)
    {
      do_restart = std::atoi(argv[1]);
    }

    AssertThrow(FE_DEGREE_VELOCITY == FE_DEGREE_SCALAR, ExcMessage("Invalid parameters!"));
    AssertThrow(REFINE_STEPS_SPACE_MIN == REFINE_STEPS_SPACE_MAX,
                ExcMessage("Invalid parameters!"));
    AssertThrow(REFINE_STEPS_TIME_MIN == 0, ExcMessage("Invalid parameters!"));
    AssertThrow(REFINE_STEPS_TIME_MIN == REFINE_STEPS_TIME_MAX, ExcMessage("Invalid parameters!"));

    Problem<DIMENSION, FE_DEGREE_VELOCITY, FE_DEGREE_PRESSURE, FE_DEGREE_SCALAR, VALUE_TYPE>
      problem(REFINE_STEPS_SPACE_MIN, REFINE_STEPS_TIME_MIN, N_SCALARS);

    problem.setup(do_restart);

    problem.solve();

    problem.analyze_computing_times();
  }
  catch(std::exception & exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }
  catch(...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }
  return 0;
}