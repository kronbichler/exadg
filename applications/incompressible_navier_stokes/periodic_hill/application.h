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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_PERIODIC_HILL_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_PERIODIC_HILL_H_

// periodic hill application
#include "include/flow_rate_controller.h"
#include "include/manifold.h"
#include "include/postprocessor.h"

namespace ExaDG
{
namespace IncNS
{
/*
 * function to distort the undeformed box grid in
 * [0, length] x [0, height_hill+height_channel_at_hill_top] x [-width, width]
 * by a smooth trigonometric function |f(x)| < 1 that is zero on the boundaries in the
 * respective direction and is scaled with some scaling factor and has `n_periods` periods.
 * Lambda has no effect if `consider_box_distort == false`.
 */
template<int dim>
dealii::Point<dim>
box_distort(dealii::Point<dim> const & point_in,
            bool const                 consider_box_distort,
            double const               length_channel,
            double const               height_hill,
            double const               height_channel_at_hill_top)
{
  dealii::Point<dim> point_out = point_in;

  if(consider_box_distort)
  {
    double const n_periods_length = 2.0;
    double const n_periods_height = 1.0;
    double const scale_length = 0.02 * std::sin(dealii::numbers::PI * point_in[0] / length_channel);
    double const scale_hight  = 0.08 * std::sin(dealii::numbers::PI * (point_in[1] - height_hill) /
                                               height_channel_at_hill_top);

    point_out[0] +=
      length_channel * scale_length *
      std::sin(dealii::numbers::PI * n_periods_length * point_in[1] / height_channel_at_hill_top);
    point_out[1] += height_channel_at_hill_top * scale_hight *
                    std::sin(dealii::numbers::PI * n_periods_height * point_in[0] / length_channel);
  }

  return point_out;
};

/*
 * Initial condition for the velocity for the standard periodic hill benchmark. Qudratic flow
 * profile in upper part of the channel with added Gaussian noise.
 */
template<int dim>
class InitialConditionVelocity : public dealii::Function<dim>
{
public:
  InitialConditionVelocity(double const bulk_velocity,
                           double const height_hill,
                           double const height_channel_at_hill_top)
    : dealii::Function<dim>(dim, 0.0),
      bulk_velocity(bulk_velocity),
      height_hill(height_hill),
      height_channel_at_hill_top(height_channel_at_hill_top)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const final
  {
    // x-velocity
    double result = 0.0;
    if(component == 0)
    {
      // initial conditions
      if(p[1] > height_hill and p[1] < (height_hill + height_channel_at_hill_top))
        result = bulk_velocity * (p[1] - height_hill) *
                 ((height_hill + height_channel_at_hill_top) - p[1]) /
                 std::pow(height_channel_at_hill_top / 2.0, 2.0);

      // add some random perturbations
      result *= (1.0 + 0.1 * (((double)rand() / RAND_MAX - 0.5) / 0.5));
    }

    return result;
  }

private:
  double const bulk_velocity, height_hill, height_channel_at_hill_top;
};

/*
 * The driving force on the right hand side is controlled by the flow rate exiting the domain to
 * achieve the desired target flow rate in the periodic hill benchmark.
 */
template<int dim>
class RightHandSide : public SerializableFunction<dim>
{
public:
  RightHandSide(FlowRateController const & flow_rate_controller)
    : SerializableFunction<dim>(dim), flow_rate_controller(flow_rate_controller)
  {
  }

  double
  value(dealii::Point<dim> const & /*p*/, unsigned int const component = 0) const final
  {
    double result = 0.0;

    // The flow is driven by constant body force in x-direction
    if(component == 0)
    {
      result = flow_rate_controller.get_body_force();
    }

    return result;
  }

  void
  write_restart_data(boost::archive::binary_oarchive & archive) const override
  {
    const auto parameters =
      flow_rate_controller.get_parameters_for_serialization(true /* print_parameters */);
    archive & parameters;
  }

  // The information is always read on rank zero and then broadcast to all
  // ranks by the additional function `broadcast_function_parameters()`
  void
  read_restart_data(boost::archive::binary_iarchive & archive) override
  {
    // get right data type for parameters by getting (at this point invalid)
    // data from the controller...
    auto parameters =
      flow_rate_controller.get_parameters_for_serialization(false /* print_parameters */);

    // ... and now set the actual parameter as read from the file
    archive & parameters;
    const_cast<FlowRateController &>(flow_rate_controller)
      .set_parameters_from_serialization(parameters, true /* print_parameters */);
  }

  void
  broadcast_function_parameters(const MPI_Comm comm, const unsigned int rank = 0) override
  {
    auto parameters = flow_rate_controller.get_parameters_for_serialization();
    parameters      = dealii::Utilities::MPI::broadcast(comm, parameters, rank);
    const_cast<FlowRateController &>(flow_rate_controller)
      .set_parameters_from_serialization(parameters, false /* print_parameters */);
  }

private:
  FlowRateController const & flow_rate_controller;
};

/*
 * Manufactured solution for incompressible flow of a Newtonian fluid in a channel of height H,
 * width W, and length L, where the convective term may be disabled.
 * u  ... velocity vector
 * p  ... kinematic pressure
 * nu ... kinematic viscosity
 * f  ... body force vector
 *
 * d/dt(u) + (grad(u)) * u + grad(p) - nu * div(grad(u)) = f
 *
 * In 3D, we derive a solution by selecting the stream function
 *
 * psi = cos(a*x) * cos^2(b*y) * cos(c*t),
 *
 * u1 = d/dy psi = - 2 * b * cos(a*x) * cos(b*y) * sin(b*y) * cos(c*t),
 *
 * u2 = -d/dx psi = a * sin(a*x) * cos^2(b*y) * cos(c*t),
 *
 * u3 = 0,
 *
 * p  = cos(a*x) * cos(c*t),
 *
 * where
 *
 * a = 2*pi/L, b = pi/H, c = 2*pi/T, and T is the period of the solution in time.
 *
 * which is periodic in [0, length], i.e., the channel's longitudinal axis, constant in z, and
 * fulfills no-slip conditions at y = +-H/2. This means we have to modify the incoming y coordinate.
 *
 * The force vector follows from the momentum balance equation.
 */
template<int dim>
class ManufacturedSolutionVelocity : public dealii::Function<dim>
{
public:
  ManufacturedSolutionVelocity(double const & height,
                               double const & length,
                               double const & y_shift,
                               double const & time_period)
    : dealii::Function<dim>(dim /* n_components */, 0.0),
      height(height),
      length(length),
      y_shift(y_shift),
      time_period(time_period)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const final
  {
    double const a = 2.0 * dealii::numbers::PI / length;
    double const b = dealii::numbers::PI / height;
    double const c = 2.0 * dealii::numbers::PI / time_period;

    double const t = this->get_time();
    double const x = p[0];
    // shift incoming y coordinate since channel is not symmetric around 0.
    double const y      = p[1] - y_shift;
    double const sin_ax = std::sin(a * x);
    double const sin_by = std::sin(b * y);
    double const cos_ax = std::cos(a * x);
    double const cos_by = std::cos(b * y);
    double const cos_ct = std::cos(c * t);

    if(component == 0)
      return -2.0 * b * cos_ax * cos_by * sin_by * cos_ct;
    else if(component == 1)
      return a * sin_ax * dealii::Utilities::fixed_power<2>(cos_by) * cos_ct;
    else
      return 0.0;
  }

private:
  double const height;
  double const length;
  double const y_shift;
  double const time_period;
};


template<int dim>
class ManufacturedSolutionPressure : public dealii::Function<dim>
{
public:
  ManufacturedSolutionPressure(double const & length, double const & time_period)
    : dealii::Function<dim>(1, 0.0), length(length), time_period(time_period)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const /*component*/) const final
  {
    double const a = 2.0 * dealii::numbers::PI / length;
    double const c = 2.0 * dealii::numbers::PI / time_period;

    double const t      = this->get_time();
    double const x      = p[0];
    double const cos_ax = std::cos(a * x);
    double const cos_ct = std::cos(c * t);

    return cos_ax * cos_ct;
  }

private:
  double const length;
  double const time_period;
};


template<int dim>
class ManufacturedRightHandSide : public dealii::Function<dim>
{
public:
  ManufacturedRightHandSide(bool const     include_convective_term,
                            double const & height,
                            double const & length,
                            double const & y_shift,
                            double const & time_period,
                            double const & kinematic_viscosity)
    : dealii::Function<dim>(dim, 0.0),
      include_convective_term(include_convective_term),
      height(height),
      length(length),
      y_shift(y_shift),
      time_period(time_period),
      kinematic_viscosity(kinematic_viscosity)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const final
  {
    /*
     * The solution laid out above is inserted into the momentum balance
     * equation to derive the vector f on the right-hand side.
     *
     * f = d/dt(u) + (grad(u)) * u + grad(p) - nu * div(grad(u))
     *
     * Note that the solution is constant in z, such that the respective entries in the gradients
     * are empty and we can adapt a case corresponding to a 2D solution.
     */

    if constexpr(dim == 3)
      if(component == 2)
        return 0.0;

    double const a = 2.0 * dealii::numbers::PI / length;
    double const b = dealii::numbers::PI / height;
    double const c = 2.0 * dealii::numbers::PI / time_period;

    double const t = this->get_time();
    double const x = p[0];
    // shift incoming y coordinate since channel is not symmetric around 0.
    double const y      = p[1] - y_shift;
    double const sin_ax = std::sin(a * x);
    double const sin_by = std::sin(b * y);
    double const cos_ax = std::cos(a * x);
    double const cos_by = std::cos(b * y);
    double const sin_ct = std::sin(c * t);
    double const cos_ct = std::cos(c * t);

    double const u1 = -2.0 * b * cos_ax * cos_by * sin_by * cos_ct;

    double const du1_dt = 2.0 * b * cos_ax * cos_by * sin_by * sin_ct * c;
    double const du1_dx = 2.0 * b * a * sin_ax * cos_by * sin_by * cos_ct;
    double const du1_dy = 2.0 * b * b * cos_ax * cos_ct * (sin_by * sin_by - cos_by * cos_by);

    double const du1_dxx = 2.0 * b * a * a * cos_ax * cos_by * sin_by * cos_ct;
    double const du1_dyy = 8.0 * b * b * b * cos_ax * cos_ct * cos_by * sin_by;

    double const u2 = a * sin_ax * dealii::Utilities::fixed_power<2>(cos_by) * cos_ct;

    double const du2_dt = -a * sin_ax * dealii::Utilities::fixed_power<2>(cos_by) * sin_ct * c;
    double const du2_dx = a * a * cos_ax * dealii::Utilities::fixed_power<2>(cos_by) * cos_ct;
    double const du2_dy = -2.0 * a * b * sin_ax * cos_by * sin_by * cos_ct;

    double const du2_dxx = -a * a * a * sin_ax * dealii::Utilities::fixed_power<2>(cos_by) * cos_ct;
    double const du2_dyy = 2.0 * a * b * b * sin_ax * cos_ct * (sin_by * sin_by - cos_by * cos_by);

    // p = cos_ax * cos_ct;
    double const dp_dx = -a * sin_ax * cos_ct;
    double const dp_dy = 0.0;

    dealii::Tensor<2, dim> grad_u;
    grad_u[0][0] = du1_dx;
    grad_u[0][1] = du1_dy;
    grad_u[1][0] = du2_dx;
    grad_u[1][1] = du2_dy;

    dealii::Tensor<1, dim> div_grad_u;
    div_grad_u[0] = du1_dxx + du1_dyy;
    div_grad_u[1] = du2_dxx + du2_dyy;

    // add time derivative terms
    dealii::Tensor<1, dim> rhs;
    rhs[0] += du1_dt;
    rhs[1] += du2_dt;

    // convective term
    if(include_convective_term)
    {
      dealii::Tensor<1, dim> u;
      u[0] = u1;
      u[1] = u2;
      rhs += grad_u * u;
    }

    // pressure gradient
    rhs[0] += dp_dx;
    rhs[1] += dp_dy;

    // viscous term
    rhs -= kinematic_viscosity * div_grad_u;

    return rhs[component];
  }

private:
  bool const   include_convective_term;
  double const height;
  double const length;
  double const y_shift;
  double const time_period;
  double const kinematic_viscosity;
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, Number>(input_file, comm)
  {
  }

  void
  add_parameters(dealii::ParameterHandler & prm) final
  {
    ApplicationBase<dim, Number>::add_parameters(prm);

    prm.enter_subsection("Application");
    {
      prm.add_parameter("UseManufacturedSolution",
                        use_manufactured_solution,
                        "Use a manufactured solution to compute errors in the box domain.",
                        dealii::Patterns::Bool());
      prm.add_parameter("ConsiderBoxDistort",
                        consider_box_distort,
                        "Set whether to distort the box domain before mapping to the hill.",
                        dealii::Patterns::Bool());
      prm.add_parameter("ConsiderMapping",
                        consider_mapping,
                        "Set whether to map the box domain to form the periodic hill.",
                        dealii::Patterns::Bool());
      prm.add_parameter("WriteRestart",
                        write_restart,
                        "Set whether to write restart files be written",
                        dealii::Patterns::Bool());
      prm.add_parameter("ReadRestart",
                        read_restart,
                        "Set whether to restart the restarted simulation from restart data",
                        dealii::Patterns::Bool());
      prm.add_parameter("RestartDirectory",
                        restart_directory,
                        "Directory with restart data to start the simulation from.");
      prm.add_parameter("RestartIntervalTime",
                        restart_interval_time,
                        "Time between writes of restart data in multiples of flow-through time.",
                        dealii::Patterns::Double());
      prm.add_parameter("TriangulationType", triangulation_type, "Type of triangulation");
      prm.add_parameter("TemporalDiscretization",
                        temporal_discretization,
                        "Temporal discretization");
      prm.add_parameter("SpatialDiscretization", spatial_discretization, "Spatial discretization");
      prm.add_parameter("Inviscid",
                        inviscid,
                        "Consider inviscid flow, i.e., zero viscosity.",
                        dealii::Patterns::Bool());
      prm.add_parameter("ReynoldsNumber",
                        Re,
                        "Reynolds number (ignored if Inviscid = true)",
                        dealii::Patterns::Double());
      prm.add_parameter("EndTime",
                        end_time_multiples,
                        "End time in multiples of flow-through time.",
                        dealii::Patterns::Double(0.0, 1000.0));
      prm.add_parameter("GridStretchFactor",
                        grid_stretch_factor,
                        "Factor describing grid stretching in vertical direction.",
                        dealii::Patterns::Double());
      prm.add_parameter("CalculateStatistics",
                        calculate_statistics,
                        "Decides whether statistics are calculated.",
                        dealii::Patterns::Bool());
      prm.add_parameter("SampleStartTime",
                        sample_start_time_multiples,
                        "Start time of sampling in multiples of flow-through time.",
                        dealii::Patterns::Integer(0.0, 1000.0));
      prm.add_parameter("SampleEveryTimeSteps",
                        sample_every_timesteps,
                        "Sample every ... time steps.",
                        dealii::Patterns::Integer(1, 1000));
      prm.add_parameter("PointsPerLine",
                        points_per_line,
                        "Points per line in vertical direction.",
                        dealii::Patterns::Integer(1, 10000));
      prm.add_parameter("CoarseMeshRefinements",
                        coarse_mesh_refinements,
                        "Number of elements in coarse mesh in x,y,z-direction.");
    }
    prm.leave_subsection();
  }

private:
  void
  parse_parameters() final
  {
    ApplicationBase<dim, Number>::parse_parameters();

    // viscosity needs to be recomputed since the parameters inviscid, Re are
    // read from the input file
    viscosity = inviscid ? 0.0 : bulk_velocity * height_hill / Re;

    // depend on values defined in input file
    end_time          = temporal_convergence_study ? 0.1 : end_time_multiples * flow_through_time;
    sample_start_time = double(sample_start_time_multiples) * flow_through_time;
    restart_interval_time = restart_interval_time * flow_through_time;

    // sample end time is equal to end time, which is read from the input file
    sample_end_time = end_time;

    // We recompute the width, since we make it dependent on the number of elements in z direction,
    // or rather, the ratio between elements in x and z direction. A certain minimum width of the
    // domain has to be guaranteed as the flow is in general three-dimensional, which is observed
    // for higher Reynolds numbers.
    double const width_factor = static_cast<double>(coarse_mesh_refinements[2]) /
                                static_cast<double>(coarse_mesh_refinements[0]);
    width_channel = length_channel * width_factor;
    AssertThrow(width_factor > 0.5 - 1e-12,
                dealii::ExcMessage("The width of the channel needs to be at least 4.5 times "
                                   "the height of the hill, i.e., W >= 4.5 * H = 0.5 * L. "
                                   "The selected refinement parameters: [" +
                                   std::to_string(coarse_mesh_refinements[0]) + ", " +
                                   std::to_string(coarse_mesh_refinements[1]) + ", " +
                                   std::to_string(coarse_mesh_refinements[2]) +
                                   "] lead to W = " + std::to_string(width_factor) + " L."));

    // recompute target flow rate as it depends on the width
    target_flow_rate = bulk_velocity * width_channel * height_channel_at_hill_top;

    // finally refresh the flow rate controller
    flow_rate_controller.reset(
      new FlowRateController(bulk_velocity,
                             target_flow_rate,
                             height_hill,
                             start_time,
                             true /* assert_non_matching_parameters_at_restart */));
  }

  void
  set_parameters() final
  {
    // MATHEMATICAL MODEL
    this->param.problem_type = ProblemType::Unsteady;
    if(inviscid)
      this->param.equation_type = EquationType::Euler;
    else
      this->param.equation_type = EquationType::NavierStokes;
    this->param.formulation_viscous_term    = FormulationViscousTerm::LaplaceFormulation;
    this->param.formulation_convective_term = FormulationConvectiveTerm::ConvectiveFormulation;
    this->param.right_hand_side             = true;


    // PHYSICAL QUANTITIES
    this->param.start_time = start_time;
    this->param.end_time   = end_time;
    this->param.viscosity  = viscosity;


    // TEMPORAL DISCRETIZATION
    this->param.solver_type                  = SolverType::Unsteady;
    this->param.temporal_discretization      = temporal_discretization;
    this->param.treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
    this->param.calculation_of_time_step_size =
      temporal_convergence_study ? TimeStepCalculation::UserSpecified : TimeStepCalculation::CFL;
    this->param.adaptive_time_stepping          = temporal_convergence_study ? false : true;
    this->param.max_velocity                    = bulk_velocity;
    this->param.cfl                             = 0.32; // 0.375;
    this->param.cfl_exponent_fe_degree_velocity = 1.5;
    this->param.time_step_size                  = 1.0e-4;
    this->param.order_time_integrator           = 3;
    this->param.start_with_low_order            = read_restart ? false : true;

    // output of solver information
    this->param.solver_info_data.interval_time =
      temporal_convergence_study ? (end_time - start_time) / 10.0 : flow_through_time / 10.0;

    // SPATIAL DISCRETIZATION
    this->param.spatial_discretization      = spatial_discretization;
    this->param.grid.triangulation_type     = triangulation_type;
    this->param.mapping_degree              = this->param.degree_u;
    this->param.mapping_degree_coarse_grids = this->param.mapping_degree;
    this->param.degree_p                    = DegreePressure::MixedOrder;

    // convective term
    this->param.upwind_factor                = 1.0;
    this->param.type_dirichlet_bc_convective = TypeDirichletBCs::Direct;

    // viscous term
    this->param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // velocity pressure coupling terms
    this->param.gradp_formulation = FormulationPressureGradientTerm::Weak;
    this->param.divu_formulation  = FormulationVelocityDivergenceTerm::Weak;

    // div-div and continuity penalty
    this->param.use_divergence_penalty        = spatial_discretization == SpatialDiscretization::L2;
    this->param.divergence_penalty_factor     = 1.0e1;
    this->param.use_continuity_penalty        = spatial_discretization == SpatialDiscretization::L2;
    this->param.continuity_penalty_factor     = this->param.divergence_penalty_factor;
    this->param.continuity_penalty_components = ContinuityPenaltyComponents::Normal;
    this->param.apply_penalty_terms_in_postprocessing_step = true;
    this->param.continuity_penalty_use_boundary_data       = true;

    // TURBULENCE
    this->param.turbulence_model_data.is_active        = false;
    this->param.turbulence_model_data.turbulence_model = TurbulenceEddyViscosityModel::Sigma;
    // Smagorinsky: 0.165
    // Vreman: 0.28
    // WALE: 0.50
    // Sigma: 1.35
    this->param.turbulence_model_data.constant = 1.35;

    // RESTART
    this->param.restarted_simulation                        = read_restart;
    this->param.restart_data.write_restart                  = write_restart;
    this->param.restart_data.write_vectors_to_vtu           = true;
    this->param.restart_data.interval_time                  = restart_interval_time;
    this->param.restart_data.directory_coarse_triangulation = restart_directory;
    this->param.restart_data.directory_read                 = restart_directory;
    this->param.restart_data.directory_write                = this->output_parameters.directory;
    this->param.restart_data.filename            = this->output_parameters.filename + "_restart";
    this->param.restart_data.interval_wall_time  = 1.e6;
    this->param.restart_data.interval_time_steps = 1e8;

    // Same `mapping_degree` and spatial resolution are the most stable options for restart,
    // polynomial degree can be varied.
    bool constexpr de_serialize_in_deformed_geometry      = false;
    this->param.restart_data.consider_mapping_write       = de_serialize_in_deformed_geometry;
    this->param.restart_data.consider_mapping_read_source = de_serialize_in_deformed_geometry;
    this->param.restart_data.consider_mapping_read_target = de_serialize_in_deformed_geometry;
    this->param.restart_data.consider_restart_time_in_mesh_movement_function = false;

    this->param.restart_data.solver_data.abs_tol        = 1e-20;
    this->param.restart_data.solver_data.rel_tol        = 1e-12;
    this->param.restart_data.solver_data.max_iter       = 1e3;
    this->param.restart_data.rpe_rtree_level            = 3;
    this->param.restart_data.rpe_tolerance_unit_cell    = 1e-6;
    this->param.restart_data.rpe_enforce_unique_mapping = false;

    // PROJECTION METHODS

    // Linear solver tolerances are stricter for convergence study runs to achieve lower errors.
    double const abs_tol_lin            = convergence_study ? 1.0e-18 : 1.0e-12;
    double const rel_tol_lin_ppe        = convergence_study ? 1.0e-09 : 1.0e-05;
    double const rel_tol_lin_projection = convergence_study ? 1.0e-09 : 1.0e-06;
    double const rel_tol_lin_momentum   = convergence_study ? 1.0e-09 : 1.0e-08;
    double const rel_tol_lin_mass       = convergence_study ? 1.0e-09 : 1.0e-04;

    // pressure Poisson equation
    this->param.solver_data_pressure_poisson =
      SolverData(1000, abs_tol_lin, rel_tol_lin_ppe, LinearSolver::CG, 100);
    this->param.preconditioner_pressure_poisson      = PreconditionerPressurePoisson::Multigrid;
    this->param.multigrid_data_pressure_poisson.type = MultigridType::cphMG;
    this->param.multigrid_data_pressure_poisson.coarse_problem.solver =
      MultigridCoarseGridSolver::Chebyshev;
    this->param.multigrid_data_pressure_poisson.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::PointJacobi;

    // projection step
    this->param.solver_data_projection =
      SolverData(1000, abs_tol_lin, rel_tol_lin_projection, LinearSolver::CG);
    this->param.preconditioner_projection        = PreconditionerProjection::InverseMassMatrix;
    this->param.update_preconditioner_projection = true;


    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulations
    this->param.order_extrapolation_pressure_nbc =
      this->param.order_time_integrator <= 2 ? this->param.order_time_integrator : 2;

    this->param.solver_data_momentum =
      SolverData(1000, abs_tol_lin, rel_tol_lin_momentum, LinearSolver::CG);
    this->param.preconditioner_momentum = spatial_discretization == SpatialDiscretization::L2 ?
                                            MomentumPreconditioner::InverseMassMatrix :
                                            MomentumPreconditioner::PointJacobi;

    this->param.inverse_mass_operator.implementation_type = InverseMassType::GlobalKrylovSolver;
    this->param.inverse_mass_operator.preconditioner      = PreconditionerMass::PointJacobi;
    this->param.inverse_mass_operator.solver_data =
      SolverData(1000, abs_tol_lin, rel_tol_lin_mass, LinearSolver::CG);

    // CONSISTENT SPLITTING SCHEME
    this->param.order_extrapolation_pressure_rhs = 2;
    this->param.apply_leray_projection           = true;
  }

  void
  create_grid(Grid<dim> &                                       grid,
              std::shared_ptr<dealii::Mapping<dim>> &           mapping,
              std::shared_ptr<MultigridMappings<dim, Number>> & multigrid_mappings) final
  {
    auto const lambda_create_triangulation = [&](dealii::Triangulation<dim, dim> & tria,
                                                 std::vector<dealii::GridTools::PeriodicFacePair<
                                                   typename dealii::Triangulation<
                                                     dim>::cell_iterator>> & periodic_face_pairs,
                                                 unsigned int const          global_refinements,
                                                 std::vector<unsigned int> const &
                                                   vector_local_refinements) {
      (void)periodic_face_pairs;
      (void)vector_local_refinements;

      dealii::Point<dim> p_1;
      p_1[0] = 0.;
      p_1[1] = height_hill;
      if(dim == 3)
        p_1[2] = -width_channel / 2.0;

      dealii::Point<dim> p_2;
      p_2[0] = length_channel;
      p_2[1] = height_hill + height_channel_at_hill_top;
      if(dim == 3)
        p_2[2] = width_channel / 2.0;

      // use 2 cells in x-direction on coarsest grid and 1 cell in y- and z-directions
      std::vector<unsigned int> refinements{
        {coarse_mesh_refinements[0], coarse_mesh_refinements[1], coarse_mesh_refinements[2]}};
      if(dim == 2)
        refinements.resize(2);
      dealii::GridGenerator::subdivided_hyper_rectangle(tria, refinements, p_1, p_2);

      AssertThrow(
        this->param.grid.triangulation_type != TriangulationType::FullyDistributed,
        dealii::ExcMessage(
          "Periodic faces might not be applied correctly for TriangulationType::FullyDistributed. "
          "Try to use another triangulation type, or try to fix these limitations in ExaDG or deal.II."));

      // periodicity in x-direction (add 10 to avoid conflicts with dirichlet boundary, which is
      // 0) make use of the fact that the mesh has only two elements

      // left element
      for(const auto & cell : tria.cell_iterators())
      {
        if(cell->at_boundary(0))
          cell->face(0)->set_all_boundary_ids(0 + 10);
        if(cell->at_boundary(1))
          cell->face(1)->set_all_boundary_ids(1 + 10);

        // periodicity in z-direction
        if(dim == 3)
        {
          // left element
          if(cell->at_boundary(4))
            cell->face(4)->set_all_boundary_ids(2 + 10);
          if(cell->at_boundary(5))
            cell->face(5)->set_all_boundary_ids(3 + 10);
        }
      }

      dealii::GridTools::collect_periodic_faces(tria, 0 + 10, 1 + 10, 0, periodic_face_pairs);
      if(dim == 3)
      {
        dealii::GridTools::collect_periodic_faces(tria, 2 + 10, 3 + 10, 2, periodic_face_pairs);
      }

      tria.add_periodicity(periodic_face_pairs);

      // Save the *coarse* triangulation for later deserialization.
      if(write_restart and this->param.grid.triangulation_type == TriangulationType::Serial)
      {
        save_coarse_triangulation<dim>(this->param.restart_data, tria);
      }

      tria.refine_global(global_refinements);
    };

    if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplittingExtruded ||
       this->param.temporal_discretization ==
         TemporalDiscretization::BDFConsistentSplittingExtruded)
      GridUtilities::create_triangulation<dim>(grid,
                                               this->mpi_comm,
                                               this->param.grid,
                                               lambda_create_triangulation,
                                               {} /* no local refinements */);
    else
      GridUtilities::create_triangulation_with_multigrid<dim>(grid,
                                                              this->mpi_comm,
                                                              this->param.grid,
                                                              this->param.involves_h_multigrid(),
                                                              lambda_create_triangulation,
                                                              {} /* no local refinements */);

    // mappings
    AssertThrow(get_element_type(*grid.triangulation) == ElementType::Hypercube,
                dealii::ExcMessage("Only implemented for hypercube elements."));

    // dummy FE for compatibility with interface of dealii::FEValues
    dealii::FE_Nothing<dim>         dummy_fe;
    dealii::MappingQ1<dim>          mapping_undeformed;
    dealii::FEValues<dim>           fe_values(mapping_undeformed,
                                    dummy_fe,
                                    dealii::QGaussLobatto<dim>(this->param.mapping_degree + 1),
                                    dealii::update_quadrature_points);
    const std::vector<unsigned int> hierarchic_to_lexicographic_numbering =
      dealii::FETools::hierarchic_to_lexicographic_numbering<dim>(this->param.mapping_degree);

    const auto mapping_q_cache =
      std::make_shared<dealii::MappingQCache<dim>>(this->param.mapping_degree);
    mapping_q_cache->initialize(
      *grid.triangulation,
      [&](typename dealii::Triangulation<dim>::cell_iterator const & cell)
        -> std::vector<dealii::Point<dim>> {
        PeriodicHillManifold<dim> manifold(height_hill,
                                           length_channel,
                                           height_channel_at_hill_top,
                                           grid_stretch_factor);
        fe_values.reinit(cell);

        std::vector<dealii::Point<dim>> points_moved(fe_values.n_quadrature_points);
        for(unsigned int i = 0; i < fe_values.n_quadrature_points; ++i)
        {
          // need to adjust for hierarchic numbering of
          // dealii::MappingQCache
          dealii::Point<dim> const point_in_box =
            box_distort(fe_values.quadrature_point(hierarchic_to_lexicographic_numbering[i]),
                        consider_box_distort,
                        length_channel,
                        height_hill,
                        height_channel_at_hill_top);
          if(consider_mapping)
            points_moved[i] = manifold.push_forward(point_in_box);
          else
            points_moved[i] = point_in_box;
        }

        return points_moved;
      });

    grid.mapping_function = [&](typename dealii::Triangulation<dim>::cell_iterator const & cell)
      -> std::vector<dealii::Point<dim>> {
      PeriodicHillManifold<dim>       manifold(height_hill,
                                         length_channel,
                                         height_channel_at_hill_top,
                                         grid_stretch_factor);
      std::vector<dealii::Point<dim>> points_moved(cell->n_vertices());
      for(unsigned int i = 0; i < cell->n_vertices(); ++i)
      {
        // need to adjust for hierarchic numbering of
        // dealii::MappingQCache
        dealii::Point<dim> const point_in_box = box_distort(cell->vertex(i),
                                                            consider_box_distort,
                                                            length_channel,
                                                            height_hill,
                                                            height_channel_at_hill_top);

        if(consider_mapping)
          points_moved[i] = manifold.push_forward(point_in_box);
        else
          points_moved[i] = point_in_box;
      }

      return points_moved;
    };
    const auto mapping_coarse = std::make_shared<dealii::MappingQCache<dim>>(1);
    mapping_coarse->initialize(*grid.triangulation, grid.mapping_function);

    mapping = mapping_q_cache;
    multigrid_mappings =
      std::make_shared<MultigridMappings<dim, Number>>(mapping_q_cache, mapping_coarse);
  }

  void
  set_boundary_descriptor() final
  {
    // set boundary conditions
    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      pair;

    // fill boundary descriptor velocity
    this->boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(0, new dealii::Functions::ZeroFunction<dim>(dim)));

    // fill boundary descriptor pressure
    this->boundary_descriptor->pressure->neumann_bc.insert(0);
  }

  void
  set_field_functions() final
  {
    if(use_manufactured_solution)
    {
      AssertThrow(use_manufactured_solution == true and consider_mapping == false,
                  dealii::ExcMessage("Manufactured solution is defined on the box grid, "
                                     "cannot consider mapping to periodic hill."));

      this->field_functions->initial_solution_velocity.reset(new ManufacturedSolutionVelocity<dim>(
        height_channel_at_hill_top, length_channel, y_shift, time_period));
      this->field_functions->initial_solution_pressure.reset(
        new ManufacturedSolutionPressure<dim>(length_channel, time_period));

      this->field_functions->analytical_solution_pressure.reset(
        new ManufacturedSolutionPressure<dim>(length_channel, time_period));
      this->field_functions->analytical_solution_velocity.reset(
        new ManufacturedSolutionVelocity<dim>(
          height_channel_at_hill_top, length_channel, y_shift, time_period));

      bool const include_convective_term = this->param.equation_type == EquationType::NavierStokes;
      this->field_functions->right_hand_side.reset(
        new ManufacturedRightHandSide<dim>(include_convective_term,
                                           height_channel_at_hill_top,
                                           length_channel,
                                           y_shift,
                                           time_period,
                                           viscosity));
    }
    else
    {
      this->field_functions->initial_solution_velocity.reset(
        new InitialConditionVelocity<dim>(bulk_velocity, height_hill, height_channel_at_hill_top));
      this->field_functions->initial_solution_pressure.reset(
        new dealii::Functions::ZeroFunction<dim>(1));
      this->field_functions->analytical_solution_pressure.reset(
        new dealii::Functions::ZeroFunction<dim>(1));
      this->field_functions->right_hand_side.reset(new RightHandSide<dim>(*flow_rate_controller));
    }
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.time_control_data.is_active  = this->output_parameters.write;
    pp_data.output_data.time_control_data.start_time = start_time;
    pp_data.output_data.time_control_data.trigger_interval =
      convergence_study ? (end_time - start_time) / 10.0 : flow_through_time / 5.0;
    ;
    pp_data.output_data.directory                 = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename                  = this->output_parameters.filename;
    pp_data.output_data.write_velocity_magnitude  = false;
    pp_data.output_data.write_divergence          = true;
    pp_data.output_data.write_vorticity           = true;
    pp_data.output_data.write_vorticity_magnitude = false;
    pp_data.output_data.write_q_criterion         = true;
    pp_data.output_data.degree             = spatial_discretization == SpatialDiscretization::L2 ?
                                               this->param.degree_u :
                                               this->param.degree_u - 1;
    pp_data.output_data.write_higher_order = true;
    pp_data.output_data.write_aspect_ratio = false;
    pp_data.output_data.write_processor_id = true;

    // calculation of velocity error
    pp_data.error_data_u.time_control_data.is_active  = true;
    pp_data.error_data_u.time_control_data.start_time = start_time;
    pp_data.error_data_u.time_control_data.trigger_interval =
      convergence_study ? (end_time - start_time) / 10.0 : flow_through_time / 5.0;
    ;
    pp_data.error_data_u.analytical_solution.reset(new ManufacturedSolutionVelocity<dim>(
      height_channel_at_hill_top, length_channel, y_shift, time_period));
    pp_data.error_data_u.name                      = "velocity";
    pp_data.error_data_u.compute_convergence_table = use_manufactured_solution;
    pp_data.error_data_u.write_errors_to_file      = use_manufactured_solution;
    pp_data.error_data_u.calculate_relative_errors = false;
    pp_data.error_data_u.directory                 = this->output_parameters.directory;

    pp_data.error_data_p.time_control_data = pp_data.error_data_u.time_control_data;
    pp_data.error_data_p.time_control_data.trigger_interval =
      convergence_study ? (end_time - start_time) / 10.0 : flow_through_time / 5.0;
    ;
    pp_data.error_data_p.analytical_solution.reset(
      new ManufacturedSolutionPressure<dim>(length_channel, time_period));
    pp_data.error_data_p.name                      = "pressure";
    pp_data.error_data_p.compute_convergence_table = use_manufactured_solution;
    pp_data.error_data_p.write_errors_to_file      = use_manufactured_solution;
    pp_data.error_data_p.calculate_relative_errors = false;
    pp_data.error_data_p.directory                 = this->output_parameters.directory;

    MyPostProcessorData<dim> my_pp_data;
    my_pp_data.pp_data = pp_data;

    // line plot data: calculate statistics along lines
    my_pp_data.line_plot_data.directory = this->output_parameters.directory;

    // mean velocity
    std::shared_ptr<Quantity> quantity_velocity;
    quantity_velocity.reset(new Quantity());
    quantity_velocity->type = QuantityType::Velocity;

    std::shared_ptr<Quantity> quantity_pressure;
    quantity_pressure.reset(new Quantity());
    quantity_pressure->type = QuantityType::Pressure;

    // Reynolds stresses
    std::shared_ptr<Quantity> quantity_reynolds;
    quantity_reynolds.reset(new Quantity());
    quantity_reynolds->type = QuantityType::ReynoldsStresses;

    // dissipation rate epsilon
    auto quantity_dissipation       = std::make_shared<QuantityDissipation<dim>>();
    quantity_dissipation->type      = QuantityType::Dissipation;
    quantity_dissipation->viscosity = viscosity;

    auto quantity_skin_friction_bulk       = std::make_shared<QuantitySkinFriction<dim>>();
    quantity_skin_friction_bulk->type      = QuantityType::SkinFriction;
    quantity_skin_friction_bulk->viscosity = viscosity;
    // dummy directions (never used away from walls)
    quantity_skin_friction_bulk->tangent_vector[0] = 1.0;
    quantity_skin_friction_bulk->normal_vector[1]  = 0.0;

    // lines
    std::shared_ptr<LineHomogeneousAveraging<dim>> vel_0, vel_005, vel_05, vel_1, vel_2, vel_3,
      vel_4, vel_5, vel_6, vel_7, vel_8, vel_9, vel_10;
    vel_0.reset(new LineHomogeneousAveraging<dim>());
    vel_005.reset(new LineHomogeneousAveraging<dim>());
    vel_05.reset(new LineHomogeneousAveraging<dim>());
    vel_1.reset(new LineHomogeneousAveraging<dim>());
    vel_2.reset(new LineHomogeneousAveraging<dim>());
    vel_3.reset(new LineHomogeneousAveraging<dim>());
    vel_4.reset(new LineHomogeneousAveraging<dim>());
    vel_5.reset(new LineHomogeneousAveraging<dim>());
    vel_6.reset(new LineHomogeneousAveraging<dim>());
    vel_7.reset(new LineHomogeneousAveraging<dim>());
    vel_8.reset(new LineHomogeneousAveraging<dim>());
    vel_9.reset(new LineHomogeneousAveraging<dim>());
    vel_10.reset(new LineHomogeneousAveraging<dim>());

    vel_0->average_homogeneous_direction   = true;
    vel_005->average_homogeneous_direction = true;
    vel_05->average_homogeneous_direction  = true;
    vel_1->average_homogeneous_direction   = true;
    vel_2->average_homogeneous_direction   = true;
    vel_3->average_homogeneous_direction   = true;
    vel_4->average_homogeneous_direction   = true;
    vel_5->average_homogeneous_direction   = true;
    vel_6->average_homogeneous_direction   = true;
    vel_7->average_homogeneous_direction   = true;
    vel_8->average_homogeneous_direction   = true;
    vel_9->average_homogeneous_direction   = true;
    vel_10->average_homogeneous_direction  = true;

    vel_0->averaging_direction   = 2;
    vel_005->averaging_direction = 2;
    vel_05->averaging_direction  = 2;
    vel_1->averaging_direction   = 2;
    vel_2->averaging_direction   = 2;
    vel_3->averaging_direction   = 2;
    vel_4->averaging_direction   = 2;
    vel_5->averaging_direction   = 2;
    vel_6->averaging_direction   = 2;
    vel_7->averaging_direction   = 2;
    vel_8->averaging_direction   = 2;
    vel_9->averaging_direction   = 2;
    vel_10->averaging_direction  = 2;

    // begin and end points of all lines
    {
      double constexpr eps     = 1.e-12;
      double const H           = height_hill;
      double const L           = length_channel;
      double const top         = height_hill + height_channel_at_hill_top - eps;
      double const flat_bottom = height_hill + eps;
      vel_0->begin             = dealii::Point<dim>(0.0 * H, H + f(0.0 * H, H, L) + eps, 0);
      vel_0->end               = dealii::Point<dim>(0.0 * H, top, 0);
      vel_005->begin           = dealii::Point<dim>(0.05 * H, H + f(0.05 * H, H, L) + eps, 0);
      vel_005->end             = dealii::Point<dim>(0.05 * H, top, 0);
      vel_05->begin            = dealii::Point<dim>(0.5 * H, H + f(0.5 * H, H, L) + eps, 0);
      vel_05->end              = dealii::Point<dim>(0.5 * H, top, 0);
      vel_1->begin             = dealii::Point<dim>(1 * H, H + f(1 * H, H, L) + eps, 0);
      vel_1->end               = dealii::Point<dim>(1 * H, top, 0);
      vel_2->begin             = dealii::Point<dim>(2 * H, H + f(2 * H, H, L) + eps, 0);
      vel_2->end               = dealii::Point<dim>(2 * H, top, 0);
      vel_3->begin             = dealii::Point<dim>(3 * H, H + f(3 * H, H, L) + eps, 0);
      vel_3->end               = dealii::Point<dim>(3 * H, top, 0);
      vel_4->begin             = dealii::Point<dim>(4 * H, H + f(4 * H, H, L) + eps, 0);
      vel_4->end               = dealii::Point<dim>(4 * H, top, 0);
      vel_5->begin             = dealii::Point<dim>(5 * H, H + f(5 * H, H, L) + eps, 0);
      vel_5->end               = dealii::Point<dim>(5 * H, top, 0);
      vel_6->begin             = dealii::Point<dim>(6 * H, H + f(6 * H, H, L) + eps, 0);
      vel_6->end               = dealii::Point<dim>(6 * H, top, 0);
      vel_7->begin             = dealii::Point<dim>(7 * H, H + f(7 * H, H, L) + eps, 0);
      vel_7->end               = dealii::Point<dim>(7 * H, top, 0);
      vel_8->begin             = dealii::Point<dim>(8 * H, H + f(8 * H, H, L) + eps, 0);
      vel_8->end               = dealii::Point<dim>(8 * H, top, 0);

      vel_9->begin = dealii::Point<dim>(0 * H, top, 0);
      vel_9->end   = dealii::Point<dim>(9 * H, top, 0);

      vel_10->begin    = dealii::Point<dim>(0 * H, flat_bottom, 0);
      vel_10->end      = dealii::Point<dim>(9 * H, flat_bottom, 0);
      vel_10->manifold = std::make_shared<PeriodicHillManifold<dim>>(H,
                                                                     L,
                                                                     height_channel_at_hill_top,
                                                                     grid_stretch_factor);
    }

    // set the number of points along the lines
    vel_0->n_points   = points_per_line;
    vel_005->n_points = points_per_line;
    vel_05->n_points  = points_per_line;
    vel_1->n_points   = points_per_line;
    vel_2->n_points   = points_per_line;
    vel_3->n_points   = points_per_line;
    vel_4->n_points   = points_per_line;
    vel_5->n_points   = points_per_line;
    vel_6->n_points   = points_per_line;
    vel_7->n_points   = points_per_line;
    vel_8->n_points   = points_per_line;
    vel_9->n_points   = points_per_line;
    vel_10->n_points  = points_per_line;

    // set the quantities that we want to compute along the lines
    vel_0->quantities.push_back(quantity_velocity);
    vel_0->quantities.push_back(quantity_reynolds);
    vel_0->quantities.push_back(quantity_dissipation);
    vel_0->quantities.push_back(quantity_skin_friction_bulk);
    vel_005->quantities.push_back(quantity_velocity);
    vel_005->quantities.push_back(quantity_reynolds);
    vel_005->quantities.push_back(quantity_dissipation);
    vel_005->quantities.push_back(quantity_skin_friction_bulk);
    vel_05->quantities.push_back(quantity_velocity);
    vel_05->quantities.push_back(quantity_reynolds);
    vel_05->quantities.push_back(quantity_dissipation);
    vel_05->quantities.push_back(quantity_skin_friction_bulk);
    vel_1->quantities.push_back(quantity_velocity);
    vel_1->quantities.push_back(quantity_reynolds);
    vel_1->quantities.push_back(quantity_dissipation);
    vel_1->quantities.push_back(quantity_skin_friction_bulk);
    vel_2->quantities.push_back(quantity_velocity);
    vel_2->quantities.push_back(quantity_reynolds);
    vel_2->quantities.push_back(quantity_dissipation);
    vel_2->quantities.push_back(quantity_skin_friction_bulk);
    vel_3->quantities.push_back(quantity_velocity);
    vel_3->quantities.push_back(quantity_reynolds);
    vel_3->quantities.push_back(quantity_dissipation);
    vel_3->quantities.push_back(quantity_skin_friction_bulk);
    vel_4->quantities.push_back(quantity_velocity);
    vel_4->quantities.push_back(quantity_reynolds);
    vel_4->quantities.push_back(quantity_dissipation);
    vel_4->quantities.push_back(quantity_skin_friction_bulk);
    vel_5->quantities.push_back(quantity_velocity);
    vel_5->quantities.push_back(quantity_reynolds);
    vel_5->quantities.push_back(quantity_dissipation);
    vel_5->quantities.push_back(quantity_skin_friction_bulk);
    vel_6->quantities.push_back(quantity_velocity);
    vel_6->quantities.push_back(quantity_reynolds);
    vel_6->quantities.push_back(quantity_dissipation);
    vel_6->quantities.push_back(quantity_skin_friction_bulk);
    vel_7->quantities.push_back(quantity_velocity);
    vel_7->quantities.push_back(quantity_reynolds);
    vel_7->quantities.push_back(quantity_dissipation);
    vel_7->quantities.push_back(quantity_skin_friction_bulk);
    vel_8->quantities.push_back(quantity_velocity);
    vel_8->quantities.push_back(quantity_reynolds);
    vel_8->quantities.push_back(quantity_dissipation);
    vel_8->quantities.push_back(quantity_skin_friction_bulk);

    vel_9->quantities.push_back(quantity_velocity);
    vel_9->quantities.push_back(quantity_reynolds);
    vel_9->quantities.push_back(quantity_pressure);
    auto quantity_skin_friction_top               = std::make_shared<QuantitySkinFriction<dim>>();
    quantity_skin_friction_top->tangent_vector[0] = -1;
    quantity_skin_friction_top->normal_vector[1]  = 1;
    quantity_skin_friction_top->viscosity         = viscosity;
    vel_9->quantities.push_back(quantity_skin_friction_top);

    vel_10->quantities.push_back(quantity_velocity);
    vel_10->quantities.push_back(quantity_reynolds);
    vel_10->quantities.push_back(quantity_pressure);
    auto quantity_skin_friction_bottom = std::make_shared<QuantitySkinFriction<dim>>();
    quantity_skin_friction_bottom->tangent_vector[0] = 1;
    quantity_skin_friction_bottom->normal_vector[1]  = -1;
    quantity_skin_friction_bottom->viscosity         = viscosity;
    vel_10->quantities.push_back(quantity_skin_friction_bottom);

    // set line names
    vel_0->name   = this->output_parameters.filename + "_x_0";
    vel_005->name = this->output_parameters.filename + "_x_005";
    vel_05->name  = this->output_parameters.filename + "_x_05";
    vel_1->name   = this->output_parameters.filename + "_x_1";
    vel_2->name   = this->output_parameters.filename + "_x_2";
    vel_3->name   = this->output_parameters.filename + "_x_3";
    vel_4->name   = this->output_parameters.filename + "_x_4";
    vel_5->name   = this->output_parameters.filename + "_x_5";
    vel_6->name   = this->output_parameters.filename + "_x_6";
    vel_7->name   = this->output_parameters.filename + "_x_7";
    vel_8->name   = this->output_parameters.filename + "_x_8";
    vel_9->name   = this->output_parameters.filename + "_top";
    vel_10->name  = this->output_parameters.filename + "_bottom";

    // insert lines
    my_pp_data.line_plot_data.lines.push_back(vel_0);
    my_pp_data.line_plot_data.lines.push_back(vel_005);
    my_pp_data.line_plot_data.lines.push_back(vel_05);
    my_pp_data.line_plot_data.lines.push_back(vel_1);
    my_pp_data.line_plot_data.lines.push_back(vel_2);
    my_pp_data.line_plot_data.lines.push_back(vel_3);
    my_pp_data.line_plot_data.lines.push_back(vel_4);
    my_pp_data.line_plot_data.lines.push_back(vel_5);
    my_pp_data.line_plot_data.lines.push_back(vel_6);
    my_pp_data.line_plot_data.lines.push_back(vel_7);
    my_pp_data.line_plot_data.lines.push_back(vel_8);
    my_pp_data.line_plot_data.lines.push_back(vel_9);
    my_pp_data.line_plot_data.lines.push_back(vel_10);

    my_pp_data.line_plot_data.time_control_data_statistics.time_control_data.is_active =
      calculate_statistics;
    my_pp_data.line_plot_data.time_control_data_statistics.time_control_data.start_time =
      sample_start_time;
    my_pp_data.line_plot_data.time_control_data_statistics.time_control_data.end_time = end_time;
    my_pp_data.line_plot_data.time_control_data_statistics.time_control_data
      .trigger_every_time_steps = sample_every_timesteps;
    my_pp_data.line_plot_data.time_control_data_statistics
      .write_preliminary_results_every_nth_time_step = sample_every_timesteps * 2000;

    // calculation of flow rate (use volume-based computation)
    my_pp_data.mean_velocity_data.calculate = true;
    my_pp_data.mean_velocity_data.directory = this->output_parameters.directory;
    my_pp_data.mean_velocity_data.filename  = this->output_parameters.filename + "_flow_rate";
    dealii::Tensor<1, dim, double> direction;
    direction[0]                                = 1.0;
    my_pp_data.mean_velocity_data.direction     = direction;
    my_pp_data.mean_velocity_data.write_to_file = true;

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new MyPostProcessor<dim, Number>(
      my_pp_data, this->mpi_comm, length_channel, *flow_rate_controller));

    return pp;
  }

  // Reynolds number, viscosity, bulk velocity

  bool   inviscid = false;
  double Re       = 5600.0; // 700, 1400, 5600, 10595, 19000

  // The undeformed box occupies the region
  // [0, length] x [height_hill, height_hill+height_channel_at_hill_top] x [-width, width],
  // while after applying the mapping, the lower bottom of the box (lying at y = height_hill) is
  // mapped in negative y direction to form the classical periodic hill domain, with the
  // minimum position in y direction being y=0.
  static double constexpr height_hill                = 0.028;
  double width_channel                               = 4.5 * height_hill;
  static double constexpr length_channel             = 9.0 * height_hill;
  static double constexpr height_channel_at_hill_top = 2.036 * height_hill;
  std::array<unsigned int, 3> coarse_mesh_refinements{{2, 1, 1}};

  // The manufactured solution is defined in a channel domain centered around the origin with
  // respect to the y coordinate. This compensates for the offset of the constructed domain.
  static double constexpr y_shift = height_hill + 0.5 * height_channel_at_hill_top;

  // For temporal convergence studies, we increase the frequency to increase the temporal error.
  static bool constexpr convergence_study         = true;
  static bool constexpr spatial_convergence_study = true;
  static bool constexpr temporal_convergence_study =
    convergence_study and not spatial_convergence_study;
  static double constexpr time_period =
    2.0 * dealii::numbers::PI * (spatial_convergence_study ? 1.0 : 1e-4);

  static double constexpr bulk_velocity = 5.6218;
  double target_flow_rate = bulk_velocity * width_channel * height_channel_at_hill_top;
  static double constexpr flow_through_time = length_channel / bulk_velocity;

  // RE_H = u_b * height_hill / nu
  double viscosity = bulk_velocity * height_hill / Re;

  // flow rate controller
  std::shared_ptr<FlowRateController> flow_rate_controller;

  // start and end time
  double const start_time         = 0.0;
  double       end_time_multiples = 10.0;
  double       end_time = temporal_convergence_study ? 0.1 : end_time_multiples * flow_through_time;

  // compute convergence study else execute benchmark
  bool use_manufactured_solution = false;

  // grid
  bool   consider_box_distort = false; // distort the box grid before mapping
  bool   consider_mapping     = true;  // map the box to give the classic periodic hill geometry
  double grid_stretch_factor  = 1.6;

  // dicretization
  TemporalDiscretization temporal_discretization = TemporalDiscretization::Undefined;
  TriangulationType      triangulation_type      = TriangulationType::Distributed;
  SpatialDiscretization  spatial_discretization  = SpatialDiscretization::L2;

  // postprocessing

  // restart
  bool        write_restart         = false;
  bool        read_restart          = false;
  double      restart_interval_time = 8.0 * flow_through_time;
  std::string restart_directory     = "./output/";

  // sampling
  bool         calculate_statistics        = true;
  unsigned int sample_start_time_multiples = 0.0;
  double       sample_start_time      = double(sample_start_time_multiples) * flow_through_time;
  double       sample_end_time        = end_time;
  unsigned int sample_every_timesteps = 1;
  unsigned int points_per_line        = 40;
};

} // namespace IncNS

} // namespace ExaDG

#include <exadg/incompressible_navier_stokes/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_PERIODIC_HILL_H_ */
