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
template<int dim>
class InitialSolutionVelocity : public dealii::Function<dim>
{
public:
  InitialSolutionVelocity(double const bulk_velocity, double const H, double const height)
    : dealii::Function<dim>(dim, 0.0), bulk_velocity(bulk_velocity), H(H), height(height)
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
      if(p[1] > H and p[1] < (H + height))
        result = bulk_velocity * (p[1] - H) * ((H + height) - p[1]) / std::pow(height / 2.0, 2.0);

      // add some random perturbations
      result *= (1.0 + 0.1 * (((double)rand() / RAND_MAX - 0.5) / 0.5));
    }

    return result;
  }

private:
  double const bulk_velocity, H, height;
};

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
      prm.add_parameter("WriteRestart", write_restart, "Should restart files be written?");
      prm.add_parameter("ReadRestart", read_restart, "Is this a restarted simulation?");
      prm.add_parameter("RestartDirectory",
                        restart_directory,
                        "Directory with restart data to start the simulation from.");
      prm.add_parameter("RestartIntervalTime",
                        restart_interval_time,
                        "Time between writes of restart data in multiples of flow-through time.");
      prm.add_parameter("TriangulationType", triangulation_type, "Type of triangulation");
      prm.add_parameter("TemporalDiscretization",
                        temporal_discretization,
                        "Temporal discretization");
      prm.add_parameter("SpatialDiscretization", spatial_discretization, "Spatial discretization");
      prm.add_parameter("Inviscid", inviscid, "Is this an inviscid simulation?");
      prm.add_parameter("ReynoldsNumber", Re, "Reynolds number (ignored if Inviscid = true)");
      prm.add_parameter("EndTime",
                        end_time_multiples,
                        "End time in multiples of flow-through time.",
                        dealii::Patterns::Double(0.0, 1000.0));
      prm.add_parameter("GridStretchFactor",
                        grid_stretch_factor,
                        "Factor describing grid stretching in vertical direction.");
      prm.add_parameter("CalculateStatistics",
                        calculate_statistics,
                        "Decides whether statistics are calculated.");
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
    viscosity = inviscid ? 0.0 : bulk_velocity * H / Re;

    // depend on values defined in input file
    end_time              = end_time_multiples * flow_through_time;
    sample_start_time     = double(sample_start_time_multiples) * flow_through_time;
    restart_interval_time = restart_interval_time * flow_through_time;

    // sample end time is equal to end time, which is read from the input file
    sample_end_time = end_time;

    // need to recompute the width, since we make it dependent on the number of elements in z
    // direction, or rather, the ratio between elements in x and z direction
    width = length * coarse_mesh_refinements[2] / coarse_mesh_refinements[0];

    // recompute target flow rate as it depends on the width
    target_flow_rate = bulk_velocity * width * height;

    // finally refresh the flow rate controller
    flow_rate_controller.reset(
      new FlowRateController(bulk_velocity,
                             target_flow_rate,
                             H,
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
    this->param.solver_type                     = SolverType::Unsteady;
    this->param.temporal_discretization         = temporal_discretization;
    this->param.treatment_of_convective_term    = TreatmentOfConvectiveTerm::Explicit;
    this->param.calculation_of_time_step_size   = TimeStepCalculation::CFL;
    this->param.adaptive_time_stepping          = true;
    this->param.max_velocity                    = bulk_velocity;
    this->param.cfl                             = 0.32; // 0.375;
    this->param.cfl_exponent_fe_degree_velocity = 1.5;
    this->param.time_step_size                  = 1.0e-1;
    this->param.order_time_integrator           = 3;
    this->param.start_with_low_order            = read_restart ? false : true;

    // output of solver information
    this->param.solver_info_data.interval_time = flow_through_time / 10.0;

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

    // pressure Poisson equation
    this->param.solver_data_pressure_poisson =
      SolverData(1000, 1.e-12, 1.e-5, LinearSolver::CG, 100);
    this->param.preconditioner_pressure_poisson      = PreconditionerPressurePoisson::Multigrid;
    this->param.multigrid_data_pressure_poisson.type = MultigridType::cphMG;
    this->param.multigrid_data_pressure_poisson.coarse_problem.solver =
      MultigridCoarseGridSolver::Chebyshev;
    this->param.multigrid_data_pressure_poisson.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::PointJacobi;

    // projection step
    this->param.solver_data_projection    = SolverData(1000, 1.e-12, 1.e-6, LinearSolver::CG);
    this->param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;
    this->param.update_preconditioner_projection = true;


    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulations
    this->param.order_extrapolation_pressure_nbc =
      this->param.order_time_integrator <= 2 ? this->param.order_time_integrator : 2;

    this->param.solver_data_momentum    = SolverData(1000, 1.e-12, 1.e-8, LinearSolver::CG);
    this->param.preconditioner_momentum = spatial_discretization == SpatialDiscretization::L2 ?
                                            MomentumPreconditioner::InverseMassMatrix :
                                            MomentumPreconditioner::PointJacobi;

    this->param.inverse_mass_operator.implementation_type = InverseMassType::GlobalKrylovSolver;
    this->param.inverse_mass_operator.preconditioner      = PreconditionerMass::PointJacobi;
    this->param.inverse_mass_operator.solver_data = SolverData(1000, 1e-12, 1e-4, LinearSolver::CG);

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
      p_1[1] = H;
      if(dim == 3)
        p_1[2] = -width / 2.0;

      dealii::Point<dim> p_2;
      p_2[0] = length;
      p_2[1] = H + height;
      if(dim == 3)
        p_2[2] = width / 2.0;

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
        PeriodicHillManifold<dim> manifold(H, length, height, grid_stretch_factor);
        fe_values.reinit(cell);

        std::vector<dealii::Point<dim>> points_moved(fe_values.n_quadrature_points);
        for(unsigned int i = 0; i < fe_values.n_quadrature_points; ++i)
        {
          // need to adjust for hierarchic numbering of
          // dealii::MappingQCache
          if(consider_mapping)
            points_moved[i] = manifold.push_forward(
              fe_values.quadrature_point(hierarchic_to_lexicographic_numbering[i]));
          else
            points_moved[i] = fe_values.quadrature_point(hierarchic_to_lexicographic_numbering[i]);
        }

        return points_moved;
      });

    grid.mapping_function = [&](typename dealii::Triangulation<dim>::cell_iterator const & cell)
      -> std::vector<dealii::Point<dim>> {
      PeriodicHillManifold<dim>       manifold(H, length, height, grid_stretch_factor);
      std::vector<dealii::Point<dim>> points_moved(cell->n_vertices());
      for(unsigned int i = 0; i < cell->n_vertices(); ++i)
      {
        // need to adjust for hierarchic numbering of
        // dealii::MappingQCache
        if(consider_mapping)
          points_moved[i] = manifold.push_forward(cell->vertex(i));
        else
          points_moved[i] = cell->vertex(i);
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
    this->field_functions->initial_solution_velocity.reset(
      new InitialSolutionVelocity<dim>(bulk_velocity, H, height));
    this->field_functions->initial_solution_pressure.reset(
      new dealii::Functions::ZeroFunction<dim>(1));
    this->field_functions->analytical_solution_pressure.reset(
      new dealii::Functions::ZeroFunction<dim>(1));
    this->field_functions->right_hand_side.reset(new RightHandSide<dim>(*flow_rate_controller));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.time_control_data.is_active        = this->output_parameters.write;
    pp_data.output_data.time_control_data.start_time       = start_time;
    pp_data.output_data.time_control_data.trigger_interval = flow_through_time / 5.0;
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
    double const eps = 1.e-12;
    vel_0->begin     = dealii::Point<dim>(0.0 * H, H + f(0.0 * H, H, length) + eps, 0);
    vel_0->end       = dealii::Point<dim>(0.0 * H, H + height - eps, 0);
    vel_005->begin   = dealii::Point<dim>(0.05 * H, H + f(0.05 * H, H, length) + eps, 0);
    vel_005->end     = dealii::Point<dim>(0.05 * H, H + height - eps, 0);
    vel_05->begin    = dealii::Point<dim>(0.5 * H, H + f(0.5 * H, H, length) + eps, 0);
    vel_05->end      = dealii::Point<dim>(0.5 * H, H + height - eps, 0);
    vel_1->begin     = dealii::Point<dim>(1 * H, H + f(1 * H, H, length) + eps, 0);
    vel_1->end       = dealii::Point<dim>(1 * H, H + height - eps, 0);
    vel_2->begin     = dealii::Point<dim>(2 * H, H + f(2 * H, H, length) + eps, 0);
    vel_2->end       = dealii::Point<dim>(2 * H, H + height - eps, 0);
    vel_3->begin     = dealii::Point<dim>(3 * H, H + f(3 * H, H, length) + eps, 0);
    vel_3->end       = dealii::Point<dim>(3 * H, H + height - eps, 0);
    vel_4->begin     = dealii::Point<dim>(4 * H, H + f(4 * H, H, length) + eps, 0);
    vel_4->end       = dealii::Point<dim>(4 * H, H + height - eps, 0);
    vel_5->begin     = dealii::Point<dim>(5 * H, H + f(5 * H, H, length) + eps, 0);
    vel_5->end       = dealii::Point<dim>(5 * H, H + height - eps, 0);
    vel_6->begin     = dealii::Point<dim>(6 * H, H + f(6 * H, H, length) + eps, 0);
    vel_6->end       = dealii::Point<dim>(6 * H, H + height - eps, 0);
    vel_7->begin     = dealii::Point<dim>(7 * H, H + f(7 * H, H, length) + eps, 0);
    vel_7->end       = dealii::Point<dim>(7 * H, H + height - eps, 0);
    vel_8->begin     = dealii::Point<dim>(8 * H, H + f(8 * H, H, length) + eps, 0);
    vel_8->end       = dealii::Point<dim>(8 * H, H + height - eps, 0);
    vel_9->begin     = dealii::Point<dim>(0 * H, H + height - eps, 0);
    vel_9->end       = dealii::Point<dim>(9 * H, H + height - eps, 0);
    vel_10->begin    = dealii::Point<dim>(0 * H, H + eps, 0);
    vel_10->end      = dealii::Point<dim>(9 * H, H + eps, 0);
    vel_10->manifold =
      std::make_shared<PeriodicHillManifold<dim>>(H, length, height, grid_stretch_factor);

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
    pp.reset(
      new MyPostProcessor<dim, Number>(my_pp_data, this->mpi_comm, length, *flow_rate_controller));

    return pp;
  }

  // Reynolds number, viscosity, bulk velocity

  bool   inviscid = false;
  double Re       = 5600.0; // 700, 1400, 5600, 10595, 19000

  double const                H      = 0.028;
  double                      width  = 4.5 * H;
  double const                length = 9.0 * H;
  double const                height = 2.036 * H;
  std::array<unsigned int, 3> coarse_mesh_refinements{{2, 1, 1}};

  double const bulk_velocity     = 5.6218;
  double       target_flow_rate  = bulk_velocity * width * height;
  double const flow_through_time = length / bulk_velocity;

  // RE_H = u_b * H / nu
  double viscosity = bulk_velocity * H / Re;

  // flow rate controller
  std::shared_ptr<FlowRateController> flow_rate_controller;

  // start and end time
  double const start_time         = 0.0;
  double       end_time_multiples = 10;
  double       end_time           = end_time_multiples * flow_through_time;

  // grid
  bool const consider_mapping    = true;
  double     grid_stretch_factor = 1.6;

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
