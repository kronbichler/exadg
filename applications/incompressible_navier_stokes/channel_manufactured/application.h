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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_CHANNEL_MANUFACTURED_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_CHANNEL_MANUFACTURED_H_

namespace ExaDG
{
namespace IncNS
{
/*
 * Manufactured solution for incompressible flow of a Newtonian fluid in a channel of height H,
 * width W, and length L, where the convective term may be disabled
 * u     ... velocity vector
 * p     ... kinematic pressure
 * nu    ... kinematic viscosity
 * f     ... body force vector
 *
 * d/dt(u) + (grad(u)) * u + grad(p) - nu * div(grad(u)) = f
 *
 * In 3D, we derive a solution by selecting the stream function
 *
 * psi = cos(a*x) * cos^2(b*y) * cos(t),
 *
 * u1 = d/dy psi = - 2 * b * cos(a*x) * cos(b*y) * sin(b*y) * cos(t),
 *
 * u2 = -d/dx psi = a * sin(a*x) * cos^2(b*y) * cos(t),
 *
 * u3 = 0,
 *
 * p  = cos(a*x) * cos(t),
 *
 * where
 *
 * a = 2*pi/L, b = pi/H
 *
 * which is periodic in [0, length], i.e., the channel's longitudinal axis, constant in z, and
 * fulfills no-slip conditions at
 * y = +-H/2.
 *
 * The force vector follows from the momentum balance equation.
 */
template<int dim>
class AnalyticalSolutionVelocity : public dealii::Function<dim>
{
public:
  AnalyticalSolutionVelocity(double const & height, double const & length)
    : dealii::Function<dim>(dim /* n_components */, 0.0), height(height), length(length)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const final
  {
    double const a = 2.0 * dealii::numbers::PI / length;
    double const b = dealii::numbers::PI / height;

    double const t      = this->get_time();
    double const x      = p[0];
    double const y      = p[1];
    double const sin_ax = std::sin(a * x);
    double const sin_by = std::sin(b * y);
    double const cos_ax = std::cos(a * x);
    double const cos_by = std::cos(b * y);
    double const cos_t  = std::cos(t);

    if(component == 0)
      return -2.0 * b * cos_ax * cos_by * sin_by * cos_t;
    else if(component == 1)
      return a * sin_ax * dealii::Utilities::fixed_power<2>(cos_by) * cos_t;
    else
      return 0.0;
  }

private:
  double const height;
  double const length;
};


template<int dim>
class AnalyticalSolutionPressure : public dealii::Function<dim>
{
public:
  AnalyticalSolutionPressure(double const & length) : dealii::Function<dim>(1, 0.0), length(length)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const /*component*/) const final
  {
    double const a = 2.0 * dealii::numbers::PI / length;

    double const t      = this->get_time();
    double const x      = p[0];
    double const cos_ax = std::cos(a * x);
    double const cos_t  = std::cos(t);

    return cos_ax * cos_t;
  }

private:
  double const length;
};


template<int dim>
class RightHandSide : public dealii::Function<dim>
{
public:
  RightHandSide(bool const     include_convective_term,
                double const & height,
                double const & length,
                double const & kinematic_viscosity)
    : dealii::Function<dim>(dim, 0.0),
      include_convective_term(include_convective_term),
      height(height),
      length(length),
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

    double const t      = this->get_time();
    double const x      = p[0];
    double const y      = p[1];
    double const sin_ax = std::sin(a * x);
    double const sin_by = std::sin(b * y);
    double const cos_ax = std::cos(a * x);
    double const cos_by = std::cos(b * y);
    double const sin_t  = std::sin(t);
    double const cos_t  = std::cos(t);

    double const u1 = -2.0 * b * cos_ax * cos_by * sin_by * cos_t;

    double const du1_dt = 2.0 * b * cos_ax * cos_by * sin_by * sin_t;
    double const du1_dx = 2.0 * b * a * sin_ax * cos_by * sin_by * cos_t;
    double const du1_dy = 2.0 * b * b * cos_ax * cos_t * (sin_by * sin_by - cos_by * cos_by);

    double const du1_dxx = 2.0 * b * a * a * cos_ax * cos_by * sin_by * cos_t;
    double const du1_dyy = 8.0 * b * b * b * cos_ax * cos_t * cos_by * sin_by;

    double const u2 = a * sin_ax * dealii::Utilities::fixed_power<2>(cos_by) * cos_t;

    double const du2_dt = -a * sin_ax * dealii::Utilities::fixed_power<2>(cos_by) * sin_t;
    double const du2_dx = a * a * cos_ax * dealii::Utilities::fixed_power<2>(cos_by) * cos_t;
    double const du2_dy = -2.0 * a * b * sin_ax * cos_by * sin_by * cos_t;

    double const du2_dxx = -a * a * a * sin_ax * dealii::Utilities::fixed_power<2>(cos_by) * cos_t;
    double const du2_dyy = 2.0 * a * b * b * sin_ax * cos_t * (sin_by * sin_by - cos_by * cos_by);

    // p = cos_ax * cos_t;
    double const dp_dx = -a * sin_ax * cos_t;
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
      prm.add_parameter("EndTime", end_time, "Simulation end time", dealii::Patterns::Double());
      prm.add_parameter("Viscosity", viscosity, "Kinematic viscosity", dealii::Patterns::Double());
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

    // need to recompute the width, since we make it dependent on the number of elements in z
    // direction, or rather, the ratio between elements in x and z direction
    width = length * coarse_mesh_refinements[2] / coarse_mesh_refinements[0];
  }

  void
  set_parameters() final
  {
    // MATHEMATICAL MODEL
    this->param.problem_type                = ProblemType::Unsteady;
    this->param.equation_type               = EquationType::NavierStokes;
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
    this->param.adaptive_time_stepping          = false;
    this->param.max_velocity                    = 1.0;
    this->param.cfl                             = 0.5;
    this->param.cfl_exponent_fe_degree_velocity = 1.5;
    this->param.time_step_size                  = 1.0e-2;
    this->param.order_time_integrator           = order_time_integrator;
    this->param.start_with_low_order            = false;

    // output of solver information
    this->param.solver_info_data.interval_time = (end_time - start_time) / 100.0;

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

    this->param.restart_data.rpe_rtree_level            = 3;
    this->param.restart_data.rpe_tolerance_unit_cell    = 1e-6;
    this->param.restart_data.rpe_enforce_unique_mapping = false;

    // PROJECTION METHODS

    // pressure Poisson equation
    this->param.solver_data_pressure_poisson =
      SolverData(1000, 1.e-12, 1.e-6, LinearSolver::CG, 100);
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

    this->param.solver_data_momentum    = SolverData(1000, 1.e-12, 1.e-6, LinearSolver::CG);
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
      p_1[1] = -height / 2.0;
      if constexpr(dim == 3)
        p_1[2] = -width / 2.0;

      dealii::Point<dim> p_2;
      p_2[0] = length;
      p_2[1] = height / 2.0;
      if constexpr(dim == 3)
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
        if constexpr(dim == 3)
        {
          // left element
          if(cell->at_boundary(4))
            cell->face(4)->set_all_boundary_ids(2 + 10);
          if(cell->at_boundary(5))
            cell->face(5)->set_all_boundary_ids(3 + 10);
        }
      }

      dealii::GridTools::collect_periodic_faces(tria, 0 + 10, 1 + 10, 0, periodic_face_pairs);
      if constexpr(dim == 3)
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
        fe_values.reinit(cell);

        std::vector<dealii::Point<dim>> points_moved(fe_values.n_quadrature_points);
        for(unsigned int i = 0; i < fe_values.n_quadrature_points; ++i)
        {
          // need to adjust for hierarchic numbering of
          // dealii::MappingQCache
          points_moved[i] = fe_values.quadrature_point(hierarchic_to_lexicographic_numbering[i]);
        }

        return points_moved;
      });

    grid.mapping_function = [&](typename dealii::Triangulation<dim>::cell_iterator const & cell)
      -> std::vector<dealii::Point<dim>> {
      std::vector<dealii::Point<dim>> points_moved(cell->n_vertices());
      for(unsigned int i = 0; i < cell->n_vertices(); ++i)
      {
        // need to adjust for hierarchic numbering of
        // dealii::MappingQCache
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

    // no-slip condition on upper and lower channel walls
    this->boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(0, new dealii::Functions::ZeroFunction<dim>(dim)));

    // zero Neumann condition on velocity Dirichlet boundaries
    this->boundary_descriptor->pressure->neumann_bc.insert(0);
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution_velocity.reset(
      new AnalyticalSolutionVelocity<dim>(height, length));
    this->field_functions->initial_solution_pressure.reset(
      new AnalyticalSolutionPressure<dim>(length));

    this->field_functions->analytical_solution_pressure.reset(
      new AnalyticalSolutionPressure<dim>(length));
    this->field_functions->analytical_solution_velocity.reset(
      new AnalyticalSolutionVelocity<dim>(height, length));

    bool const include_convective_term = this->param.equation_type == EquationType::NavierStokes;
    this->field_functions->right_hand_side.reset(
      new RightHandSide<dim>(include_convective_term, height, length, viscosity));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.time_control_data.is_active        = this->output_parameters.write;
    pp_data.output_data.time_control_data.start_time       = start_time;
    pp_data.output_data.time_control_data.trigger_interval = (end_time - start_time) / 100.0;
    pp_data.output_data.directory          = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename           = this->output_parameters.filename;
    pp_data.output_data.write_divergence   = true;
    pp_data.output_data.degree             = spatial_discretization == SpatialDiscretization::L2 ?
                                               this->param.degree_u :
                                               this->param.degree_u - 1;
    pp_data.output_data.write_higher_order = true;
    pp_data.output_data.write_processor_id = true;
    pp_data.output_data.write_boundary_IDs = true;
    pp_data.output_data.write_grid         = false;

    // calculation of velocity error
    pp_data.error_data_u.compute_convergence_table          = true;
    pp_data.error_data_u.write_errors_to_file               = true;
    pp_data.error_data_u.time_control_data.is_active        = true;
    pp_data.error_data_u.time_control_data.start_time       = start_time;
    pp_data.error_data_u.time_control_data.trigger_interval = (end_time - start_time);
    pp_data.error_data_u.analytical_solution.reset(
      new AnalyticalSolutionVelocity<dim>(height, length));
    pp_data.error_data_u.calculate_relative_errors = true;
    pp_data.error_data_u.name                      = "velocity";
    pp_data.error_data_u.directory                 = this->output_parameters.directory;

    // ... pressure error
    pp_data.error_data_p.write_errors_to_file               = true;
    pp_data.error_data_p.time_control_data.is_active        = true;
    pp_data.error_data_p.time_control_data.start_time       = start_time;
    pp_data.error_data_p.time_control_data.trigger_interval = (end_time - start_time);
    pp_data.error_data_p.analytical_solution.reset(new AnalyticalSolutionPressure<dim>(length));
    pp_data.error_data_p.calculate_relative_errors = true;
    pp_data.error_data_p.name                      = "pressure";
    pp_data.error_data_p.directory                 = this->output_parameters.directory;

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  // Geometric parameters.
  double                      width; // updated in `parse_parameters()` for good aspect ratio
  double const                length = 4.0;
  double const                height = 1.0;
  std::array<unsigned int, 3> coarse_mesh_refinements{{4, 1, 1}};

  double viscosity = 1e-2;

  // start and end time
  double const start_time = 0.0;
  double       end_time   = 1.0;

  // dicretization
  TemporalDiscretization temporal_discretization = TemporalDiscretization::Undefined;
  TriangulationType      triangulation_type      = TriangulationType::Distributed;
  SpatialDiscretization  spatial_discretization  = SpatialDiscretization::L2;
  unsigned int           order_time_integrator   = 2;

  // postprocessing

  // restart
  bool        write_restart         = false;
  bool        read_restart          = false;
  double      restart_interval_time = (end_time - start_time) * 0.8;
  std::string restart_directory     = "./output/";
};

} // namespace IncNS

} // namespace ExaDG

#include <exadg/incompressible_navier_stokes/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_CHANNEL_MANUFACTURED_H_ */
