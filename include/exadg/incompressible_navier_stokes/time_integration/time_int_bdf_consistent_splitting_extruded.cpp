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
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/numerics/vector_tools_mean_value.h>

#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_consistent_splitting_extruded.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/laplace_operator_extruded.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/momentum_operator_rt.h>
#include <exadg/incompressible_navier_stokes/time_integration/helper_functions.h>
#include <exadg/incompressible_navier_stokes/time_integration/poisson_solver.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_consistent_splitting_extruded.h>
#include <exadg/incompressible_navier_stokes/user_interface/parameters.h>
#include <exadg/time_integration/restart.h>
#include <exadg/time_integration/time_step_calculation.h>
#include <exadg/time_integration/vector_handling.h>
#include <exadg/utilities/print_solver_results.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
TimeIntBDFConsistentSplittingExtruded<dim, Number>::TimeIntBDFConsistentSplittingExtruded(
  std::shared_ptr<Operator>                       operator_in,
  std::shared_ptr<HelpersALE<dim, Number> const>  helpers_ale_in,
  std::shared_ptr<PostProcessorInterface<Number>> postprocessor_in,
  Parameters const &                              param_in,
  MPI_Comm const &                                mpi_comm_in,
  bool const                                      is_test_in)
  : Base(operator_in, helpers_ale_in, postprocessor_in, param_in, mpi_comm_in, is_test_in),
    pde_operator(operator_in),
    velocity(this->order),
    velocity_red(4),
    velocity_matvec(4 * 2),
    pressure(5),
    pressure_matvec(5),
    convective_divergence_rhs(this->param.order_extrapolation_pressure_nbc),
    divergences(this->order),
    pressure_nbc_rhs(this->param.order_extrapolation_pressure_nbc),
    factor_cfl(-1.0),
    iterations_pressure({0, 0}),
    iterations_viscous({0, {0, 0}}),
    extra_pressure_nbc(this->param.order_extrapolation_pressure_nbc,
                       this->param.start_with_low_order),
    extra_pressure_rhs(this->param.order_extrapolation_pressure_rhs,
                       this->param.start_with_low_order)
{
  op_rt = std::make_shared<RTOperator::RaviartThomasOperatorBase<dim, Number>>();

  pde_operator->momentum_operator = op_rt;
  pde_operator->velocity_vector   = &velocity[0];
}

template<int dim, typename Number>
void
TimeIntBDFConsistentSplittingExtruded<dim, Number>::update_time_integrator_constants()
{
  // call function of base class to update the standard time integrator constants
  Base::update_time_integrator_constants();

  // update time integrator constants for extrapolation scheme of pressure Neumann bc
  extra_pressure_nbc.update(this->get_time_step_number(),
                            this->adaptive_time_stepping,
                            this->get_time_step_vector());

  extra_pressure_rhs.update(this->get_time_step_number(),
                            this->adaptive_time_stepping,
                            this->get_time_step_vector());

  // use this function to check the correctness of the time integrator constants
  //    std::cout << "Coefficients extrapolation scheme pressure NBC:" << std::endl;
  //    extra_pressure_nbc.print(this->pcout);
}

template<int dim, typename Number>
void
TimeIntBDFConsistentSplittingExtruded<dim, Number>::setup_derived()
{
}

template<int dim, typename Number>
void
TimeIntBDFConsistentSplittingExtruded<dim, Number>::write_vector_norms() const
{
  // write vector norms describing the state of the solver
  this->pcout << std::scientific << std::setprecision(8) << std::setw(20) << "\n";

  this->pcout << "VECTORS:\n";

  this->pcout << "velocity_np.l2_norm() = " << velocity_np.l2_norm() << "\n";
  for(unsigned int i = 0; i < velocity.size(); ++i)
  {
    this->pcout << "velocity[" << i << "].l2_norm() = " << velocity[i].l2_norm() << "\n";
  }
  for(unsigned int i = 0; i < velocity_red.size(); ++i)
  {
    this->pcout << "velocity_red[" << i << "].l2_norm() = " << velocity_red[i].l2_norm() << "\n";
  }
  for(unsigned int i = 0; i < velocity_matvec.size(); ++i)
  {
    this->pcout << "velocity_matvec[" << i << "].l2_norm() = " << velocity_matvec[i].l2_norm()
                << "\n";
  }

  this->pcout << "pressure_np.l2_norm() = " << pressure_np.l2_norm() << "\n";
  for(unsigned int i = 0; i < pressure.size(); ++i)
  {
    this->pcout << "pressure[" << i << "].l2_norm() = " << pressure[i].l2_norm() << "\n";
  }
  for(unsigned int i = 0; i < pressure_matvec.size(); ++i)
  {
    this->pcout << "pressure_matvec[" << i << "].l2_norm() = " << pressure_matvec[i].l2_norm()
                << "\n";
  }
  for(unsigned int i = 0; i < convective_divergence_rhs.size(); ++i)
  {
    this->pcout << "convective_divergence_rhs[" << i
                << "].l2_norm() = " << convective_divergence_rhs[i].l2_norm() << "\n";
  }
  for(unsigned int i = 0; i < divergences.size(); ++i)
  {
    this->pcout << "divergences[" << i << "].l2_norm() = " << divergences[i].l2_norm() << "\n";
  }
  for(unsigned int i = 0; i < pressure_nbc_rhs.size(); ++i)
  {
    this->pcout << "pressure_nbc_rhs[" << i << "].l2_norm() = " << pressure_nbc_rhs[i].l2_norm()
                << "\n";
  }

  this->pcout << "\n";
}

template<int dim, typename Number>
void
TimeIntBDFConsistentSplittingExtruded<dim, Number>::allocate_vectors()
{
  Base::allocate_vectors();

  // velocity
  pde_operator->initialize_vector_velocity(velocity_np);

  // pressure
  pde_operator->initialize_vector_pressure(pressure_np);
  pde_operator->initialize_vector_pressure(pressure_rhs);
  for(VectorType & vector : divergences)
    pde_operator->initialize_vector_pressure(vector);
  for(VectorType & vector : convective_divergence_rhs)
    pde_operator->initialize_vector_pressure(vector);

  // do test for matrix-free operator of optimized kind
  const std::vector<unsigned int> cell_vectorization_category =
    Helper::compute_vectorization_category(pde_operator->get_dof_handler_u().get_triangulation());
  op_rt->reinit(*pde_operator->get_mapping(),
                pde_operator->get_dof_handler_u(),
                pde_operator->get_constraint_u(),
                cell_vectorization_category,
                dealii::QGauss<1>(pde_operator->get_dof_handler_u().get_fe().degree + 1),
                this->is_test);

  op_rt->set_penalty_parameters(pde_operator->get_viscous_kernel_data().IP_factor);
  op_rt->initialize_dof_vector(solution_rt);
  op_rt->initialize_dof_vector(rhs_rt);

  op_rt_float = std::make_shared<RTOperator::RaviartThomasOperatorBase<dim, float>>();
  op_rt_float->reinit(*pde_operator->get_mapping(),
                      pde_operator->get_dof_handler_u(),
                      pde_operator->get_constraint_u(),
                      cell_vectorization_category,
                      dealii::QGauss<1>(pde_operator->get_dof_handler_u().get_fe().degree + 1),
                      this->is_test);

  op_rt_float->set_penalty_parameters(pde_operator->get_viscous_kernel_data().IP_factor);

  op_rt_float->set_parameters(1.0, 0.0);
  op_rt_float->compute_diagonal(diagonal_mass);

  preconditioner_mass.get_vector().reinit(diagonal_mass, true);
  const unsigned int local_size = diagonal_mass.locally_owned_size();
  DEAL_II_OPENMP_SIMD_PRAGMA
  for(unsigned int i = 0; i < local_size; ++i)
  {
    AssertThrow(diagonal_mass.local_element(i) > 1e-30, dealii::ExcInternalError());
    preconditioner_mass.get_vector().local_element(i) = 1. / diagonal_mass.local_element(i);
  }
  op_rt_float->set_parameters(0.0, 1.0);
  op_rt_float->compute_diagonal(diagonal_laplace);

  for(VectorType & vector : velocity)
    op_rt->initialize_dof_vector(vector);

  for(VectorType & vec : this->vec_convective_term)
    op_rt->initialize_dof_vector(vec);
  // vector not used
  this->convective_term_np.reinit(0);
  for(VectorType & vector : pressure_nbc_rhs)
    op_rt->initialize_vector_pressure_neumann_boundary(vector);

  for(auto & vector : velocity_red)
    op_rt_float->initialize_dof_vector(vector);
  for(auto & vector : velocity_matvec)
    op_rt_float->initialize_dof_vector(vector);
  op_rt_float->initialize_dof_vector(rhs_float);

  laplace_op = std::make_shared<LaplaceOperator::LaplaceOperatorDG<dim, Number>>();
  laplace_op->reinit(*pde_operator->get_mapping(),
                     pde_operator->get_dof_handler_p(),
                     pde_operator->get_constraint_p(),
                     cell_vectorization_category,
                     dealii::QGauss<1>(pde_operator->get_dof_handler_p().get_fe().degree + 1),
                     this->is_test);
  laplace_op->set_penalty_parameters(
    pde_operator->laplace_operator.get_data().kernel_data.IP_factor);

  op_rt->verify_other_cell_level_index(laplace_op->get_cell_level_index());
  op_rt->initialize_coupling_pressure(pde_operator->get_dof_handler_p().get_fe(),
                                      laplace_op->get_dof_indices());

  poisson_preconditioner = std::make_shared<LaplaceOperator::PoissonPreconditionerMG<dim, float>>(
    *pde_operator->get_mapping(),
    pde_operator->get_dof_handler_p(),
    cell_vectorization_category,
    pde_operator->get_grid().mapping_function,
    pde_operator->laplace_operator.get_data().kernel_data.IP_factor,
    this->is_test);

  for(unsigned int i = 0; i < pressure.size(); ++i)
    poisson_preconditioner->get_dg_matrix().initialize_dof_vector(pressure[i]);
  for(unsigned int i = 0; i < pressure_matvec.size(); ++i)
    poisson_preconditioner->get_dg_matrix().initialize_dof_vector(pressure_matvec[i]);
}

template<int dim, typename Number>
unsigned int
TimeIntBDFConsistentSplittingExtruded<dim, Number>::get_size_velocity() const
{
  return velocity.size();
}

template<int dim, typename Number>
unsigned int
TimeIntBDFConsistentSplittingExtruded<dim, Number>::get_size_pressure() const
{
  return pressure.size();
}

template<int dim, typename Number>
void
TimeIntBDFConsistentSplittingExtruded<dim, Number>::initialize_current_solution()
{
  if(this->param.ale_formulation)
    this->helpers_ale->move_grid(this->get_time());

  pde_operator->prescribe_initial_conditions(velocity_np, pressure_np, this->get_time());
  op_rt->copy_mf_to_this_vector(velocity_np, velocity[0]);
  velocity_red[0].copy_locally_owned_data_from(velocity[0]);

  factor_cfl = op_rt->evaluate_convective_and_divergence_term(velocity[0],
                                                              this->vec_convective_term[0],
                                                              convective_divergence_rhs[0],
                                                              divergences[0]);
  op_rt->evaluate_pressure_neumann_from_velocity(
    velocity[0],
    false,
    this->pde_operator->get_viscous_kernel_data().viscosity,
    pressure_nbc_rhs[0]);
}

template<int dim, typename Number>
void
TimeIntBDFConsistentSplittingExtruded<dim, Number>::initialize_former_multistep_dof_vectors()
{
  // note that the loop begins with i=1, since i=0 is not necessary.
  for(unsigned int i = 1; i < velocity.size(); ++i)
  {
    if(this->param.ale_formulation)
      this->helpers_ale->move_grid(this->get_previous_time(i));

    VectorType tmp_velocity, tmp_pressure;
    tmp_velocity.reinit(velocity_np);
    tmp_pressure.reinit(pressure_np);
    pde_operator->prescribe_initial_conditions(tmp_velocity,
                                               tmp_pressure,
                                               this->get_previous_time(i));

    // Copy vectors of `VectorType` to RT vector type.
    op_rt->copy_mf_to_this_vector(tmp_velocity, velocity[i]);

    // Convert `VectorType` to `VectorTypeFloat`, `pressure` has a fixed size unrelated to the
    // extrapolation/integration order. Fill the entries that correspond to `velocity`.
    pressure[i].copy_locally_owned_data_from(tmp_pressure);

    // Construct pressure Neumann data vector, which has a number of entries depending on
    // extrapolation order of the pressure for the Neumann BC, `order_extrapolation_pressure_nbc`.
    if(i < pressure_nbc_rhs.size())
      op_rt->evaluate_pressure_neumann_from_velocity(
        velocity[i],
        false,
        this->pde_operator->get_viscous_kernel_data().viscosity,
        pressure_nbc_rhs[i]);
  }
}

template<int dim, typename Number>
typename TimeIntBDFConsistentSplittingExtruded<dim, Number>::VectorType const &
TimeIntBDFConsistentSplittingExtruded<dim, Number>::get_velocity() const
{
  return velocity_np;
}

template<int dim, typename Number>
typename TimeIntBDFConsistentSplittingExtruded<dim, Number>::VectorType const &
TimeIntBDFConsistentSplittingExtruded<dim, Number>::get_velocity_np() const
{
  return velocity_np;
}

template<int dim, typename Number>
typename TimeIntBDFConsistentSplittingExtruded<dim, Number>::VectorType const &
TimeIntBDFConsistentSplittingExtruded<dim, Number>::get_pressure() const
{
  return pressure_np;
}

template<int dim, typename Number>
typename TimeIntBDFConsistentSplittingExtruded<dim, Number>::VectorType const &
TimeIntBDFConsistentSplittingExtruded<dim, Number>::get_pressure(const unsigned int i) const
{
  AssertThrow(i == 0, dealii::ExcNotImplemented());
  return pressure_np;
}

template<int dim, typename Number>
typename TimeIntBDFConsistentSplittingExtruded<dim, Number>::VectorType const &
TimeIntBDFConsistentSplittingExtruded<dim, Number>::get_pressure_np() const
{
  return pressure_np;
}

template<int dim, typename Number>
typename TimeIntBDFConsistentSplittingExtruded<dim, Number>::VectorType const &
TimeIntBDFConsistentSplittingExtruded<dim, Number>::get_velocity(unsigned int i) const
{
  AssertThrow(i == 0, dealii::ExcNotImplemented());
  return velocity_np;
}

template<int dim, typename Number>
void
TimeIntBDFConsistentSplittingExtruded<dim, Number>::set_velocity(VectorType const & velocity_in,
                                                                 unsigned int const i)
{
  // We need to copy since the vector sizes are different.
  op_rt->copy_mf_to_this_vector(velocity_in, velocity[i]);
}

template<int dim, typename Number>
void
TimeIntBDFConsistentSplittingExtruded<dim, Number>::set_pressure(VectorType const & pressure_in,
                                                                 unsigned int const i)
{
  // Convert entries in `VectorType` to entry type of `VectorTypeFloat` if necessary.
  pressure[i].copy_locally_owned_data_from(pressure_in);
}

template<int dim, typename Number>
void
TimeIntBDFConsistentSplittingExtruded<dim, Number>::do_timestep_solve()
{
  // perform the sub-steps of the consistent-splitting method
  pressure_step();

  momentum_step();

  // evaluate convective term once the final solution at time
  // t_{n+1} is known
  evaluate_convective_term();
}

template<int dim, typename Number>
void
TimeIntBDFConsistentSplittingExtruded<dim, Number>::evaluate_convective_term()
{
  dealii::Timer timer;
  timer.restart();

  if(this->param.convective_problem())
  {
    if(this->param.ale_formulation == false) // Eulerian case
    {
      factor_cfl = op_rt->evaluate_convective_and_divergence_term(solution_rt,
                                                                  this->vec_convective_term.back(),
                                                                  convective_divergence_rhs.back(),
                                                                  divergences.back());

      op_rt->evaluate_pressure_neumann_from_velocity(
        solution_rt,
        false,
        this->pde_operator->get_viscous_kernel_data().viscosity,
        pressure_nbc_rhs.back());
    }
  }

  this->timer_tree->insert({"Timeloop", "Evaluate convection"}, timer.wall_time());
}



template<int dim, typename Number>
void
TimeIntBDFConsistentSplittingExtruded<dim, Number>::pressure_step()
{
  dealii::Timer timer;
  timer.restart();

  // compute right-hand-side vector
  rhs_pressure(pressure_rhs);

  const double t_rhs = timer.wall_time();

  // extrapolate old solution to get a good initial estimate for the solver
  dealii::Timer             timer2;
  std::pair<double, double> extrapolate_accuracy(0., 0.);
  if(this->use_extrapolation)
  {
    pressure[0].copy_locally_owned_data_from(pressure_np);
    poisson_preconditioner->get_dg_matrix().vmult(pressure_matvec[0], pressure[0]);
    extrapolate_accuracy =
      compute_least_squares_fit(pressure_matvec, pressure_rhs, pressure, pressure_np);
  }
  const double t_extrapol = timer2.wall_time();
  timer2.restart();

  // solve linear system of equations
  unsigned int                 n_iter = 0;
  dealii::SolverControl        control(this->param.solver_data_pressure_poisson.max_iter,
                                extrapolate_accuracy.first *
                                  this->param.solver_data_pressure_poisson.rel_tol);
  dealii::SolverCG<VectorType> solver(control);
  solver.solve(*laplace_op, pressure_np, pressure_rhs, *poisson_preconditioner);
  n_iter = control.last_step();

  iterations_pressure.first += 1;
  iterations_pressure.second += n_iter;
  const double t_sol = timer2.wall_time();

  // special case: pressure level is undefined
  // Adjust the pressure level in order to allow a calculation of the pressure error.
  // This is necessary because otherwise the pressure solution moves away from the exact solution.
  pde_operator->adjust_pressure_level_if_undefined(pressure_np, this->get_next_time());

  // write output
  if(this->print_solver_info() and not(this->is_test))
  {
    this->pcout << std::endl
                << "Pressure step prepare: " << t_rhs << "/" << t_extrapol << " s, solve " << t_sol
                << std::endl
                << "Solve pressure step (projection reduced residual from "
                << extrapolate_accuracy.first << " to " << extrapolate_accuracy.second
                << " solve to " << control.last_value() << "):";
    print_solver_info_linear(this->pcout, n_iter, timer.wall_time());
  }

  this->timer_tree->insert({"Timeloop", "Pressure step"}, timer.wall_time());
}

template<int dim, typename Number>
void
TimeIntBDFConsistentSplittingExtruded<dim, Number>::rhs_pressure(VectorType & rhs)
{
  /*
   *  I. convective extrapolation
   */
  std::vector<Number> factors(extra_pressure_rhs.get_order());
  for(unsigned int i = 0; i < extra_pressure_rhs.get_order(); ++i)
    factors[i] = extra_pressure_rhs.get_beta(i);
  extrapolate_vectors(factors, convective_divergence_rhs, rhs);

  /*
   *  II. calculate Leray projection
   */
  if(this->param.apply_leray_projection)
  {
    factors.resize(divergences.size());
    for(unsigned int i = 0; i < divergences.size(); ++i)
      factors[i] = -this->bdf.get_alpha(i) / this->get_time_step_size();
    extrapolate_vectors_and_add(factors, divergences, rhs);
  }

  /*
   *  III. forcing term
   */
  laplace_op->evaluate_add_divergence_body_force(
    this->get_next_time(), *pde_operator->get_field_functions()->right_hand_side, rhs);

  /*
   *  IV. handle consistent boundary condition
   */
  /*
   *  IV.1 compute curl-curl term by extrapolating the prepared values from
   *  previous times, contributing on Dirichlet boundary only
   */
  if(op_rt->n_cells_at_dirichlet_boundary() > 0)
  {
    factors.resize(extra_pressure_nbc.get_order());
    for(unsigned int i = 0; i < extra_pressure_nbc.get_order(); ++i)
      factors[i] = extra_pressure_nbc.get_beta(i);

    extrapolate_vectors(factors, pressure_nbc_rhs, pressure_nbc_rhs.back());
    op_rt->add_pressure_neumann_boundary_to_vector(pressure_nbc_rhs.back(), rhs);
  }

  // If Leray projection is not applied then the contributions from the old time derivative do not
  // cancel, so they have to be added
  if(not this->param.apply_leray_projection)
  {
    // TODO
    AssertThrow(false,
                dealii::ExcMessage("Leray projection cancels some terms, "
                                   "these are missing here."));
  }

  // IV.2. pressure Dirichlet boundary conditions
  if(not pde_operator->is_pressure_level_undefined())
  {
    // TODO
    AssertThrow(false, dealii::ExcMessage("Dirichlet BCs on the pressure are not implemented."));
  }

  // special case: pressure level is undefined
  // Set mean value of rhs to zero in order to obtain a consistent linear system of equations.
  // This is really necessary for the dual-splitting scheme in contrast to the pressure-correction
  // scheme and coupled solution approach due to the Dirichlet BC prescribed for the intermediate
  // velocity field and the pressure Neumann BC in case of the dual-splitting scheme.
  if(pde_operator->is_pressure_level_undefined())
    dealii::VectorTools::subtract_mean_value(rhs);
}

template<int dim, typename Number>
void
TimeIntBDFConsistentSplittingExtruded<dim, Number>::momentum_step()
{
  dealii::Timer timer;
  timer.restart();

  // in case we need to iteratively solve a linear or nonlinear system of equations
  if(this->param.viscous_problem() or this->param.non_explicit_convective_problem())
  {
    /*
     *  Calculate the right-hand side of the linear system of equations. This
     *  uses the interface to overlap vector operations before and after the
     *  loop for the two extrapolations that are necessary for this step, in
     *  order to increase the data locality
     */
    std::vector<Number> factors_velocity_rhs(velocity.size());
    std::vector<Number> factors_convective_term(velocity.size());
    for(unsigned int i = 0; i < velocity.size(); ++i)
    {
      factors_velocity_rhs[i]    = this->bdf.get_alpha(i) / this->get_time_step_size();
      factors_convective_term[i] = -this->extra.get_beta(i);
    }

    std::shared_ptr<dealii::Function<dim>> right_hand_side;
    if(this->param.right_hand_side == true)
    {
      right_hand_side = pde_operator->get_field_functions()->right_hand_side;
      right_hand_side->set_time(this->get_next_time());
    }
    op_rt->evaluate_momentum_rhs(
      pressure_np,
      solution_rt,
      1.0,
      rhs_rt,
      [&](const unsigned int begin, const unsigned int end) {
        extrapolate_vectors_range<false>(begin, end, factors_velocity_rhs, velocity, solution_rt);
        std::fill(rhs_rt.begin() + begin, rhs_rt.begin() + end, 0.);
      },
      [&](const unsigned int begin, const unsigned int end) {
        extrapolate_vectors_range<true>(
          begin, end, factors_convective_term, this->vec_convective_term, rhs_rt);
      },
      right_hand_side);

    const double t_rhs = timer.wall_time();

    dealii::Timer timer2;

    // solve linear system of equations
    std::pair<double, double> extrapolate_accuracy(0., 0.);
    unsigned int              n_iter = 0;

    const Number factor_mass = this->get_scaling_factor_time_derivative_term();
    const Number factor_lapl = this->pde_operator->get_viscous_kernel_data().viscosity;

    // Compute least squares fit - use the version that runs the inner product
    // computations embedded into the matrix-vector product of the current
    // vector to overlap memory access with computations
    const unsigned int  n_vectors = velocity_red.size();
    std::vector<double> projection_vector(n_vectors);
    AssertDimension(velocity_matvec.size(), 2 * n_vectors);
    AssertThrow(projection_vector.size() <= 5, dealii::ExcNotImplemented());
    std::array<const float *, 10> vec_ptrs = {};
    for(unsigned int j = 0; j < 2 * n_vectors; ++j)
      vec_ptrs[j] = velocity_matvec[j].begin();
    dealii::ndarray<dealii::VectorizedArray<double>, 5, 6> local_sums = {};

    op_rt_float->set_parameters(0.0, factor_lapl);
    op_rt_float->vmult_mass_and_laplace(velocity_matvec[1],
                                        velocity_matvec[0],
                                        velocity_red[0],
                                        [&](const unsigned int begin, const unsigned int end) {
                                          do_compute_inner_products_range<true>(begin,
                                                                                end,
                                                                                n_vectors,
                                                                                factor_mass,
                                                                                vec_ptrs,
                                                                                rhs_rt.begin(),
                                                                                local_sums);
                                        });
    extrapolate_accuracy = do_compute_least_squares_fit(solution_rt.get_mpi_communicator(),
                                                        local_sums,
                                                        projection_vector);

    const double t_proj = timer2.wall_time();
    timer2.restart();

    if(preconditioner_viscous.get_vector().get_partitioner().get() !=
       diagonal_mass.get_partitioner().get())
      preconditioner_viscous.get_vector().reinit(diagonal_mass, true);

    // This loop computes the residual vector in double precision as a
    // right-hand side for the actual CG solution and copies the vector to
    // single precision for the subsequent solution with CG. Note we compute
    // the extrapolation from the least-squares projection first at this
    // stage, to benefit from data locality in accessing the solution_rt
    // vector
    op_rt->set_parameters(-factor_mass, -factor_lapl);
    op_rt->vmult(
      rhs_rt,
      solution_rt,
      [&](const unsigned int begin, const unsigned int end) {
        extrapolate_vectors_range<false>(begin, end, projection_vector, velocity_red, solution_rt);
      },
      [&](const unsigned int begin, const unsigned int end) {
        std::copy(rhs_rt.begin() + begin, rhs_rt.begin() + end, rhs_float.begin() + begin);
      });

    // Do the last tasks before solving: (i) initialize the solution vector to
    // 0 before starting the solver to benefit from the low residual with the
    // double residual and (ii) set the right entries of the preconditioner.
    const unsigned int owned_size = solution_rt.locally_owned_size();
    DEAL_II_OPENMP_SIMD_PRAGMA
    for(unsigned int i = 0; i < owned_size; ++i)
      preconditioner_viscous.get_vector().local_element(i) =
        1.0f / (float(factor_mass) * diagonal_mass.local_element(i) +
                float(factor_lapl) * diagonal_laplace.local_element(i));
    velocity_red.back() = 0.f;

    const double t_residual = timer2.wall_time();
    timer2.restart();

    dealii::SolverControl control(this->param.solver_data_momentum.max_iter,
                                  extrapolate_accuracy.first *
                                    this->param.solver_data_momentum.rel_tol);

    dealii::SolverCG<VectorTypeFloat> solver_cg(control);
    op_rt_float->set_parameters(factor_mass, factor_lapl);
    solver_cg.solve(*op_rt_float, velocity_red.back(), rhs_float, preconditioner_viscous);
    n_iter = control.last_step();

    DEAL_II_OPENMP_SIMD_PRAGMA
    for(unsigned int i = 0; i < owned_size; ++i)
    {
      const Number u_i = solution_rt.local_element(i) + velocity_red.back().local_element(i);
      solution_rt.local_element(i)         = u_i;
      velocity_np.local_element(i)         = u_i;
      velocity_red.back().local_element(i) = u_i;
    }

    if(this->print_solver_info() and not(this->is_test))
    {
      this->pcout << std::endl
                  << std::setprecision(5) << "Viscous step prepare: " << t_rhs << "/" << t_proj
                  << "/" << t_residual << " s, solve " << timer2.wall_time() << " s";
    }

    iterations_viscous.first += 1;
    std::get<1>(iterations_viscous.second) += n_iter;

    if(this->print_solver_info() and not(this->is_test))
    {
      this->pcout << std::endl
                  << "Solve viscous step (projection reduced residual from "
                  << extrapolate_accuracy.first << " to " << extrapolate_accuracy.second
                  << " solve to " << control.last_value() << "):";
      print_solver_info_linear(this->pcout, n_iter, timer.wall_time());
    }
  }
  else // no viscous term and no (linearly) implicit convective term, i.e. there is nothing to do in
       // this step of the dual splitting scheme
  {
    // nothing to do
    AssertThrow(this->param.equation_type == EquationType::Euler and
                  this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit,
                dealii::ExcMessage("Logical error."));
  }

  this->timer_tree->insert({"Timeloop", "Viscous step"}, timer.wall_time());
}

template<int dim, typename Number>
void
TimeIntBDFConsistentSplittingExtruded<dim, Number>::prepare_vectors_for_next_timestep()
{
  swap_back_one_step(velocity);
  velocity[0].swap(solution_rt);

  swap_back_one_step(pressure);
  swap_back_one_step(pressure_matvec);

  swap_back_one_step(velocity_red);

  swap_back_one_step(this->vec_convective_term);
  swap_back_one_step(convective_divergence_rhs);
  swap_back_one_step(divergences);
  swap_back_one_step(pressure_nbc_rhs);

  // swap two steps because we keep viscous and mass vectors for viscosity
  swap_back_one_step(velocity_matvec);
  swap_back_one_step(velocity_matvec);
}

template<int dim, typename Number>
double
TimeIntBDFConsistentSplittingExtruded<dim, Number>::calculate_time_step_size()
{
  double time_step = 1.0;

  if(this->param.calculation_of_time_step_size == TimeStepCalculation::CFL)
  {
    double time_step_global = this->operator_base->calculate_time_step_cfl_global();
    time_step_global *= this->cfl;

    this->pcout << std::endl
                << "Calculation of time step size according to CFL condition:" << std::endl
                << std::endl;
    print_parameter(this->pcout, "CFL", this->cfl);
    print_parameter(this->pcout, "Time step size (global)", time_step_global);

    if(this->adaptive_time_stepping == true)
    {
      // if u(x,t=0)=0, this time step size will tend to infinity
      // Note that in the ALE case there is no possibility to know the grid velocity at this point
      // and to use it for the calculation of the time step size.
      AssertThrow(factor_cfl > 0, dealii::ExcNotInitialized());
      double time_step_adap = dealii::Utilities::truncate_to_n_digits(
        1.0 / (std::pow(this->param.degree_u, this->param.cfl_exponent_fe_degree_velocity) *
               factor_cfl),
        4);
      time_step_adap *= this->cfl;

      // use adaptive time step size only if it is smaller, otherwise use temporary time step size
      time_step = std::min(time_step_adap, time_step_global);

      // make sure that the maximum allowable time step size is not exceeded
      time_step = std::min(time_step, this->param.time_step_size_max);

      print_parameter(this->pcout, "Time step size (adaptive)", time_step);
    }
    else
    {
      time_step = adjust_time_step_to_hit_end_time(this->param.start_time,
                                                   this->param.end_time,
                                                   time_step_global);

      this->pcout << std::endl
                  << "Adjust time step size to hit end time:" << std::endl
                  << std::endl;
      print_parameter(this->pcout, "Time step size", time_step);
    }
  }
  else
  {
    AssertThrow(false,
                dealii::ExcMessage("Specified type of time step calculation is not implemented."));
  }

  return time_step;
}

template<int dim, typename Number>
double
TimeIntBDFConsistentSplittingExtruded<dim, Number>::recalculate_time_step_size() const
{
  AssertThrow(this->param.calculation_of_time_step_size == TimeStepCalculation::CFL,
              dealii::ExcMessage(
                "Adaptive time step is not implemented for this type of time step calculation."));

  double new_time_step_size;
  if(this->param.ale_formulation == true)
  {
    AssertThrow(false, dealii::ExcNotImplemented());
  }
  else
  {
    AssertThrow(factor_cfl > 0, dealii::ExcNotInitialized());
    new_time_step_size = dealii::Utilities::truncate_to_n_digits(
      1.0 /
        (std::pow(this->param.degree_u, this->param.cfl_exponent_fe_degree_velocity) * factor_cfl),
      4);
  }

  new_time_step_size *= this->cfl;

  // make sure that time step size does not exceed maximum allowable time step size
  new_time_step_size = std::min(new_time_step_size, this->param.time_step_size_max);

  bool use_limiter = true;
  if(use_limiter)
  {
    double last_time_step_size = this->get_time_step_size();
    double factor              = this->param.adaptive_time_stepping_limiting_factor;
    limit_time_step_change(new_time_step_size, last_time_step_size, factor);
  }

  return new_time_step_size;
}

template<int dim, typename Number>
void
TimeIntBDFConsistentSplittingExtruded<dim, Number>::solve_steady_problem()
{
  AssertThrow(false, dealii::ExcMessage("Steady solver not implemented yet."));

  this->pcout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number>
void
TimeIntBDFConsistentSplittingExtruded<dim, Number>::print_iterations() const
{
  std::vector<std::string> names;
  std::vector<double>      iterations_avg;

  if(this->param.nonlinear_problem_has_to_be_solved())
  {
    AssertThrow(false, dealii::ExcNotImplemented());
  }
  else
  {
    names = {"Pressure step", "Viscous step"};

    iterations_avg.resize(2);
    iterations_avg[0] =
      (double)iterations_pressure.second / std::max(1., (double)iterations_pressure.first);
    iterations_avg[1] = (double)std::get<1>(iterations_viscous.second) /
                        std::max(1., (double)iterations_viscous.first);
  }

  print_list_of_iterations(this->pcout, names, iterations_avg);
}

template<int dim, typename Number>
void
TimeIntBDFConsistentSplittingExtruded<dim, Number>::write_restart_vectors() const
{
  // Restart vectors need to be stored in a format compatible with the
  // DoFHandler, so the vectors written for
  // `RTOperator::RaviartThomasOperatorBase` need to be converted
  // appropriately.
  std::vector<VectorType>         velocity_copied(this->get_size_velocity());
  std::vector<VectorType const *> vectors_velocity;
  std::vector<VectorType const *> vectors_pressure;

  for(unsigned int i = 0; i < this->get_size_velocity(); i++)
  {
    velocity_copied[i].reinit(velocity_np);
    op_rt->copy_this_to_mf_vector(velocity[i], velocity_copied[i]);
    pde_operator->get_constraint_u().distribute(velocity_copied[i]);
    vectors_velocity.push_back(&velocity_copied[i]);
  }
  vectors_pressure.push_back(&pressure_np);

  this->operator_base->serialize_vectors(vectors_velocity, vectors_pressure);
}

template<int dim, typename Number>
void
TimeIntBDFConsistentSplittingExtruded<dim, Number>::read_restart_vectors()
{
  // Restart vectors need to be stored in a format compatible with the
  // DoFHandler, so the vectors written for
  // `RTOperator::RaviartThomasOperatorBase` need to be converted
  // appropriately.
  std::vector<VectorType> vectors_velocity(this->get_size_velocity());
  for(unsigned int i = 0; i < this->get_size_velocity(); i++)
  {
    vectors_velocity[i].reinit(velocity_np);
  }
  std::vector<VectorType *> vectors_velocity_ptr;
  std::vector<VectorType *> vectors_pressure_ptr;
  for(unsigned int i = 0; i < this->get_size_velocity(); i++)
  {
    vectors_velocity_ptr.push_back(&vectors_velocity[i]);
  }
  vectors_pressure_ptr.push_back(&pressure_np);

  this->operator_base->deserialize_vectors(vectors_velocity_ptr, vectors_pressure_ptr);

  for(int i = this->get_size_velocity() - 1; i >= 0; --i)
  {
    op_rt->copy_mf_to_this_vector(vectors_velocity[i], velocity[i]);
    factor_cfl = op_rt->evaluate_convective_and_divergence_term(
      velocity[i],
      this->vec_convective_term[i],
      convective_divergence_rhs[std::min<int>(i, convective_divergence_rhs.size() - 1)],
      divergences[i]);
    if(static_cast<std::size_t>(i) < velocity_red.size())
    {
      // Initialize the old solutions for better extrapolation
      velocity_red[i].copy_locally_owned_data_from(velocity[i]);
      op_rt_float->vmult_mass_and_laplace(velocity_matvec[2 * i + 1],
                                          velocity_matvec[2 * i],
                                          velocity_red[i],
                                          [](const unsigned int, const unsigned int) {});
    }

    if(static_cast<std::size_t>(i) < pressure_nbc_rhs.size())
      op_rt->evaluate_pressure_neumann_from_velocity(
        velocity[i],
        false,
        this->pde_operator->get_viscous_kernel_data().viscosity,
        pressure_nbc_rhs[i]);
  }
  velocity_np.copy_locally_owned_data_from(vectors_velocity[0]);
}

// instantiations

template class TimeIntBDFConsistentSplittingExtruded<3, double>;

} // namespace IncNS
} // namespace ExaDG
