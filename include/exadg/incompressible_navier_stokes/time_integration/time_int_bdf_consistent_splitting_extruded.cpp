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

#include <algorithm>

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
    velocity(this->order + 1),
    velocity_red(4),
    velocity_matvec(velocity_red.size() * 2),
    pressure(5),
    pressure_matvec(pressure.size()),
    convective_divergence_rhs(this->param.order_extrapolation_pressure_nbc),
    divergences(this->order),
    pressure_nbc_rhs(this->param.order_extrapolation_pressure_nbc),
    factors_time_step_mass{},
    momentum_sliding_max_eigenvalue(0),
    momentum_sliding_min_eigenvalue(0),
    momentum_sliding_iteration_count(0),
    factor_cfl(-1.0),
    extra_pressure_nbc(this->param.order_extrapolation_pressure_nbc,
                       this->param.start_with_low_order),
    extra_pressure_rhs(this->param.order_extrapolation_pressure_rhs,
                       this->param.start_with_low_order),
    momentum_solver_in_extrapolation_mode(false),
    previous_time_steps(velocity_red.size()),
    extra_initial_guess(velocity_red.size(),
                        /* start_with_low_order */ true)
{
  op_rt = std::make_shared<RTOperator::RaviartThomasOperator<dim, Number>>();

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
  const std::vector<double> & time_steps = this->get_time_step_vector();
  extra_pressure_nbc.update(this->get_time_step_number(), this->adaptive_time_stepping, time_steps);

  extra_pressure_rhs.update(this->get_time_step_number(), this->adaptive_time_stepping, time_steps);

  // Record one more time step for the extrapolation formula for the initial
  // guess. Note that this shift assumes that this function is called exactly
  // once per time step. ALE serves as a proxy for multiphysics coupling FSI,
  // where this function is called multiple times.
  AssertThrow(not this->param.ale_formulation,
              dealii::ExcMessage("The time step history for the initial guess assumes one "
                                 "call per time step, which is not guaranteed."));
  for(unsigned int i = previous_time_steps.size() - 1; i >= time_steps.size(); --i)
    previous_time_steps[i] = previous_time_steps[i - 1];
  for(unsigned int i = 0; i < time_steps.size(); ++i)
    previous_time_steps[i] = time_steps[i];

  extra_initial_guess.update(this->param.start_with_low_order ?
                               this->get_time_step_number() :
                               this->get_time_step_number() + this->order - 1,
                             this->adaptive_time_stepping,
                             previous_time_steps);
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

  op_rt_float = std::make_shared<RTOperator::RaviartThomasOperator<dim, float>>();
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

  // depending on whether we have additional grids provided, enter the path
  // with semi-coarsening possibilities or the standard path with a classical
  // polynomial/geometric coarsening
  if(pde_operator->get_grid().coarse_triangulations.empty())
    poisson_preconditioner = std::make_shared<LaplaceOperator::PoissonPreconditionerMG<dim, float>>(
      *pde_operator->get_mapping(),
      pde_operator->get_dof_handler_p(),
      cell_vectorization_category,
      pde_operator->get_grid().mapping_function,
      pde_operator->laplace_operator.get_data().kernel_data.IP_factor,
      this->is_test);
  else
    poisson_preconditioner = std::make_shared<LaplaceOperator::PoissonPreconditionerMG<dim, float>>(
      *pde_operator->get_mapping(),
      pde_operator->get_dof_handler_p(),
      cell_vectorization_category,
      pde_operator->get_grid().coarse_triangulations,
      pde_operator->get_grid().coarse_mappings,
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
  return this->order;
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
  for(unsigned int i = 1; i < this->order; ++i)
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

  // Compute right-hand-side vector
  rhs_pressure(pressure_rhs);

  const double t_rhs = timer.wall_time();
  stat_pressure_time_rhs.add_sample(t_rhs);

  // Projection to combine solutions from previous time steps to get a good
  // initial estimate for the linear solver, following the algorithm by
  // Loehner, 'Projective prediction of pressure increments', 2004,
  // https://doi.org/10.1002/cnm.743
  dealii::Timer             timer2;
  std::pair<double, double> extrapolate_accuracy(0., 0.);
  if(this->use_extrapolation)
  {
    pressure[0].copy_locally_owned_data_from(pressure_np);

    // In first time step, need to get the right-hand side from the mat-vec,
    // as it is not in restart data, later just retrieve it from the old
    // right-hand side vectors.
    if(this->get_time_step_number() == 1)
      poisson_preconditioner->get_dg_matrix().vmult(pressure_matvec[0], pressure[0]);
    extrapolate_accuracy =
      compute_least_squares_fit(pressure_matvec, pressure_rhs, pressure, pressure_np);
    pressure_matvec.back().copy_locally_owned_data_from(pressure_rhs);
  }
  const double t_extrapol = timer2.wall_time();
  stat_pressure_time_projection.add_sample(t_extrapol);
  timer2.restart();

  // Solve linear system of equations: Try if we can converge in single
  // precision first, allowing at most 4 iterations (typical value for
  // expected error reduction). If we fail, use SolverCG in double precision
  // as the more robust choice.
  std::pair<unsigned int, double> n_iter{0, 1.};
  n_iter = poisson_preconditioner->solve(*laplace_op,
                                         pressure_rhs,
                                         pressure_np,
                                         4 /* n_max_iterations */,
                                         extrapolate_accuracy.first *
                                           this->param.solver_data_pressure_poisson.rel_tol);
  if(n_iter.second > extrapolate_accuracy.first * this->param.solver_data_pressure_poisson.rel_tol)
  {
    dealii::SolverControl        control(this->param.solver_data_pressure_poisson.max_iter,
                                  extrapolate_accuracy.first *
                                    this->param.solver_data_pressure_poisson.rel_tol);
    dealii::SolverCG<VectorType> solver(control);
    solver.solve(*laplace_op, pressure_np, pressure_rhs, *poisson_preconditioner);
    n_iter.first += control.last_step();
    n_iter.second = control.last_value();
  }

  const double t_sol = timer2.wall_time();

  stat_pressure_iterations.add_sample(n_iter.first);
  stat_pressure_rhs_norm.add_sample(extrapolate_accuracy.first);
  stat_pressure_residual_start.add_sample(extrapolate_accuracy.second);
  stat_pressure_residual_stop.add_sample(n_iter.second);
  stat_pressure_time_solve.add_sample(t_sol);

  // special case: pressure level is undefined
  // Adjust the pressure level in order to allow a calculation of the pressure error.
  // This is necessary because otherwise the pressure solution moves away from the exact solution.
  pde_operator->adjust_pressure_level_if_undefined(pressure_np, this->get_next_time());

  // write output
  if(this->print_solver_info() and not(this->is_test))
  {
    this->pcout << std::endl
                << std::setprecision(4) << "Pressure step prepare: " << t_rhs << "/" << t_extrapol
                << " s, solve " << t_sol << std::endl
                << "Solve pressure step (projection reduced residual from "
                << extrapolate_accuracy.first << " to " << extrapolate_accuracy.second
                << " solve to " << n_iter.second << "):";
    print_solver_info_linear(this->pcout, n_iter.first, timer.wall_time());
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

  // in case we need to iteratively solve a linear or nonlinear system of equations
  if(this->param.viscous_problem() or this->param.non_explicit_convective_problem())
  {
    /*
     * I. right-hand side
     */

    // Calculate the right-hand side of the linear system of equations. This
    // uses the interface to overlap vector operations before and after the
    // loop for the two extrapolations that are necessary for this step, in
    // order to increase the data locality.
    std::vector<Number> factors_velocity_rhs(this->order);
    std::vector<Number> factors_convective_term(this->order);
    std::vector<Number> factors_initial_guess(extra_initial_guess.get_order());
    for(unsigned int i = 0; i < this->order; ++i)
    {
      factors_velocity_rhs[i]    = this->bdf.get_alpha(i) / this->get_time_step_size();
      factors_convective_term[i] = -this->extra.get_beta(i);
    }
    for(unsigned int i = 0; i < factors_initial_guess.size(); ++i)
      factors_initial_guess[i] = extra_initial_guess.get_beta(i);

    std::shared_ptr<dealii::Function<dim>> right_hand_side;
    if(this->param.right_hand_side == true)
    {
      right_hand_side = pde_operator->get_field_functions()->right_hand_side;
      right_hand_side->set_time(this->get_next_time());
    }

    // When in extrapolation mode, we can compute the initial guess for
    // 'solution_rt' using the vectors that are also used to compute the time
    // derivative terms in the right-hand side term, so amortize the memory
    // access for that second purpose. The vector 'velocity.back()' holds the
    // extrapolated time derivative terms used for the rhs computation and
    // must not be used elsewhere in this solver
    if(momentum_solver_in_extrapolation_mode)
      op_rt->evaluate_momentum_rhs(
        pressure_np,
        velocity.back(),
        1.0,
        rhs_rt,
        [&](const unsigned int begin, const unsigned int end) {
          extrapolate_two_vectors_range(begin,
                                        end,
                                        factors_velocity_rhs,
                                        factors_initial_guess,
                                        velocity,
                                        velocity.back(),
                                        solution_rt);
          extrapolate_vectors_range<false>(
            begin, end, factors_convective_term, this->vec_convective_term, rhs_rt);
        },
        [&](const unsigned int, const unsigned int) {},
        right_hand_side);
    else
      op_rt->evaluate_momentum_rhs(
        pressure_np,
        velocity.back(),
        1.0,
        rhs_rt,
        [&](const unsigned int begin, const unsigned int end) {
          extrapolate_vectors_range<false>(
            begin, end, factors_velocity_rhs, velocity, velocity.back());
          extrapolate_vectors_range<false>(
            begin, end, factors_convective_term, this->vec_convective_term, rhs_rt);
        },
        [&](const unsigned int, const unsigned int) {},
        right_hand_side);

    const double t_rhs = timer.wall_time();
    stat_momentum_time_rhs.add_sample(t_rhs);

    dealii::Timer timer2;

    /*
     * II. initial guess by least-squares projection (if in that mode)
     */

    std::pair<double, double> extrapolate_accuracy(0., 0.);
    unsigned int              n_iterations = 0;

    const Number factor_mass = this->get_scaling_factor_time_derivative_term();
    const Number factor_lapl = this->pde_operator->get_viscous_kernel_data().viscosity;

    // Compute least squares fit, again using Loehner's approach - use the
    // version that runs the inner product computations embedded into the
    // matrix-vector product for the mass matrix vmult() of the current vector
    // to overlap memory access with computations on vmult(). We need the
    // second vector because the size of the mass matrix term changes from
    // time step to time step, but we do not want to form 'n_vectors' many
    // matrix-vector product with the momentum operator, so just replace the
    // difference in the product, which is done inside the evaluation of
    // do_compute_inner_products_range with the 'factors_time_step_mass'
    // argument.
    bool                may_leave_projection_mode = false;
    std::vector<double> projection_vector;
    double              t_proj = 0;
    if(not momentum_solver_in_extrapolation_mode)
    {
      const unsigned int n_vectors = velocity_red.size();
      AssertDimension(velocity_matvec.size(), 2 * n_vectors);
      AssertThrow(n_vectors <= 5, dealii::ExcNotImplemented());
      std::array<const float *, 10> vec_ptrs = {};
      for(unsigned int j = 0; j < 2 * n_vectors; ++j)
        vec_ptrs[j] = velocity_matvec[j].begin();
      dealii::ndarray<dealii::VectorizedArray<double>, 5, 6> local_sums = {};
      factors_time_step_mass[0]                                         = factor_mass;

      // Need to get mass operator to slot '1', the operator from the previous
      // time step (obtained from right-hand side) is in 'velocity_matvec[0]'
      // and all other even indices.
      op_rt_float->set_parameters(1.0, 0);
      op_rt_float->vmult(
        velocity_matvec[1],
        velocity_red[0],
        [&](const unsigned int begin, const unsigned int end) {
          std::fill(velocity_matvec[1].begin() + begin, velocity_matvec[1].begin() + end, 0.f);
        },
        [&](const unsigned int begin, const unsigned int end) {
          do_compute_inner_products_range<true>(
            begin, end, n_vectors, factors_time_step_mass, vec_ptrs, rhs_rt.begin(), local_sums);
        });

      double residual_with_extrapolation = std::numeric_limits<double>::infinity();
      projection_vector.resize(n_vectors);
      extrapolate_accuracy = do_compute_least_squares_fit(solution_rt.get_mpi_communicator(),
                                                          local_sums,
                                                          projection_vector,
                                                          factors_initial_guess,
                                                          &residual_with_extrapolation);

      // Further down, we only allow to leave the least-squares option if the
      // residual is sufficiently small.
      if(residual_with_extrapolation < 1.5 * extrapolate_accuracy.second)
        may_leave_projection_mode = true;

      t_proj = timer2.wall_time();
      stat_momentum_time_projection.add_sample(t_proj);
      timer2.restart();
    }

    /*
     * III. compute high-precision residual, usually in FP64/double
     */

    if(preconditioner_viscous.get_vector().get_partitioner().get() !=
       diagonal_mass.get_partitioner().get())
      preconditioner_viscous.get_vector().reinit(diagonal_mass, true);

    // This loop computes the residual vector in double precision as a
    // right-hand side for the actual CG solution and copies the vector to
    // single precision for the subsequent solution with CG. Note we compute
    // the extrapolation from the least-squares projection first at this
    // stage, to benefit from data locality in accessing the solution_rt
    // vector. Also note that we also copy the right-hand side into the second
    // to last slot of velocity_matvec for usage in the next time step, and
    // use the last slot of velocity_matvec for the solver. This vector will
    // not be needed again, as it is overwritten with the mass matrix mat-vec
    // in the next time step (done in the code above).
    op_rt->set_parameters(-factor_mass, -factor_lapl);
    if(momentum_solver_in_extrapolation_mode)
    {
      const auto compute_vector_norm_sqr_on_range =
        [this](const unsigned int begin,
               const unsigned int end) -> dealii::VectorizedArray<Number> {
        constexpr unsigned int          n_lanes   = dealii::VectorizedArray<Number>::size();
        const Number *                  ptr       = rhs_rt.begin() + begin;
        const unsigned int              n_batches = (end - begin) / (2 * n_lanes);
        dealii::VectorizedArray<Number> sum_0 = 0, sum_1 = 0;
        for(unsigned int i = 0; i < n_batches; ++i)
        {
          dealii::VectorizedArray<Number> tmp;
          tmp.load(ptr + n_lanes * 2 * i);
          sum_0 += tmp * tmp;
          tmp.load(ptr + n_lanes * (2 * i + 1));
          sum_1 += tmp * tmp;
        }
        unsigned int remainder = end - begin - 2 * n_lanes * n_batches;
        if(remainder > 0)
        {
          ptr += 2 * n_lanes * n_batches;
          if(remainder >= n_lanes)
          {
            dealii::VectorizedArray<Number> tmp;
            tmp.load(ptr);
            sum_0 += tmp * tmp;
            ptr += n_lanes;
            remainder -= n_lanes;
          }
          for(unsigned int i = 0; i < remainder; ++i)
            sum_1[i] += ptr[i] * ptr[i];
        }
        return sum_0 + sum_1;
      };

      dealii::VectorizedArray<Number> norm_sqr_rhs = 0, norm_sqr_residual = 0;
      op_rt->vmult(
        rhs_rt,
        solution_rt,
        [&](const unsigned int begin, const unsigned int end) {
          norm_sqr_rhs += compute_vector_norm_sqr_on_range(begin, end);
        },
        [&](const unsigned int begin, const unsigned int end) {
          std::copy(rhs_rt.begin() + begin,
                    rhs_rt.begin() + end,
                    velocity_matvec.back().begin() + begin);
          norm_sqr_residual += compute_vector_norm_sqr_on_range(begin, end);
        });
      std::array<double, 2> sums = {{norm_sqr_rhs.sum(), norm_sqr_residual.sum()}};
      dealii::Utilities::MPI::sum(dealii::ArrayView<const Number>(sums.data(), 2),
                                  solution_rt.get_mpi_communicator(),
                                  dealii::ArrayView<Number>(sums.data(), 2));
      extrapolate_accuracy = std::make_pair(std::sqrt(sums[0]), std::sqrt(sums[1]));
    }
    else
      op_rt->vmult(
        rhs_rt,
        solution_rt,
        [&](const unsigned int begin, const unsigned int end) {
          std::copy(rhs_rt.begin() + begin,
                    rhs_rt.begin() + end,
                    velocity_matvec[velocity_matvec.size() - 2].begin() + begin);
          extrapolate_vectors_range<false>(
            begin, end, projection_vector, velocity_red, solution_rt);
        },
        [&](const unsigned int begin, const unsigned int end) {
          std::copy(rhs_rt.begin() + begin,
                    rhs_rt.begin() + end,
                    velocity_matvec.back().begin() + begin);
        });

    /*
     * IV. Update point-Jacobi preconditioner entries
     */

    const unsigned int owned_size = solution_rt.locally_owned_size();
    DEAL_II_OPENMP_SIMD_PRAGMA
    for(unsigned int i = 0; i < owned_size; ++i)
      preconditioner_viscous.get_vector().local_element(i) =
        1.0f / (float(factor_mass) * diagonal_mass.local_element(i) +
                float(factor_lapl) * diagonal_laplace.local_element(i));

    const double t_residual = timer2.wall_time();
    stat_momentum_time_dp_residual.add_sample(t_residual);
    timer2.restart();

    /*
     * V. Solve the linear system with either CG or Chebyshev
     */

    op_rt_float->set_parameters(factor_mass, factor_lapl);

    double reached_residual = std::numeric_limits<double>::quiet_NaN();

    const double absolute_residual_target =
      extrapolate_accuracy.first * this->param.solver_data_momentum.rel_tol;
    const double initial_residual = extrapolate_accuracy.second;
    AssertThrow(initial_residual > 0.0,
                dealii::ExcMessage("Got invalid residual norm " +
                                   std::to_string(initial_residual)));
    const double max_reduction_float = 1e-7;

    AssertThrow(this->param.solver_data_momentum.linear_solver == LinearSolver::CG ||
                  this->param.solver_data_momentum.linear_solver == LinearSolver::Chebyshev,
                dealii::ExcMessage("For the viscous solver with the extruded specialization "
                                   "of the consistent splitting scheme, only the conjugate "
                                   "gradient (cg) and the Chebyshev solver are implemented."));

    // In the startup phase or at certain times, run the CG solver even if
    // Chebyshev was set as the solver. The reason to run it every 32 steps is
    // to provide a new eigenvalue estimate but, more importantly, let the CG
    // solver with its directional minimization clean up the situation from
    // time to time: Using just Chebyshev might leave some low-amplitude error
    // components below the threshold for the residual unnoticed for a longer
    // time, until they reach the residual tolerance. Adding some nonlinearity
    // like the minimization of CG can be more robust against those undesired
    // artifacts. Note that this was not tested thoroughly yet.
    const bool try_chebyshev_solver =
      this->param.solver_data_momentum.linear_solver == LinearSolver::Chebyshev &&
      this->get_time_step_number() > 10 && this->get_time_step_number() % 32 != 0 &&
      momentum_solver_chebyshev != nullptr;

    double solve_time = 0;
    if(try_chebyshev_solver)
    {
      const double target_reduction    = absolute_residual_target / initial_residual;
      const double requested_reduction = std::clamp(target_reduction, max_reduction_float, 1.9999);

      // This is a convergence formula for the Chebyshev solver, based on the
      // book by Varga, Matrix Iterative Analysis. Their Chapter 5, page 156,
      // formula (5.27) provides an error formula, which is then solved for
      // the iteration number 'm'.
      const double iteration_count_guess =
        std::log(2. / requested_reduction) * 0.5 *
        std::sqrt(std::max(momentum_sliding_max_eigenvalue / momentum_sliding_min_eigenvalue, 1.0));

      // In case we have past solutions with Chebyshev, we might have found
      // out that the estimation with the Varga formula is too pessimistic (as
      // not all eigenvalues are presented in the present right-hand side). We
      // thus use an exponential sliding average with the worst-case estimate
      // 'iteration_count_guess' and previous experience.
      if(momentum_sliding_iteration_count == 0)
        momentum_sliding_iteration_count = iteration_count_guess;
      else
        momentum_sliding_iteration_count =
          0.8 * momentum_sliding_iteration_count + 0.2 * iteration_count_guess;

      n_iterations =
        std::clamp(static_cast<unsigned int>(std::ceil(momentum_sliding_iteration_count)),
                   1u,
                   this->param.solver_data_momentum.max_iter);
      momentum_solver_chebyshev->set_degree(n_iterations);

      reached_residual =
        momentum_solver_chebyshev->vmult_with_last_residual_norm(velocity_red.back(),
                                                                 velocity_matvec.back());
      const double elapsed_time = timer2.wall_time();
      solve_time += elapsed_time;
      stat_momentum_time_solve_chebyshev.add_sample(elapsed_time);
      timer2.restart();
    }
    else
      velocity_red.back() = 0.f;

    // As a fall-back in case the Chebyshev residual does not meet the
    // tolerance or in case Chebyshev was not selected, we use the CG
    // solver. As to the tolerance to determine whether Chebyshev was good
    // enough, we intentionally choose a rather big failure factor of 2.5
    // because the Chebyshev solver actually does one more iteration after the
    // one where the residual is gathered.
    const double chebyshev_failure_factor        = 2.5;
    const double chebyshev_severe_failure_factor = 6.0;
    const bool   need_cg_solver =
      !try_chebyshev_solver ||
      reached_residual > chebyshev_failure_factor * absolute_residual_target;
    const bool had_severe_chebyshev_failure =
      try_chebyshev_solver &&
      reached_residual > chebyshev_severe_failure_factor * absolute_residual_target;

    if(need_cg_solver)
    {
      dealii::SolverControl control(this->param.solver_data_momentum.max_iter,
                                    absolute_residual_target);

      double                            min_eig = 0.0, max_eig = 0.0;
      dealii::SolverCG<VectorTypeFloat> solver_cg(control);
      if(this->param.solver_data_momentum.linear_solver == LinearSolver::Chebyshev)
        solver_cg.connect_eigenvalues_slot(
          [&min_eig, &max_eig](const std::vector<double> & eigval) {
            if(!eigval.empty())
            {
              min_eig = eigval.front();
              max_eig = eigval.back();
            }
          });

      solver_cg.solve(*op_rt_float,
                      velocity_red.back(),
                      velocity_matvec.back(),
                      preconditioner_viscous);
      n_iterations += control.last_step();
      reached_residual = control.last_value();

      // Prepare the Chebyshev solver for the other cases in case this is not
      // the cleanup part; we only accept the upper eigenvalue bound for
      // Chebyshev if we did at least 5 iterations, otherwise the CG solution
      // might be too inaccurate and miss high eigenvalues (which leads to
      // convergence problems with Chebyshev).
      constexpr unsigned int min_eigenvalue_iterations = 5;
      if(control.last_step() >= min_eigenvalue_iterations && min_eig > 0.0 && max_eig >= min_eig)
      {
        // Update the estimate for the eigenvalues that we pass to the
        // Chebyshev solver. We use an exponential sliding average here,
        // meaning that the previous value of the quantities is taken into
        // account. If we found more extremal eigenvalues than what we have in
        // our sliding averages, we use them immediately, otherwise we move
        // slowly to those updated values.

        // Only update the upper eigenvalue estimate when we did not do the
        // PreconditionChebyshev case before, because Chebyshev will have
        // damped the high-eigenvalue modes already and thus make them less
        // visible to SolverCG.  An experience from practice is that the
        // maximum eigenvalue of a Jacobi-preconditioned matrix stemming from
        // a reaction-diffusion PDE is at least 2, so use 2.05 as a lower
        // bound even if CG did not find such a big value. For other
        // preconditions, we do not have gathered practical bounds that could
        // be applied yet. This setting avoids some failures in Chebyshev when
        // CG only does a small number of iterations.
        if(not try_chebyshev_solver)
        {
          const double limited_max_eigenvalue =
            (this->param.preconditioner_momentum == MomentumPreconditioner::PointJacobi ?
               std::max(2.05, 1.1 * max_eig) :
               1.1 * max_eig);
          if(momentum_sliding_max_eigenvalue <= 0 ||
             limited_max_eigenvalue > momentum_sliding_max_eigenvalue)
            momentum_sliding_max_eigenvalue = limited_max_eigenvalue;
          else
            momentum_sliding_max_eigenvalue =
              0.7 * momentum_sliding_max_eigenvalue + 0.3 * limited_max_eigenvalue;
        }

        // The Ritz values estimate by SolverCG will always be within the true
        // eigenvalue range, so add a safety factor to the min eigenvalue as
        // well. When updating the sliding average, only accept those
        // estimates not too far off from the previous estimate.
        const double limited_min_eigenvalue = 0.9 * min_eig;
        if(momentum_sliding_min_eigenvalue <= 0 ||
           limited_min_eigenvalue < momentum_sliding_min_eigenvalue)
          momentum_sliding_min_eigenvalue = limited_min_eigenvalue;
        else if(limited_min_eigenvalue < 2 * momentum_sliding_min_eigenvalue)
          momentum_sliding_min_eigenvalue =
            0.7 * momentum_sliding_min_eigenvalue + 0.3 * limited_min_eigenvalue;

        typename ChebyshevSolver::AdditionalData chebyshev_data;
        chebyshev_data.preconditioner.reset(&preconditioner_viscous, [](auto *) {
          // custom deleter that does nothing, since the surrounding class owns
          // the preconditioner
        });
        chebyshev_data.eig_cg_n_iterations = 0;

        chebyshev_data.max_eigenvalue = momentum_sliding_max_eigenvalue;
        chebyshev_data.smoothing_range =
          momentum_sliding_max_eigenvalue / momentum_sliding_min_eigenvalue;

        // the actual degree will be set when we know the tolerance for the
        // solver
        chebyshev_data.degree = 1;

        momentum_solver_chebyshev = std::make_shared<ChebyshevSolver>();
        momentum_solver_chebyshev->initialize(*op_rt_float, chebyshev_data);
      }

      // If we got here after failing to meet the tolerance of Chebyshev, our
      // iteration guess that is based on observing previous convergence was
      // probably too aggressive (or, in rare cases, the estimates for the
      // eigenvalues yielded bounds where Chebyshev does not converge). For
      // severe failures, we should use the CG solver in the next iteration to
      // refresh the eigenvalue/smoothing range parameters and get a better
      // estimate of the sliding averages of eigenvalues. In any case, also
      // increase the iterations of Chebyshev to avoid a similar problem next
      // time around.
      if(had_severe_chebyshev_failure)
        momentum_solver_chebyshev.reset();
      if(try_chebyshev_solver)
        momentum_sliding_iteration_count += 1;

      const double elapsed_time = timer2.wall_time();
      solve_time += elapsed_time;
      stat_momentum_time_solve_cg.add_sample(elapsed_time);
      timer2.restart();
    }
    else
    {
      // If Chebyshev succeeded, we compute an estimate of the apparent
      // condition number from the achieved residual reduction, that can then
      // inform us about what iteration count should have been used. It is
      // done by rearranging the above Chebyshev iteration count formula for
      // the condition number. Note that we subtract 1 from the iteration
      // count because the guess returned by the Chebyshev solver indicates
      // the residual one iteration before the end.
      const double achieved_reduction = reached_residual / initial_residual;
      if(n_iterations > 1 && achieved_reduction > 0 && achieved_reduction < 2.)
      {
        const double denominator = std::log(2.0 / achieved_reduction);
        if(std::abs(denominator) > max_reduction_float)
        {
          momentum_sliding_iteration_count =
            0.7 * momentum_sliding_iteration_count +
            0.3 * (momentum_sliding_iteration_count - 1) *
              std::log(2. / (absolute_residual_target / initial_residual)) / denominator;
        }
      }
    }

    /*
     * VI. Add computed solution update to the high-precision vector to
     *     finalize solution
     */

    // With extrapolation, we do not need the 'float' vectors stored in
    // 'velocity_red' that are going into the least-squares system.
    if(momentum_solver_in_extrapolation_mode)
    {
      DEAL_II_OPENMP_SIMD_PRAGMA
      for(unsigned int i = 0; i < owned_size; ++i)
      {
        const Number u_i = solution_rt.local_element(i) + velocity_red.back().local_element(i);
        solution_rt.local_element(i) = u_i;
        velocity_np.local_element(i) = u_i;
      }
    }
    else
    {
      DEAL_II_OPENMP_SIMD_PRAGMA
      for(unsigned int i = 0; i < owned_size; ++i)
      {
        const Number u_i = solution_rt.local_element(i) + velocity_red.back().local_element(i);
        solution_rt.local_element(i)         = u_i;
        velocity_np.local_element(i)         = u_i;
        velocity_red.back().local_element(i) = u_i;
      }
    }

    /*
     * VII. Select the initial guess strategy for the next time step.
     */
    // Remember the strategy for output below.
    const bool used_extrapolation = momentum_solver_in_extrapolation_mode;

    // Update the strategy.
    if(this->get_time_step_number() > 10 && may_leave_projection_mode && velocity_red.size() == 4 &&
       this->order == 3)
      momentum_solver_in_extrapolation_mode = true;

    if(this->print_solver_info() and not(this->is_test))
    {
      this->pcout << std::endl << std::setprecision(4) << "Viscous step prepare: " << t_rhs;
      if(not used_extrapolation)
        this->pcout << "/" << t_proj;
      this->pcout << "/" << t_residual << " s, solve " << solve_time + timer2.wall_time() << " s";
    }

    stat_momentum_iterations.add_sample(n_iterations);
    stat_momentum_rhs_norm.add_sample(extrapolate_accuracy.first);
    stat_momentum_residual_start.add_sample(initial_residual);
    stat_momentum_residual_stop.add_sample(reached_residual);

    if(this->print_solver_info() and not(this->is_test))
    {
      this->pcout << std::endl
                  << "Solve viscous step (" << (used_extrapolation ? "extrapolation" : "projection")
                  << " reduced residual from " << extrapolate_accuracy.first << " to "
                  << initial_residual << " solve " << (need_cg_solver ? "CG" : "Ch") << " to "
                  << reached_residual << "):";
      print_solver_info_linear(this->pcout, n_iterations, timer.wall_time());
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

  // swap two steps because we need the right-hand side vector and the mass
  // matrix to construct the projection initial guess based on the velocity field
  swap_back_one_step(velocity_matvec);
  swap_back_one_step(velocity_matvec);

  for(unsigned int i = factors_time_step_mass.size() - 1; i > 0; --i)
    std::swap(factors_time_step_mass[i], factors_time_step_mass[i - 1]);
}



template<int dim, typename Number>
double
TimeIntBDFConsistentSplittingExtruded<dim, Number>::calculate_time_step_size()
{
  double time_step = 1.0;

  if(this->param.calculation_of_time_step_size == TimeStepCalculation::UserSpecified)
  {
    time_step = calculate_const_time_step(this->param.time_step_size, this->refine_steps_time);

    this->pcout << std::endl << "User specified time step size:" << std::endl << std::endl;
    print_parameter(this->pcout, "time step size", time_step);
  }
  else if(this->param.calculation_of_time_step_size == TimeStepCalculation::CFL)
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
    names = {"Pressure step", "Momentum step"};

    iterations_avg.resize(2);
    iterations_avg[0] = stat_pressure_iterations.get_mean();
    iterations_avg[1] = stat_momentum_iterations.get_mean();
  }

  print_list_of_iterations(this->pcout, names, iterations_avg);

  this->pcout << std::endl
              << "Detailed statistics of consistent splitting solver" << std::endl
              << "  Category                     min          geometric mean  "
              << "arithmetic mean       max" << std::scientific << std::endl;
  stat_pressure_iterations.print_statistics(this->pcout, "Pressure iterations", 25);
  stat_pressure_rhs_norm.print_statistics(this->pcout, "P rhs norm", 25);
  stat_pressure_residual_start.print_statistics(this->pcout, "P residual start solve", 25);
  stat_pressure_residual_stop.print_statistics(this->pcout, "P residual end solve", 25);
  stat_pressure_time_rhs.print_statistics(this->pcout, "P time rhs [s]", 25, &this->mpi_comm);
  stat_pressure_time_projection.print_statistics(this->pcout,
                                                 "P time projection [s]",
                                                 25,
                                                 &this->mpi_comm);
  stat_pressure_time_solve.print_statistics(this->pcout, "P time solve [s]", 25, &this->mpi_comm);
  this->pcout << std::endl;
  stat_momentum_iterations.print_statistics(this->pcout, "Momentum iterations", 25);
  stat_momentum_rhs_norm.print_statistics(this->pcout, "U rhs norm", 25);
  stat_momentum_residual_start.print_statistics(this->pcout, "U residual start solve", 25);
  stat_momentum_residual_stop.print_statistics(this->pcout, "U residual end solve", 25);
  stat_momentum_time_rhs.print_statistics(this->pcout, "U time rhs [s]", 25, &this->mpi_comm);
  stat_momentum_time_projection.print_statistics(
    this->pcout,
    "U time project (" + std::to_string(stat_momentum_time_projection.get_n_samples()) + "x) [s]",
    25,
    &this->mpi_comm);
  stat_momentum_time_dp_residual.print_statistics(this->pcout,
                                                  "U time residual FP64 [s]",
                                                  25,
                                                  &this->mpi_comm);
  stat_momentum_time_solve_cg.print_statistics(
    this->pcout,
    "U solve_cg (" + std::to_string(stat_momentum_time_solve_cg.get_n_samples()) + "x) [s]",
    25,
    &this->mpi_comm);
  stat_momentum_time_solve_chebyshev.print_statistics(
    this->pcout,
    "U solve_cheb (" + std::to_string(stat_momentum_time_solve_chebyshev.get_n_samples()) +
      "x) [s]",
    25,
    &this->mpi_comm);
  this->pcout << std::endl;
}



template<int dim, typename Number>
void
TimeIntBDFConsistentSplittingExtruded<dim, Number>::write_restart_vectors() const
{
  // Restart vectors need to be stored in a format compatible with the
  // DoFHandler, so the vectors written for
  // `RTOperator::RaviartThomasOperator` need to be converted
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
  // `RTOperator::RaviartThomasOperator` need to be converted
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
      // Initialize the old solutions for better extrapolation. Note that we
      // store the right-hand side for later times, but just the viscous
      // contribution for the restart vectors. This works because we set the
      // initial weights in the relevant array to zero.
      velocity_red[i].copy_locally_owned_data_from(velocity[i]);
      op_rt_float->set_parameters(0, this->pde_operator->get_viscous_kernel_data().viscosity);
      op_rt_float->vmult(velocity_matvec[2 * i], velocity_red[i]);
      op_rt_float->set_parameters(1, 0);
      op_rt_float->vmult(velocity_matvec[2 * i + 1], velocity_red[i]);
      factors_time_step_mass[i + 1] = 0.;
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
