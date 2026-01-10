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

#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_dual_splitting.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/laplace_operator_extruded.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/momentum_operator_rt.h>
#include <exadg/incompressible_navier_stokes/time_integration/helper_functions.h>
#include <exadg/incompressible_navier_stokes/time_integration/poisson_solver.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_dual_splitting_extruded.h>
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
TimeIntBDFDualSplittingExtruded<dim, Number>::TimeIntBDFDualSplittingExtruded(
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
    pressure(4),
    pressure_matvec(4),
    pressure_nbc_rhs(this->param.order_extrapolation_pressure_nbc),
    velocity_for_restart(this->order),
    pressure_for_restart(4),
    velocity_red_for_restart(4),
    velocity_matvec_for_restart(4 * 2),
    pressure_matvec_for_restart(4),
    iterations_pressure({0, 0}),
    iterations_projection({0, 0}),
    iterations_viscous({0, {0, 0}}),
    iterations_penalty({0, 0}),
    iterations_mass({0, 0}),
    extra_pressure_nbc(this->param.order_extrapolation_pressure_nbc,
                       this->param.start_with_low_order)
{
}

template<int dim, typename Number>
void
TimeIntBDFDualSplittingExtruded<dim, Number>::update_time_integrator_constants()
{
  // call function of base class to update the standard time integrator constants
  Base::update_time_integrator_constants();

  // update time integrator constants for extrapolation scheme of pressure Neumann bc
  extra_pressure_nbc.update(this->get_time_step_number(),
                            this->adaptive_time_stepping,
                            this->get_time_step_vector());

  // use this function to check the correctness of the time integrator constants
  //    std::cout << "Coefficients extrapolation scheme pressure NBC:" << std::endl;
  //    extra_pressure_nbc.print(this->pcout);
}

template<int dim, typename Number>
void
TimeIntBDFDualSplittingExtruded<dim, Number>::setup_derived()
{
  op_rt->evaluate_convective_term(velocity[0], 1.0, this->vec_convective_term[0]);
  op_rt->evaluate_pressure_neumann_from_velocity(
    velocity[0],
    true,
    this->pde_operator->get_viscous_kernel_data().viscosity,
    pressure_nbc_rhs[0]);
}

template<int dim, typename Number>
unsigned int
TimeIntBDFDualSplittingExtruded<dim, Number>::get_size_velocity() const
{
  return velocity.size();
}

template<int dim, typename Number>
unsigned int
TimeIntBDFDualSplittingExtruded<dim, Number>::get_size_pressure() const
{
  return pressure.size();
}

template<int dim, typename Number>
void
TimeIntBDFDualSplittingExtruded<dim, Number>::copy_to_vec_convective_term_for_restart(
  unsigned int const i) const
{
  // The standard matrix-free vector and the format used here are not identical, we overwrite the
  // method in the base class.
  op_rt->copy_this_to_mf_vector(this->vec_convective_term[i],
                                this->vec_convective_term_for_restart[i]);
}

template<int dim, typename Number>
void
TimeIntBDFDualSplittingExtruded<dim, Number>::copy_from_vec_convective_term_for_restart(
  unsigned int const i)
{
  // The standard matrix-free vector and the format used here are not identical, we overwrite the
  // method in the base class.
  op_rt->copy_mf_to_this_vector(this->vec_convective_term_for_restart[i],
                                this->vec_convective_term[i]);
}

template<int dim, typename Number>
void
TimeIntBDFDualSplittingExtruded<dim, Number>::get_vectors_serialization(
  std::vector<VectorType const *> & vectors_velocity,
  std::vector<VectorType const *> & vectors_pressure) const
{
  (void)vectors_pressure;
  (void)vectors_velocity;

  // // velocity vectors: velocity_np
  // vectors_velocity.push_back(&velocity_np);

  // // velocity vectors: velocity_red, needs to be converted to standard matrix-free vector
  // for(unsigned int i = 0; i < velocity_red_for_restart.size(); ++i)
  // {
  //   // initialize standard matrix-free vector
  //   pde_operator->initialize_vector_velocity(velocity_red_for_restart[i]);
  //   // copy entries converting float to Number
  //   op_rt_float->copy_this_to_mf_vector(velocity_red[i], velocity_red_for_restart[i]);
  //   // add pointer to serialization list
  //   vectors_velocity.push_back(&velocity_red_for_restart[i]);
  // }

  // // velocity vectors: velocity_matvec, needs to be converted to standard matrix-free vector
  // for(unsigned int i = 0; i < velocity_matvec_for_restart.size(); ++i)
  // {
  //   // initialize standard matrix-free vector
  //   pde_operator->initialize_vector_velocity(velocity_matvec_for_restart[i]);
  //   // copy entries converting float to Number
  //   op_rt_float->copy_this_to_mf_vector(velocity_matvec[i], velocity_matvec_for_restart[i]);
  //   // add pointer to serialization list
  //   vectors_velocity.push_back(&velocity_matvec_for_restart[i]);
  // }

  // // pressure vectors: pressure_np
  // vectors_pressure.push_back(&pressure_np);

  // // pressure vectors: pressure_matvec, needs to be converted to standard matrix-free vector
  // for(unsigned int i = 0; i < pressure_matvec_for_restart.size(); ++i)
  // {
  //   // initialize standard matrix-free vector
  //   pde_operator->initialize_vector_pressure(pressure_matvec_for_restart[i]);
  //   // copy entries converting float to Number
  //   pressure_matvec_for_restart[i] = pressure_matvec[i];

  //   // add pointer to serialization list
  //   vectors_pressure.push_back(&pressure_matvec_for_restart[i]);
  // }

  // // pressure vectors: pressure_nbc_rhs
  // for(unsigned int i = 0; i < pressure_nbc_rhs.size(); i++)
  // {
  //   vectors_pressure.push_back(&pressure_nbc_rhs[i]);
  // }

  // // PRINT ALL THE VECTORS WE DE-/DESERIALIZE
  // this->pcout << std::scientific << std::setprecision(8);

  // for(unsigned int i = 0; i < vectors_velocity.size(); ++i)
  // {
  //   this->pcout << "vectors_velocity[" << i << "]->l2_norm() : " <<
  //   vectors_velocity[i]->l2_norm()
  //               << "\n";
  // }
  // for(unsigned int i = 0; i < vectors_pressure.size(); ++i)
  // {
  //   this->pcout << "vectors_pressure[" << i << "]->l2_norm() : " <<
  //   vectors_pressure[i]->l2_norm()
  //               << "\n";
  // }


  // this->pcout << "velocity_np.l2_norm() = " << velocity_np.l2_norm() << "\n";
  // for(unsigned int i = 0; i < velocity.size(); ++i)
  // {
  //   this->pcout << "velocity[" << i << "].l2_norm() = " << velocity[i].l2_norm() << "\n";
  // }
  // for(unsigned int i = 0; i < velocity_red.size(); ++i)
  // {
  //   this->pcout << "velocity_red[" << i << "].l2_norm() = " << velocity_red[i].l2_norm() << "\n";
  // }
  // for(unsigned int i = 0; i < velocity_matvec.size(); ++i)
  // {
  //   this->pcout << "velocity_matvec[" << i << "].l2_norm() = " << velocity_matvec[i].l2_norm()
  //               << "\n";
  // }

  // this->pcout << "pressure_np.l2_norm() = " << pressure_np.l2_norm() << "\n";
  // for(unsigned int i = 0; i < pressure.size(); ++i)
  // {
  //   this->pcout << "pressure[" << i << "].l2_norm() = " << pressure[i].l2_norm() << "\n";
  // }
  // for(unsigned int i = 0; i < pressure_matvec.size(); ++i)
  // {
  //   this->pcout << "pressure_matvec[" << i << "].l2_norm() = " << pressure_matvec[i].l2_norm()
  //               << "\n";
  // }
  // for(unsigned int i = 0; i < pressure_nbc_rhs.size(); ++i)
  // {
  //   this->pcout << "pressure_nbc_rhs[" << i << "].l2_norm() = " << pressure_nbc_rhs[i].l2_norm()
  //               << "\n";
  // }
}

template<int dim, typename Number>
void
TimeIntBDFDualSplittingExtruded<dim, Number>::set_vectors_deserialization(
  std::vector<VectorType> const & vectors_velocity,
  std::vector<VectorType> const & vectors_pressure)
{
  (void)vectors_pressure;
  (void)vectors_velocity;

  // // velocity vectors: velocity_np
  // velocity_np = vectors_velocity[0];

  // // velocity vectors: velocity_red, needs to be converted from standard matrix-free vector
  // unsigned int start_index_velocity = 1;
  // for(unsigned int i = 0; i < velocity_red.size(); ++i)
  // {
  //   // copy entries converting float to Number
  //   op_rt_float->copy_mf_to_this_vector(vectors_velocity[start_index_velocity + i],
  //   velocity_red[i]);
  // }

  // // velocity vectors: velocity_matvec, needs to be converted from standard matrix-free vector
  // start_index_velocity = 1 + velocity_red.size();
  // for(unsigned int i = 0; i < velocity_matvec.size(); ++i)
  // {
  //   // copy entries converting float to Number
  //   op_rt_float->copy_mf_to_this_vector(vectors_velocity[start_index_velocity + i],
  //   velocity_matvec[i]);
  // }

  // // pressure vectors: pressure_np
  // pressure_np = vectors_pressure[0];

  // // pressure vectors: pressure_matvec, needs to be converted from standard matrix-free vector
  // unsigned int start_index_pressure = 1;
  // for(unsigned int i = 0; i < pressure_matvec.size(); ++i)
  // {
  //   // copy entries converting Number to float
  //   pressure_matvec[i] = vectors_pressure[start_index_pressure + i];
  // }

  // // pressure vectors: pressure_nbc_rhs
  // start_index_pressure = 1 + pressure_matvec.size();
  // for(unsigned int i = 0; i < pressure_nbc_rhs.size(); i++)
  // {
  //   pressure_nbc_rhs[i] = vectors_pressure[start_index_pressure + i];
  // }

  // // PRINT ALL THE VECTORS WE DE-/DESERIALIZE
  // this->pcout << std::scientific << std::setprecision(8);


  // for(unsigned int i = 0; i < vectors_velocity.size(); ++i)
  // {
  //   this->pcout << "vectors_velocity[" << i << "].l2_norm() : " << vectors_velocity[i].l2_norm()
  //               << "\n";
  // }
  // for(unsigned int i = 0; i < vectors_pressure.size(); ++i)
  // {
  //   this->pcout << "vectors_pressure[" << i << "].l2_norm() : " << vectors_pressure[i].l2_norm()
  //               << "\n";
  // }


  // this->pcout << "velocity_np.l2_norm() = " << velocity_np.l2_norm() << "\n";
  // for(unsigned int i = 0; i < velocity.size(); ++i)
  // {
  //   this->pcout << "velocity[" << i << "].l2_norm() = " << velocity[i].l2_norm() << "\n";
  // }
  // for(unsigned int i = 0; i < velocity_red.size(); ++i)
  // {
  //   this->pcout << "velocity_red[" << i << "].l2_norm() = " << velocity_red[i].l2_norm() << "\n";
  // }
  // for(unsigned int i = 0; i < velocity_matvec.size(); ++i)
  // {
  //   this->pcout << "velocity_matvec[" << i << "].l2_norm() = " << velocity_matvec[i].l2_norm()
  //               << "\n";
  // }

  // this->pcout << "pressure_np.l2_norm() = " << pressure_np.l2_norm() << "\n";
  // for(unsigned int i = 0; i < pressure.size(); ++i)
  // {
  //   this->pcout << "pressure[" << i << "].l2_norm() = " << pressure[i].l2_norm() << "\n";
  // }
  // for(unsigned int i = 0; i < pressure_matvec.size(); ++i)
  // {
  //   this->pcout << "pressure_matvec[" << i << "].l2_norm() = " << pressure_matvec[i].l2_norm()
  //               << "\n";
  // }
  // for(unsigned int i = 0; i < pressure_nbc_rhs.size(); ++i)
  // {
  //   this->pcout << "pressure_nbc_rhs[" << i << "].l2_norm() = " << pressure_nbc_rhs[i].l2_norm()
  //               << "\n";
  // }
}

template<int dim, typename Number>
void
TimeIntBDFDualSplittingExtruded<dim, Number>::allocate_vectors()
{
  Base::allocate_vectors();

  // restart vectors
  if(this->param.restarted_simulation or this->param.restart_data.write_restart)
  {
    for(unsigned int i = 0; i < velocity_for_restart.size(); ++i)
    {
      pde_operator->initialize_vector_velocity(velocity_for_restart[i]);
    }
    for(unsigned int i = 0; i < pressure_for_restart.size(); ++i)
    {
      pde_operator->initialize_vector_pressure(pressure_for_restart[i]);
    }
  }

  // velocity
  pde_operator->initialize_vector_velocity(velocity_np);

  // pressure
  pde_operator->initialize_vector_pressure(pressure_np);
  pde_operator->initialize_vector_pressure(pressure_rhs);
  for(VectorType & vector : pressure_nbc_rhs)
    pde_operator->initialize_vector_pressure(vector);

  // do test for matrix-free operator of optimized kind
  const std::vector<unsigned int> cell_vectorization_category =
    Helper::compute_vectorization_category(pde_operator->get_dof_handler_u().get_triangulation());
  op_rt = std::make_shared<RTOperator::RaviartThomasOperatorBase<dim, Number>>();
  op_rt->reinit(*pde_operator->get_mapping(),
                pde_operator->get_dof_handler_u(),
                pde_operator->get_constraint_u(),
                cell_vectorization_category,
                dealii::QGauss<1>(pde_operator->get_dof_handler_u().get_fe().degree + 1));

  op_rt->set_penalty_parameters(pde_operator->get_viscous_kernel_data().IP_factor);
  op_rt->initialize_dof_vector(solution_rt);
  op_rt->initialize_dof_vector(rhs_rt);

  op_rt_float = std::make_shared<RTOperator::RaviartThomasOperatorBase<dim, float>>();
  op_rt_float->reinit(*pde_operator->get_mapping(),
                      pde_operator->get_dof_handler_u(),
                      pde_operator->get_constraint_u(),
                      cell_vectorization_category,
                      dealii::QGauss<1>(pde_operator->get_dof_handler_u().get_fe().degree + 1));

  op_rt_float->set_penalty_parameters(pde_operator->get_viscous_kernel_data().IP_factor);

  if constexpr(dim == 3)
  {
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

    solutions_convective.resize(velocity.size());
    for(VectorType & vector : solutions_convective)
      op_rt->initialize_dof_vector(vector);

    for(VectorType & vec : this->vec_convective_term)
      op_rt->initialize_dof_vector(vec);
    op_rt->initialize_dof_vector(this->convective_term_np);

    for(auto & vector : velocity_red)
      op_rt_float->initialize_dof_vector(vector);
    for(auto & vector : velocity_matvec)
      op_rt_float->initialize_dof_vector(vector);
    op_rt_float->initialize_dof_vector(rhs_float);
  }

  laplace_op = std::make_shared<LaplaceOperator::LaplaceOperatorDG<dim, Number>>();
  laplace_op->reinit(*pde_operator->get_mapping(),
                     pde_operator->get_dof_handler_p(),
                     pde_operator->get_constraint_p(),
                     cell_vectorization_category,
                     dealii::QGauss<1>(pde_operator->get_dof_handler_p().get_fe().degree + 1));
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
    pde_operator->laplace_operator.get_data().kernel_data.IP_factor);

  for(unsigned int i = 0; i < pressure.size(); ++i)
    poisson_preconditioner->get_dg_matrix().initialize_dof_vector(pressure[i]);
  for(unsigned int i = 0; i < pressure_matvec.size(); ++i)
    poisson_preconditioner->get_dg_matrix().initialize_dof_vector(pressure_matvec[i]);
}



template<int dim, typename Number>
void
TimeIntBDFDualSplittingExtruded<dim, Number>::initialize_current_solution()
{
  if(this->param.ale_formulation)
    this->helpers_ale->move_grid(this->get_time());

  pde_operator->prescribe_initial_conditions(velocity_np, pressure_np, this->get_time());
  op_rt->copy_mf_to_this_vector(velocity_np, velocity[0]);
}

template<int dim, typename Number>
void
TimeIntBDFDualSplittingExtruded<dim, Number>::initialize_former_multistep_dof_vectors()
{
  // note that the loop begins with i=1! (we could also start with i=0 but this is not necessary)
  for(unsigned int i = 1; i < velocity.size(); ++i)
  {
    AssertThrow(false, dealii::ExcNotImplemented());
    if(this->param.ale_formulation)
      this->helpers_ale->move_grid(this->get_previous_time(i));

    VectorType tmp;
    tmp.reinit(velocity_np);
    pde_operator->prescribe_initial_conditions(tmp, pressure_np, this->get_previous_time(i));
    op_rt->copy_mf_to_this_vector(tmp, velocity[i]);
  }
}

template<int dim, typename Number>
typename TimeIntBDFDualSplittingExtruded<dim, Number>::VectorType const &
TimeIntBDFDualSplittingExtruded<dim, Number>::get_velocity() const
{
  return velocity_np;
}

template<int dim, typename Number>
typename TimeIntBDFDualSplittingExtruded<dim, Number>::VectorType const &
TimeIntBDFDualSplittingExtruded<dim, Number>::get_velocity_np() const
{
  return velocity_np;
}

template<int dim, typename Number>
typename TimeIntBDFDualSplittingExtruded<dim, Number>::VectorType const &
TimeIntBDFDualSplittingExtruded<dim, Number>::get_pressure() const
{
  return pressure_np;
}

template<int dim, typename Number>
typename TimeIntBDFDualSplittingExtruded<dim, Number>::VectorType const &
TimeIntBDFDualSplittingExtruded<dim, Number>::get_pressure(const unsigned int i) const
{
  AssertThrow(this->param.restarted_simulation or this->param.restart_data.write_restart,
              dealii::ExcMessage("Pressure vectors not initialized"));

  // Convert `VectorTypeFloat` to `VectorType` if necessary.
  pressure_for_restart[i].copy_locally_owned_data_from(pressure[i]);
  return pressure_for_restart[i];
}

template<int dim, typename Number>
typename TimeIntBDFDualSplittingExtruded<dim, Number>::VectorType const &
TimeIntBDFDualSplittingExtruded<dim, Number>::get_pressure_np() const
{
  return pressure_np;
}

template<int dim, typename Number>
typename TimeIntBDFDualSplittingExtruded<dim, Number>::VectorType const &
TimeIntBDFDualSplittingExtruded<dim, Number>::get_velocity(unsigned int i) const
{
  AssertThrow(this->param.restarted_simulation or this->param.restart_data.write_restart,
              dealii::ExcMessage("Velocity vectors not initialized"));

  // We need to copy since the `IndexSet`s are different.
  op_rt->copy_this_to_mf_vector(velocity[i], velocity_for_restart[i]);
  return velocity_for_restart[i];
}

template<int dim, typename Number>
void
TimeIntBDFDualSplittingExtruded<dim, Number>::set_velocity(VectorType const & velocity_in,
                                                           unsigned int const i)
{
  // We need to copy since the `IndexSet`s are different.
  op_rt->copy_mf_to_this_vector(velocity_in, velocity[i]);
}

template<int dim, typename Number>
void
TimeIntBDFDualSplittingExtruded<dim, Number>::set_pressure(VectorType const & pressure_in,
                                                           unsigned int const i)
{
  // Converts the incoming `VectorType` to `VectorTypeFloat` if necessary.
  pressure[i].copy_locally_owned_data_from(pressure_in);
}

template<int dim, typename Number>
void
TimeIntBDFDualSplittingExtruded<dim, Number>::do_timestep_solve()
{
  // perform the sub-steps of the dual-splitting method
  convective_step();

  pressure_step();

  viscous_step();

  // evaluate convective term once the final solution at time
  // t_{n+1} is known
  evaluate_convective_term();
}

template<int dim, typename Number>
void
TimeIntBDFDualSplittingExtruded<dim, Number>::convective_step()
{
  dealii::Timer timer;
  timer.restart();

  std::vector<Number> factors(this->vec_convective_term.size());
  // compute convective term and extrapolate convective term (if not Stokes equations)
  if(this->param.convective_problem())
  {
    if(this->param.ale_formulation)
    {
      for(unsigned int i = 0; i < this->vec_convective_term.size(); ++i)
      {
        // in a general setting, we only know the boundary conditions at time t_{n+1}
        pde_operator->evaluate_convective_term(this->vec_convective_term[i],
                                               velocity[i],
                                               this->get_next_time());
      }
    }

    for(unsigned int i = 0; i < this->extra.get_order(); ++i)
      factors[i] = -this->extra.get_beta(i);
    extrapolate_vectors(factors, this->vec_convective_term, rhs_rt);
  }
  else
  {
    rhs_rt = 0.0;
  }

  // compute body force vector
  if(this->param.right_hand_side == true)
  {
    op_rt->evaluate_add_body_force(this->get_next_time(),
                                   *pde_operator->get_field_functions()->right_hand_side,
                                   rhs_rt);
  }

  // apply inverse mass operator
  unsigned int             n_iter_mass = 0;
  dealii::ReductionControl control(this->param.inverse_mass_operator.solver_data.max_iter,
                                   this->param.inverse_mass_operator.solver_data.abs_tol,
                                   this->param.inverse_mass_operator.solver_data.rel_tol);

  dealii::SolverCG<VectorType> solver_cg(control);
  op_rt->set_parameters(1.0, 0.0);
  for(unsigned int i = 0; i < this->extra.get_order(); ++i)
    factors[i] = this->extra.get_beta(i);
  extrapolate_vectors(factors, solutions_convective, solutions_convective.back());
  solver_cg.solve(*op_rt, solutions_convective.back(), rhs_rt, preconditioner_mass);
  n_iter_mass = control.last_step();
  for(unsigned int i = solutions_convective.size() - 1; i > 0; --i)
    solutions_convective[i].swap(solutions_convective[i - 1]);

  iterations_mass.first += 1;
  iterations_mass.second += n_iter_mass;

  // calculate sum (alpha_i/dt * u_i) and add to velocity_np
  solution_rt = solutions_convective[0];
  for(unsigned int i = 0; i < velocity.size() - 1; ++i)
  {
    solution_rt.add(this->bdf.get_alpha(i) / this->get_time_step_size(), velocity[i]);
  }
  solution_rt.sadd(this->get_time_step_size() / this->bdf.get_gamma0(),
                   this->bdf.get_alpha(velocity.size() - 1) / this->bdf.get_gamma0(),
                   velocity.back());

  if(this->print_solver_info() and not(this->is_test))
  {
    if(this->param.spatial_discretization == SpatialDiscretization::HDIV)
    {
      this->pcout << std::endl << "Convective step:";
      print_solver_info_linear(this->pcout, n_iter_mass, timer.wall_time());
    }
    else if(this->param.spatial_discretization == SpatialDiscretization::L2)
    {
      this->pcout << std::endl << "Explicit convective step:";
      print_wall_time(this->pcout, timer.wall_time());
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }
  }

  this->timer_tree->insert({"Timeloop", "Convective step"}, timer.wall_time());
}

template<int dim, typename Number>
void
TimeIntBDFDualSplittingExtruded<dim, Number>::evaluate_convective_term()
{
  dealii::Timer timer;
  timer.restart();

  if(this->param.convective_problem())
  {
    if(this->param.ale_formulation == false) // Eulerian case
    {
      op_rt->evaluate_convective_term(solution_rt, 0.5, this->convective_term_np);

      swap_back_one_step(pressure_nbc_rhs);
      op_rt->evaluate_pressure_neumann_from_velocity(
        solution_rt,
        true,
        this->pde_operator->get_viscous_kernel_data().viscosity,
        pressure_nbc_rhs[0]);
    }
  }

  this->timer_tree->insert({"Timeloop", "Convective step"}, timer.wall_time());
}



template<int dim, typename Number>
void
TimeIntBDFDualSplittingExtruded<dim, Number>::pressure_step()
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
    poisson_preconditioner->get_dg_matrix().vmult(pressure_matvec[0], pressure[0]);
    extrapolate_accuracy =
      compute_least_squares_fit(pressure_matvec, pressure_rhs, pressure, pressure_np);
  }
  else
  {
    pressure_np = pressure_last_iter;
  }
  const double t_extrapol = timer2.wall_time();
  timer2.restart();

  // solve linear system of equations
  unsigned int                 n_iter = 0;
  dealii::ReductionControl     control(this->param.solver_data_pressure_poisson.max_iter,
                                   this->param.solver_data_pressure_poisson.abs_tol,
                                   this->param.solver_data_pressure_poisson.rel_tol);
  dealii::SolverCG<VectorType> solver(control);
  solver.solve(*laplace_op, pressure_np, pressure_rhs, *poisson_preconditioner);
  n_iter = control.last_step();

  iterations_pressure.first += 1;
  iterations_pressure.second += n_iter;
  const double t_sol = timer2.wall_time();

  // pde_operator->apply_laplace_operator(tmp, pressure_np);
  // const double res_norm4 = std::sqrt(tmp.add_and_dot(-1.0, rhs, tmp));
  // this->pcout << "Residual norms:   " << std::setprecision(3) << rhs_norm << " " << res_norm << "
  // "
  //             << res2_norm << " " << res_norm4 << " (" << n_iter << ")" << std::endl;

  // special case: pressure level is undefined
  // Adjust the pressure level in order to allow a calculation of the pressure error.
  // This is necessary because otherwise the pressure solution moves away from the exact solution.
  pde_operator->adjust_pressure_level_if_undefined(pressure_np, this->get_next_time());

  if(this->store_solution)
    pressure_last_iter = pressure_np;

  // write output
  if(this->print_solver_info() and not(this->is_test))
  {
    this->pcout << std::endl
                << "Pressure step prepare: " << t_rhs << "/" << t_extrapol << " s, solve " << t_sol
                << std::endl
                << "Solve pressure step (projection reduced residual from "
                << extrapolate_accuracy.first << " to " << extrapolate_accuracy.second << "):";
    print_solver_info_linear(this->pcout, n_iter, timer.wall_time());
  }

  this->timer_tree->insert({"Timeloop", "Pressure step"}, timer.wall_time());
}

template<int dim, typename Number>
void
TimeIntBDFDualSplittingExtruded<dim, Number>::rhs_pressure(VectorType & rhs) const
{
  /*
   * I. Pressure Neumann boundary terms
   */
  rhs.equ(this->extra_pressure_nbc.get_beta(0), pressure_nbc_rhs[0]);
  for(unsigned int i = 1; i < extra_pressure_nbc.get_order(); ++i)
    rhs.add(this->extra_pressure_nbc.get_beta(i), pressure_nbc_rhs[i]);

  if(this->param.right_hand_side)
    op_rt->evaluate_add_pressure_neumann_from_body_force(
      *pde_operator->get_field_functions()->right_hand_side, rhs);

  /*
   * II. calculate divergence term
   */
  op_rt->evaluate_add_velocity_divergence(solution_rt,
                                          -this->bdf.get_gamma0() / this->get_time_step_size(),
                                          rhs);


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
TimeIntBDFDualSplittingExtruded<dim, Number>::viscous_step()
{
  dealii::Timer timer;
  timer.restart();

  // in case we need to iteratively solve a linear or nonlinear system of equations
  if(this->param.viscous_problem() or this->param.non_explicit_convective_problem())
  {
    /*
     *  Calculate the right-hand side of the linear system of equations.
     */
    op_rt->evaluate_momentum_rhs(pressure_np,
                                 solution_rt,
                                 this->bdf.get_gamma0() / this->get_time_step_size(),
                                 rhs_rt);

    const double t_rhs = timer.wall_time();

    dealii::Timer timer2;

    // solve linear system of equations
    std::pair<double, double> extrapolate_accuracy(0., 0.);
    unsigned int              n_iter = 0;

    const Number factor_mass = this->get_scaling_factor_time_derivative_term();
    const Number factor_lapl = this->pde_operator->get_viscous_kernel_data().viscosity;

    op_rt_float->set_parameters(0.0, factor_lapl);
    op_rt_float->vmult(velocity_matvec[0], velocity_red[0]);
    op_rt_float->set_parameters(1.0, 0.0);
    op_rt_float->vmult(velocity_matvec[1], velocity_red[0]);
    extrapolate_accuracy = compute_least_squares_fit<VectorTypeFloat, VectorType, true>(
      velocity_matvec, rhs_rt, velocity_red, solution_rt, factor_mass);
    const double t_proj = timer2.wall_time();
    timer2.restart();

    preconditioner_viscous.get_vector().reinit(diagonal_mass, true);

    const unsigned int owned_size = diagonal_mass.locally_owned_size();
    DEAL_II_OPENMP_SIMD_PRAGMA
    for(unsigned int i = 0; i < owned_size; ++i)
      preconditioner_viscous.get_vector().local_element(i) =
        1.0 / (factor_mass * diagonal_mass.local_element(i) +
               factor_lapl * diagonal_laplace.local_element(i));
    const double t_prec = timer2.wall_time();
    timer2.restart();

    op_rt->set_parameters(-factor_mass, -factor_lapl);
    op_rt->vmult_add(rhs_rt, solution_rt);
    rhs_float.copy_locally_owned_data_from(rhs_rt);
    const double t_residual = timer2.wall_time();
    timer2.restart();

    dealii::ReductionControl          control(this->param.solver_data_momentum.max_iter,
                                     this->param.solver_data_momentum.abs_tol,
                                     this->param.solver_data_momentum.rel_tol);
    dealii::SolverCG<VectorTypeFloat> solver_cg(control);
    velocity_red.back() = 0;
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
                  << "Viscous step prepare: " << t_rhs << "/" << t_proj << "/" << t_prec << "/"
                  << t_residual << " s, solve " << timer2.wall_time() << " s";
    }

    iterations_viscous.first += 1;
    std::get<1>(iterations_viscous.second) += n_iter;

    if(this->print_solver_info() and not(this->is_test))
    {
      this->pcout << std::endl
                  << "Solve viscous step (projection reduced residual from "
                  << extrapolate_accuracy.first << " to " << extrapolate_accuracy.second << "):";
      print_solver_info_linear(this->pcout, n_iter, timer.wall_time());
    }

    if(this->store_solution)
      velocity_viscous_last_iter = velocity_np;
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
TimeIntBDFDualSplittingExtruded<dim, Number>::prepare_vectors_for_next_timestep()
{
  Base::prepare_vectors_for_next_timestep();

  swap_back_one_step(velocity);
  velocity[0].swap(solution_rt);

  swap_back_one_step(pressure);
  pressure[0] = pressure_np;

  swap_back_one_step(pressure_matvec);
  swap_back_one_step(velocity_red);

  // swap two steps because we keep viscous and mass vectors for viscosity
  swap_back_one_step(velocity_matvec);
  swap_back_one_step(velocity_matvec);
}

template<int dim, typename Number>
void
TimeIntBDFDualSplittingExtruded<dim, Number>::solve_steady_problem()
{
  AssertThrow(false, dealii::ExcMessage("Steady solver not implemented yet."));

  this->pcout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number>
void
TimeIntBDFDualSplittingExtruded<dim, Number>::print_iterations() const
{
  std::vector<std::string> names;
  std::vector<double>      iterations_avg;

  if(this->param.nonlinear_problem_has_to_be_solved())
  {
    names = {"Convective step",
             "Pressure step",
             "Projection step",
             "Viscous step (nonlinear)",
             "Viscous step (accumulated)",
             "Viscous step (linear per nonlinear)"};

    iterations_avg.resize(6);
    iterations_avg[0] = 0.0; // explicit convective step
    iterations_avg[1] =
      (double)iterations_pressure.second / std::max(1., (double)iterations_pressure.first);
    iterations_avg[2] =
      (double)iterations_projection.second / std::max(1., (double)iterations_projection.first);
    iterations_avg[3] = (double)std::get<0>(iterations_viscous.second) /
                        std::max(1., (double)iterations_viscous.first);
    iterations_avg[4] = (double)std::get<1>(iterations_viscous.second) /
                        std::max(1., (double)iterations_viscous.first);

    if(iterations_avg[3] > std::numeric_limits<double>::min())
      iterations_avg[5] = iterations_avg[4] / iterations_avg[3];
    else
      iterations_avg[5] = iterations_avg[4];
  }
  else
  {
    names = {"Convective step", "Pressure step", "Projection step", "Viscous step"};

    iterations_avg.resize(4);
    iterations_avg[0] = 0.0; // explicit convective step
    iterations_avg[1] =
      (double)iterations_pressure.second / std::max(1., (double)iterations_pressure.first);
    iterations_avg[2] =
      (double)iterations_projection.second / std::max(1., (double)iterations_projection.first);
    iterations_avg[3] = (double)std::get<1>(iterations_viscous.second) /
                        std::max(1., (double)iterations_viscous.first);
  }

  if(this->param.spatial_discretization == SpatialDiscretization::HDIV)
  {
    names.push_back("Mass solver");
    iterations_avg.push_back(iterations_mass.second / std::max(1., (double)iterations_mass.first));
  }

  if(this->param.apply_penalty_terms_in_postprocessing_step)
  {
    names.push_back("Penalty step");
    iterations_avg.push_back((double)iterations_penalty.second /
                             std::max(1., (double)iterations_penalty.first));
  }

  print_list_of_iterations(this->pcout, names, iterations_avg);
}

// instantiations

template class TimeIntBDFDualSplittingExtruded<3, double>;

} // namespace IncNS
} // namespace ExaDG
