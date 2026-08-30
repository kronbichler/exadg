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

#ifndef EXADG_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_CONSISTENT_SPLITTING_EXTRUDED_H_
#define EXADG_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_CONSISTENT_SPLITTING_EXTRUDED_H_

// ExaDG
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf.h>

namespace RTOperator
{
template<int dim, typename Number>
class RaviartThomasOperatorBase;
}

namespace LaplaceOperator
{
template<int dim, typename Number>
class LaplaceOperatorDG;
template<int dim, typename Number>
class LaplaceOperatorFE;
template<int dim, typename Number>
class PoissonPreconditionerMG;
} // namespace LaplaceOperator

namespace ExaDG
{
namespace IncNS
{
// forward declarations
template<int dim, typename Number>
class OperatorConsistentSplitting;

template<int dim, typename Number>
class TimeIntBDFConsistentSplittingExtruded : public TimeIntBDF<dim, Number>
{
private:
  using BoostInputArchiveType  = TimeIntBase::BoostInputArchiveType;
  using BoostOutputArchiveType = TimeIntBase::BoostOutputArchiveType;

  typedef TimeIntBDF<dim, Number> Base;

  typedef typename Base::VectorType VectorType;
  using VectorTypeFloat = dealii::LinearAlgebra::distributed::Vector<float>;

  typedef OperatorConsistentSplittingExtruded<dim, Number> Operator;

public:
  TimeIntBDFConsistentSplittingExtruded(
    std::shared_ptr<Operator>                       operator_in,
    std::shared_ptr<HelpersALE<dim, Number> const>  helpers_ale_in,
    std::shared_ptr<PostProcessorInterface<Number>> postprocessor_in,
    Parameters const &                              param_in,
    MPI_Comm const &                                mpi_comm_in,
    bool const                                      is_test_in);

  virtual ~TimeIntBDFConsistentSplittingExtruded()
  {
  }

  void
  print_iterations() const final;

  VectorType const &
  get_velocity() const final;

  VectorType const &
  get_velocity_np() const final;

  VectorType const &
  get_pressure() const final;

  VectorType const &
  get_pressure_np() const final;

  double
  calculate_time_step_size() override;

  double
  recalculate_time_step_size() const override;

  void
  write_restart_vectors() const override;

  void
  read_restart_vectors() override;

private:
  void
  allocate_vectors() final;

  void
  setup_derived() final;

  void
  write_vector_norms() const;

  unsigned int
  get_size_velocity() const final;

  unsigned int
  get_size_pressure() const final;

  void
  do_timestep_solve() final;

  void
  prepare_vectors_for_next_timestep() final;

  void
  evaluate_convective_term();

  void
  update_time_integrator_constants() final;

  void
  initialize_current_solution() final;

  void
  initialize_former_multistep_dof_vectors() final;

  void
  initialize_velocity_dbc();

  void
  pressure_step();

  void
  rhs_pressure(VectorType & rhs);

  void
  momentum_step();

  void
  solve_steady_problem() final;

  double
  evaluate_residual();

  VectorType const &
  get_velocity(unsigned int i /* t_{n-i} */) const final;

  VectorType const &
  get_pressure(unsigned int i /* t_{n-i} */) const final;

  void
  set_velocity(VectorType const & velocity, unsigned int const i /* t_{n-i} */) final;

  void
  set_pressure(VectorType const & pressure, unsigned int const i /* t_{n-i} */) final;

  std::shared_ptr<Operator> pde_operator;

  std::vector<VectorType> velocity; // `op_rt` vector

  VectorType velocity_np; // mf vector

  std::vector<VectorTypeFloat> velocity_red;    // `op_rt_float` vector
  std::vector<VectorTypeFloat> velocity_matvec; // `op_rt_float` vector

  VectorType pressure_np;  // mf vector
  VectorType pressure_rhs; // mf vector, do not serialize

  std::vector<VectorTypeFloat> pressure;                  // `dg_matrix` vector
  std::vector<VectorTypeFloat> pressure_matvec;           // `dg_matrix` vector
  std::vector<VectorType>      convective_divergence_rhs; // mf vector
  std::vector<VectorType>      divergences;               // mf vector
  std::vector<VectorType>      pressure_nbc_rhs;          // vector only living on boundary

  std::shared_ptr<RTOperator::RaviartThomasOperatorBase<dim, Number>>   op_rt;
  std::shared_ptr<RTOperator::RaviartThomasOperatorBase<dim, float>>    op_rt_float;
  std::shared_ptr<LaplaceOperator::LaplaceOperatorDG<dim, Number>>      laplace_op;
  std::shared_ptr<LaplaceOperator::PoissonPreconditionerMG<dim, float>> poisson_preconditioner;
  VectorTypeFloat                                                       diagonal_mass;
  VectorTypeFloat                                                       diagonal_laplace;
  dealii::DiagonalMatrix<VectorTypeFloat>                               preconditioner_viscous;
  dealii::DiagonalMatrix<VectorType>                                    preconditioner_mass;
  VectorType                                                            solution_rt;
  std::vector<VectorType>                                               solutions_convective;
  VectorType                                                            rhs_rt;
  std::array<double, 6>                                                 factors_time_step_mass;

  using ChebyshevSolver =
    dealii::PreconditionChebyshev<RTOperator::RaviartThomasOperatorBase<dim, float>,
                                  VectorTypeFloat,
                                  dealii::DiagonalMatrix<VectorTypeFloat>>;
  std::shared_ptr<ChebyshevSolver> momentum_solver_chebyshev;
  double                           momentum_sliding_condition_number_estimate;
  double                           momentum_sliding_iteration_count;
  double                           momentum_sliding_smoothing_range;

  double factor_cfl;

  // iteration counts and solver info
  Utilities::StatisticalQuantity stat_pressure_iterations;
  Utilities::StatisticalQuantity stat_pressure_rhs_norm;
  Utilities::StatisticalQuantity stat_pressure_residual_start;
  Utilities::StatisticalQuantity stat_pressure_residual_stop;
  Utilities::StatisticalQuantity stat_pressure_time_rhs;
  Utilities::StatisticalQuantity stat_pressure_time_projection;
  Utilities::StatisticalQuantity stat_pressure_time_solve;

  Utilities::StatisticalQuantity stat_momentum_iterations;
  Utilities::StatisticalQuantity stat_momentum_rhs_norm;
  Utilities::StatisticalQuantity stat_momentum_residual_start;
  Utilities::StatisticalQuantity stat_momentum_residual_stop;
  Utilities::StatisticalQuantity stat_momentum_time_rhs;
  Utilities::StatisticalQuantity stat_momentum_time_projection;
  Utilities::StatisticalQuantity stat_momentum_time_dp_residual;
  Utilities::StatisticalQuantity stat_momentum_time_solve_chebyshev;
  Utilities::StatisticalQuantity stat_momentum_time_solve_cg;

  // time integrator constants: extrapolation scheme
  ExtrapolationConstants extra_pressure_nbc;
  ExtrapolationConstants extra_pressure_rhs;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* EXADG_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_DUAL_SPLITTING_H_ \
        */
