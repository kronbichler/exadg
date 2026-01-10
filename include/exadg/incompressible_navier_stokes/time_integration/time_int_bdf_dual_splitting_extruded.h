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

#ifndef EXADG_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_DUAL_SPLITTING_EXTRUDED_H_
#define EXADG_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_DUAL_SPLITTING_EXTRUDED_H_

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
class OperatorDualSplitting;

template<int dim, typename Number>
class TimeIntBDFDualSplittingExtruded : public TimeIntBDF<dim, Number>
{
private:
  using BoostInputArchiveType  = TimeIntBase::BoostInputArchiveType;
  using BoostOutputArchiveType = TimeIntBase::BoostOutputArchiveType;

  typedef TimeIntBDF<dim, Number> Base;

  typedef typename Base::VectorType VectorType;
  using VectorTypeFloat = dealii::LinearAlgebra::distributed::Vector<float>;

  typedef OperatorDualSplitting<dim, Number> Operator;

public:
  TimeIntBDFDualSplittingExtruded(std::shared_ptr<Operator>                       operator_in,
                                  std::shared_ptr<HelpersALE<dim, Number> const>  helpers_ale_in,
                                  std::shared_ptr<PostProcessorInterface<Number>> postprocessor_in,
                                  Parameters const &                              param_in,
                                  MPI_Comm const &                                mpi_comm_in,
                                  bool const                                      is_test_in);

  virtual ~TimeIntBDFDualSplittingExtruded()
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

private:
  void
  allocate_vectors() final;

  void
  setup_derived() final;

  unsigned int
  get_size_velocity() const final;

  unsigned int
  get_size_pressure() const final;

  void
  copy_to_vec_convective_term_for_restart(unsigned int const i) const final;

  void
  copy_from_vec_convective_term_for_restart(unsigned int const i) final;

  void
  get_vectors_serialization(std::vector<VectorType const *> & vectors_velocity,
                            std::vector<VectorType const *> & vectors_pressure) const final;

  void
  set_vectors_deserialization(std::vector<VectorType> const & vectors_velocity,
                              std::vector<VectorType> const & vectors_pressure) final;

  void
  do_timestep_solve() final;

  void
  prepare_vectors_for_next_timestep() final;

  void
  convective_step();

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
  rhs_pressure(VectorType & rhs) const;

  void
  viscous_step();

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

  std::vector<VectorType> velocity;

  VectorType velocity_np; // standard mf vector

  std::vector<VectorTypeFloat> velocity_red;    // op_rt vector
  std::vector<VectorTypeFloat> velocity_matvec; // op_rt vector

  VectorType pressure_np;  // standard mf vector
  VectorType pressure_rhs; // standard mf vector, not to be deserialized

  std::vector<VectorTypeFloat> pressure;         // op_rt vector
  std::vector<VectorTypeFloat> pressure_matvec;  // standard mf vector, but initialized differently?
  std::vector<VectorType>      pressure_nbc_rhs; // op_rt vector

  mutable std::vector<VectorType> velocity_for_restart; // standard mf vector
  mutable std::vector<VectorType> pressure_for_restart; // standard mf vector

  mutable std::vector<VectorType> velocity_red_for_restart;    // standard mf vector
  mutable std::vector<VectorType> velocity_matvec_for_restart; // standard mf vector
  mutable std::vector<VectorType> pressure_matvec_for_restart; // standard mf vector

  // required for strongly-coupled partitioned FSI
  VectorType pressure_last_iter;
  VectorType velocity_projection_last_iter;
  VectorType velocity_viscous_last_iter;

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
  VectorTypeFloat                                                       rhs_float;

  // iteration counts
  std::pair<unsigned int /* calls */, unsigned long long /* iteration counts */>
    iterations_pressure;
  std::pair<unsigned int /* calls */, unsigned long long /* iteration counts */>
    iterations_projection;
  std::pair<
    unsigned int /* calls */,
    std::tuple<unsigned long long, unsigned long long> /* iteration counts {Newton, linear} */>
    iterations_viscous;

  std::pair<unsigned int /* calls */, unsigned long long /* iteration counts */> iterations_penalty;
  std::pair<unsigned int /* calls */, unsigned long long /* iteration counts */> iterations_mass;

  // time integrator constants: extrapolation scheme
  ExtrapolationConstants extra_pressure_nbc;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* EXADG_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_DUAL_SPLITTING_H_ \
        */
