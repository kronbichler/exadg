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

#ifndef EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_LINE_PLOT_CALCULATION_STATISTICS_HOMOGENEOUS_H_
#define EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_LINE_PLOT_CALCULATION_STATISTICS_HOMOGENEOUS_H_

// C/C++
#include <fstream>

// deal.II
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>
#include <deal.II/non_matching/mapping_info.h>

// ExaDG
#include <exadg/incompressible_navier_stokes/postprocessor/line_plot_data.h>
#include <exadg/postprocessor/time_control.h>

namespace RTOperator
{
template<int, typename>
class RaviartThomasOperator;
}

namespace ExaDG
{
namespace IncNS
{
/*
 * This function calculates statistics along lines over time
 * and one spatial, homogeneous direction (averaging_direction = {0,1,2}), e.g.,
 * in the x-direction with a line in the y-z plane.
 *
 * NOTE: This functionality can only be used for hypercube meshes and for
 * geometries/meshes for which the cells are aligned with the coordinate axis.
 */

template<int dim, typename Number>
class LinePlotCalculatorStatisticsHomogeneous
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef typename std::vector<
    std::pair<typename dealii::DoFHandler<dim>::active_cell_iterator, dealii::Point<dim>>>
    TYPE;

  LinePlotCalculatorStatisticsHomogeneous(dealii::DoFHandler<dim> const & dof_handler_velocity_in,
                                          dealii::DoFHandler<dim> const & dof_handler_pressure_in,
                                          dealii::Mapping<dim> const &    mapping_in,
                                          MPI_Comm const &                mpi_comm_in);

  void
  setup(LinePlotDataStatistics<dim> const & data_in);

  void
  setup(LinePlotDataStatistics<dim> const &                    data_in,
        RTOperator::RaviartThomasOperator<dim, Number> const & rt_operator);

  void
  evaluate(VectorType const & velocity,
           VectorType const & pressure,
           double const       time,
           double const       dt);

  void
  write_output(double const time);

  TimeControlStatistics time_control_statistics;

private:
  /**
   * This function computes, for every line in the given line plot statistics,
   * whether the given cell intersects with the line and if it does, stores
   * the reference coordinates of the line points on that cell in the
   * cells_and_ref_points data member of this class. The function returns a
   * 'true' value in case any of the points on the lines intersects with the
   * cell and needs the velocity, in order to be able to construct the related
   * solution representation.
   */
  bool
  intersect_lines_with_cell(const typename dealii::DoFHandler<dim>::active_cell_iterator & cell);

  void
  assert_all_line_points_are_found() const;

  void
  print_headline(std::ofstream & f, unsigned int const number_of_samples) const;

  void
  do_evaluate(VectorType const & velocity,
              VectorType const & pressure,
              double const       time,
              double const       dt);

  void
  average_pressure_for_given_point(VectorType const & pressure,
                                   TYPE const &       vector_cells_and_ref_points,
                                   double &           length_local,
                                   double &           pressure_local);

  void
  do_write_output(double const time) const;

  /**
   * Resets the state of the internal data used for the time-integral (mean)
   * computation, i.e., the sample/time bookkeeping and the accumulated
   * time-integral quantities for all lines. Called from `write_output()` if
   * `data.reset_integral_on_write == true`.
   */
  void
  reset_time_integral_data();

  mutable bool clear_files;

  dealii::DoFHandler<dim> const & dof_handler_velocity;
  dealii::DoFHandler<dim> const & dof_handler_pressure;
  dealii::Mapping<dim> const &    mapping;
  MPI_Comm                        mpi_comm;

  LinePlotDataStatistics<dim> data;

  // Global points
  std::vector<std::vector<dealii::Point<dim>>> global_points;

  // For all lines: list of all relevant cells and list of points in ref
  // coordinates on that cell
  std::vector<std::vector<std::pair<typename dealii::DoFHandler<dim>::active_cell_iterator,
                                    std::vector<std::pair<unsigned int, dealii::Point<dim>>>>>>
    cells_and_ref_points;

  // For all lines: for pressure reference point: list of all relevant cells and
  // points in ref coordinates
  std::vector<std::vector<
    std::pair<typename dealii::DoFHandler<dim>::active_cell_iterator, dealii::Point<dim>>>>
    cells_and_ref_points_ref_pressure;

  // number of samples for averaging in time
  unsigned int number_of_samples;

  // accumulated physical time for time-weighted averaging
  double accumulated_time;

  // time instant of the last call to `do_evaluate()`
  double time_last;

  // homogeneous direction for averaging in space
  unsigned int averaging_direction;

  // We hold for each quantity global vectors for all lines and all points along
  // the lines
  // i) the accumulating time integral of values along the line,
  // ii) instanteneous values of the last processed field, and
  // iii) the instantaneous (not time-integrated) spatial variance of the last
  //      processed field along the homogeneous direction. Since sample points
  //      are not evenly spaced, we use the weighted variance
  //      Var = (1/W) * sum_i w_i (mean - value_i)^2,
  //      W = sum_i w_i,
  //      with w_i the same quadrature/Jacobian weight used for the mean. Since
  //      sum_i w_i*value_i = mean*W by definition of the mean, this yields
  //      Var = <x^2>_z - mean^2, i.e., the weighted second moment minus mean^2.

  // Velocity quantities. Note that the spatial variance of the velocity is not stored
  // separately, since it is identical to the diagonal entries of the Reynolds stresses
  // (`reynolds_last_global[d][d] - velocity_last_global[d] * velocity_last_global[d]`).
  std::vector<std::vector<dealii::Tensor<1, dim, double>>> velocity_time_integral_global;
  std::vector<std::vector<dealii::Tensor<1, dim, double>>> velocity_last_global;

  // Skin Friction quantities
  std::vector<std::vector<double>> wall_shear_time_integral_global;
  std::vector<std::vector<double>> wall_shear_last_global;
  std::vector<std::vector<double>> wall_shear_variance_last_global;

  // Reynolds Stress quantities
  std::vector<std::vector<dealii::SymmetricTensor<2, dim, double>>> reynolds_time_integral_global;
  std::vector<std::vector<dealii::SymmetricTensor<2, dim, double>>> reynolds_last_global;

  // Dissipation quantities
  std::vector<std::vector<double>> dissipation_time_integral_global; // = epsilon
  std::vector<std::vector<double>> dissipation_last_global;          // = epsilon
  std::vector<std::vector<double>> dissipation_variance_last_global; // = epsilon

  // Grid size quantities (he)
  std::vector<std::vector<double>> grid_size_time_integral_global; //= he

  // Pressure quantities
  std::vector<std::vector<double>> pressure_time_integral_global;
  std::vector<std::vector<double>> pressure_last_global;
  std::vector<std::vector<double>> pressure_variance_last_global;
  // For all lines
  std::vector<double> reference_pressure_time_integral_global;
  std::vector<double> reference_pressure_last_global;

  // write final output
  bool write_final_output;

  // Jacobians needed for the evaluation of the RT polynomial space on the
  // support points of an FE_DGQ for later evaluation with FE_PointEvaluation.
  dealii::Table<2, dealii::Tensor<2, dim>> jacobians_at_nodal_points;

  // Inverse Jacobians needed for the evaluation of the FE_DGQ representation
  // of the velocity field
  std::vector<dealii::Tensor<2, dim>> inverse_jacobians_on_lines;

  // dof indices on cell for Raviart-Thomas elements
  std::vector<std::array<unsigned int, 2 * dim + 1>> dof_indices_on_cell;

  // tabulated 1D shape data used for evaluation with RT element
  dealii::AlignedVector<dealii::VectorizedArray<Number>> shape_values_eo_n;
  dealii::AlignedVector<dealii::VectorizedArray<Number>> shape_values_eo_t;
  dealii::AlignedVector<Number>                          shape_values_eo_dgq;

  RTOperator::RaviartThomasOperator<dim, Number> const * rt_operator;
  std::vector<std::array<unsigned int, dealii::VectorizedArray<Number>::size()>>
                                                   list_of_cells_to_evaluate;
  dealii::Table<2, dealii::Tensor<1, dim, Number>> evaluated_dg_values_on_cells;
  std::vector<unsigned int>                        active_cell_index_to_evaluate_index;

  // polynomials for FE_DGQ representing Lagrange polynomials in node points
  // of FE_DGQ, which is the basis for the computation along the lines
  std::vector<dealii::Polynomials::Polynomial<double>> polynomials_nodal;

  // evaluator for the pressure
  std::shared_ptr<dealii::FEPointEvaluation<1, dim, dim, Number>>     evaluator_p;
  std::shared_ptr<dealii::NonMatching::MappingInfo<dim, dim, Number>> nonmatching_mapping_info;
  std::vector<unsigned int>                                           pressure_dof_indices_on_cell;

  // timer results
  double time_all;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_LINE_PLOT_CALCULATION_STATISTICS_HOMOGENEOUS_H_ \
        */
