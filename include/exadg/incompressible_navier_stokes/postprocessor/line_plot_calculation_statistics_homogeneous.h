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
class RaviartThomasOperatorBase;
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
 * NOTE: This functionality can only be used for hypercube meshes and for geometries/meshes for
 * which the cells are aligned with the coordinate axis.
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
  setup(LinePlotDataStatistics<dim> const &                        data_in,
        RTOperator::RaviartThomasOperatorBase<dim, Number> const & rt_operator);

  void
  evaluate(VectorType const & velocity, VectorType const & pressure, double const dt);

  void
  write_output() const;

  TimeControlStatistics time_control_statistics;

private:
  bool
  extract_line_data_on_cell(const typename dealii::DoFHandler<dim>::active_cell_iterator &cell);

  void
  print_headline(std::ofstream & f, unsigned int const number_of_samples) const;

  void
  do_evaluate(VectorType const & velocity, VectorType const & pressure, double const dt);

  void
  average_pressure_for_given_point(VectorType const & pressure,
                                   TYPE const &       vector_cells_and_ref_points,
                                   double &           length_local,
                                   double &           pressure_local);

  void
  do_write_output() const;

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

  // For all lines: for pressure reference point: list of all relevant cells and points in ref
  // coordinates
  std::vector<std::vector<
    std::pair<typename dealii::DoFHandler<dim>::active_cell_iterator, dealii::Point<dim>>>>
    cells_and_ref_points_ref_pressure;

  // number of samples for averaging in time
  unsigned int number_of_samples;

  // accumulated physical time for time-weighted averaging
  double accumulated_time;

  // homogeneous direction for averaging in space
  unsigned int averaging_direction;

  // Velocity quantities
  // For all lines: for all points along the line
  std::vector<std::vector<dealii::Tensor<1, dim, double>>> velocity_global;

  // Skin Friction quantities
  // For all lines: for all points along the line
  std::vector<std::vector<double>> wall_shear_global;

  // Reynolds Stress quantities
  // For all lines: for all points along the line
  std::vector<std::vector<dealii::SymmetricTensor<2, dim, double>>> reynolds_global;

  // Dissipation quantities
  // For all lines: for all points along the line
  std::vector<std::vector<double>> dissipation_global; // = epsilon

  // Grid size quantities (he)
  // For all lines: for all points along the line
  std::vector<std::vector<double>> grid_size_global; //= he

  // Pressure quantities
  // For all lines: for all points along the line
  std::vector<std::vector<double>> pressure_global;
  // For all lines
  std::vector<double> reference_pressure_global;

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

  RTOperator::RaviartThomasOperatorBase<dim, Number> const * rt_operator;
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
