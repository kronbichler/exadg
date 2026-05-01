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

// deal.II
#include <deal.II/base/timer.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>

// ExaDG
#include <exadg/grid/grid_data.h>
#include <exadg/incompressible_navier_stokes/postprocessor/line_plot_calculation_statistics_homogeneous.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/momentum_operator_rt.h>
#include <exadg/utilities/create_directories.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
LinePlotCalculatorStatisticsHomogeneous<dim, Number>::LinePlotCalculatorStatisticsHomogeneous(
  dealii::DoFHandler<dim> const & dof_handler_velocity_in,
  dealii::DoFHandler<dim> const & dof_handler_pressure_in,
  dealii::Mapping<dim> const &    mapping_in,
  MPI_Comm const &                mpi_comm_in)
  : clear_files(true),
    dof_handler_velocity(dof_handler_velocity_in),
    dof_handler_pressure(dof_handler_pressure_in),
    mapping(mapping_in),
    mpi_comm(mpi_comm_in),
    number_of_samples(0),
    accumulated_time(0.0),
    averaging_direction(2),
    write_final_output(false),
    rt_operator(nullptr)
{
  AssertThrow(get_element_type(dof_handler_velocity.get_triangulation()) == ElementType::Hypercube,
              dealii::ExcMessage("Only implemented for hypercube elements."));
}

template<int dim>
dealii::Point<dim>
find_unit_point(dealii::Mapping<dim, dim> const &                               mapping,
                typename dealii::Triangulation<dim, dim>::cell_iterator const & cell,
                dealii::Point<dim> const &                                      point)
{
  // coarse search first that identifies candidates by affine approximation
  auto const vertices = mapping.get_vertices(cell);
  auto const A_b      = dealii::GridTools::affine_cell_approximation<dim, dim>(vertices);
  dealii::DerivativeForm<1, dim, dim> const A_inv = A_b.first.covariant_form().transpose();
  dealii::Point<dim>                        p_unit(apply_transformation(A_inv, point - A_b.second));

  // if point is far away in affine approximation of unit cell, do not try
  // more accurate search to limit computational costs
  if(dealii::GeometryInfo<dim>::is_inside_unit_cell(p_unit, 1.0))
  {
    try
    {
      p_unit = mapping.transform_real_to_unit_cell(cell, point);
    }
    catch(...)
    {
      // A point that does not lie on the reference cell.
      p_unit[0] = 2.0;
    }
  }
  return p_unit;
}

template<int dim, typename Number>
void
LinePlotCalculatorStatisticsHomogeneous<dim, Number>::setup(
  LinePlotDataStatistics<dim> const & data_in)
{
  time_all = 0;
  data     = data_in;

  AssertThrow(Utilities::is_valid_timestep(
                data_in.time_control_data_statistics.write_preliminary_results_every_nth_time_step),
              dealii::ExcMessage("write_preliminary_results_every_nth_time_step has to be set."));
  time_control_statistics.setup(data_in.time_control_data_statistics);

  if(data_in.time_control_data_statistics.time_control_data.is_active)
  {
    AssertThrow(dim == 3, dealii::ExcMessage("Not implemented."));

    AssertThrow(data.lines.size() > 0, dealii::ExcMessage("Empty data"));

    global_points.resize(data.lines.size());
    cells_and_ref_points.resize(data.lines.size());
    cells_and_ref_points_ref_pressure.resize(data.lines.size());

    velocity_global.resize(data.lines.size());
    wall_shear_global.resize(data.lines.size());
    reynolds_global.resize(data.lines.size());
    pressure_global.resize(data.lines.size());
    reference_pressure_global.resize(data.lines.size());
    dissipation_global.resize(data.lines.size());
    grid_size_global.resize(data.lines.size());

    // make sure that line type is correct
    std::shared_ptr<LineHomogeneousAveraging<dim>> line_hom =
      std::dynamic_pointer_cast<LineHomogeneousAveraging<dim>>(data.lines[0]);
    AssertThrow(line_hom.get() != 0,
                dealii::ExcMessage("Invalid line type, expected LineHomogeneousAveraging<dim>"));
    averaging_direction = line_hom->averaging_direction;

    AssertThrow(averaging_direction == 0 or averaging_direction == 1 or averaging_direction == 2,
                dealii::ExcMessage("Take the average either in x, y or z-direction"));

    unsigned int line_iterator = 0;
    for(const std::shared_ptr<Line<dim>> & line : data.lines)
    {
      // make sure that line type is correct
      std::shared_ptr<LineHomogeneousAveraging<dim>> line_hom =
        std::dynamic_pointer_cast<LineHomogeneousAveraging<dim>>(line);

      AssertThrow(line_hom.get() != 0,
                  dealii::ExcMessage("Invalid line type, expected LineHomogeneousAveraging<dim>"));

      AssertThrow(averaging_direction == line_hom->averaging_direction,
                  dealii::ExcMessage("All lines must use the same averaging direction."));

      // Resize global variables for # of points on line
      if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
      {
        velocity_global[line_iterator].resize(line->n_points);
        pressure_global[line_iterator].resize(line->n_points);
        wall_shear_global[line_iterator].resize(line->n_points);
        reynolds_global[line_iterator].resize(line->n_points);
        dissipation_global[line_iterator].resize(line->n_points);
        grid_size_global[line_iterator].resize(line->n_points);
      }

      // initialize global_points: use equidistant points along line
      global_points[line_iterator].reserve(line->n_points);
      for(unsigned int i = 0; i < line->n_points; ++i)
      {
        dealii::Point<dim> point =
          line->begin + double(i) / double(line->n_points - 1) * (line->end - line->begin);
        if(line->manifold.get() != nullptr)
          point = line->manifold->push_forward(point);
        global_points[line_iterator].push_back(point);
      }
      ++line_iterator;
    }

    // Save all cells and corresponding points on unit cell
    // that are relevant for a given point along the line.

    // use a tolerance to check whether a point is inside the unit cell; we
    // also use this as bias to make sure exactly one cell finds points
    // located at the cell boundary
    double const tolerance = 1.e-8;

    pressure_dof_indices_on_cell.resize(dof_handler_velocity.get_triangulation().n_active_cells(),
                                        dealii::numbers::invalid_unsigned_int);
    std::vector<dealii::types::global_dof_index> pressure_dof_indices(
      dof_handler_pressure.get_fe().dofs_per_cell);
    // For velocity and pressure quantities:
    for(auto const & cell : dof_handler_velocity.active_cell_iterators())
    {
      if(cell->is_locally_owned())
      {
        unsigned int line_iterator = 0;
        for(const std::shared_ptr<Line<dim>> & line : data.lines)
        {
          AssertThrow(line->quantities.size() > 0,
                      dealii::ExcMessage("No quantities specified for line."));

          bool velocity_has_to_be_evaluated = false;
          bool pressure_has_to_be_evaluated = false;
          for(const std::shared_ptr<Quantity> & quantity : line->quantities)
          {
            if(quantity->type == QuantityType::Velocity or
               quantity->type == QuantityType::SkinFriction or
               quantity->type == QuantityType::ReynoldsStresses)
            {
              velocity_has_to_be_evaluated = true;
            }

            if(quantity->type == QuantityType::Pressure)
              pressure_has_to_be_evaluated = true;
          }

          if(velocity_has_to_be_evaluated == true || pressure_has_to_be_evaluated == true)
          {
            bool found_a_point_on_this_cell = false;
            // cells and reference points for all points along a line
            for(unsigned int p = 0; p < line->n_points; ++p)
            {
              // First, we move the line to the position of the current cell (vertex 0) in
              // averaging direction and check whether this new point is inside the current cell
              dealii::Point<dim> translated_point   = global_points[line_iterator][p];
              translated_point[averaging_direction] = cell->vertex(0)[averaging_direction];

              // If the new point lies in the current cell, we have to take the current cell into
              // account
              dealii::Point<dim> const p_unit = find_unit_point(mapping, cell, translated_point);

              // Use a relaxed tolerance if we are at the boundary in a certain direction
              bool point_within_cell = true;
              for(unsigned int d = 0; d < dim; ++d)
                if(d != averaging_direction)
                {
                  if(p_unit[d] <= -tolerance && !cell->at_boundary(2 * d))
                    point_within_cell = false;
                  // bias to always consider point on one cell, should be
                  // stable also with multiple MPI ranks
                  if(p_unit[d] >= 1. - tolerance && !cell->at_boundary(2 * d + 1))
                    point_within_cell = false;
                }

              if(point_within_cell)
              {
                if(not found_a_point_on_this_cell)
                {
                  cells_and_ref_points[line_iterator].emplace_back(
                    cell, std::vector<std::pair<unsigned int, dealii::Point<dim>>>());
                  found_a_point_on_this_cell = true;
                }
                cells_and_ref_points[line_iterator].back().second.emplace_back(
                  p, dealii::GeometryInfo<dim>::project_to_unit_cell(p_unit));
              }
            }
          }

          if(pressure_has_to_be_evaluated)
          {
            typename dealii::DoFHandler<dim>::active_cell_iterator cell_p =
              cell->as_dof_handler_iterator(dof_handler_pressure);
            cell_p->get_dof_indices(pressure_dof_indices);
            pressure_dof_indices_on_cell[cell_p->active_cell_index()] =
              dof_handler_pressure.locally_owned_dofs().index_within_set(pressure_dof_indices[0]);
          }
          ++line_iterator;
        }
      }
    }

    // Save all cells and corresponding points on unit cell that are relevant for a given point
    // along the line.
    for(auto const & cell : dof_handler_pressure.active_cell_iterators())
    {
      if(cell->is_locally_owned())
      {
        unsigned int line_iterator = 0;
        for(const std::shared_ptr<Line<dim>> & line : data.lines)
        {
          AssertThrow(line->quantities.size() > 0,
                      dealii::ExcMessage("No quantities specified for line."));

          // cells and reference points for reference pressure (only one point for each line)
          for(const std::shared_ptr<Quantity> & quantity : line->quantities)
          {
            AssertThrow(line->quantities.size() > 0,
                        dealii::ExcMessage("No quantities specified for line."));

            // evaluate quantities that involve pressure
            if(quantity->type == QuantityType::PressureCoefficient)
            {
              std::shared_ptr<QuantityPressureCoefficient<dim>> quantity_ref_pressure =
                std::dynamic_pointer_cast<QuantityPressureCoefficient<dim>>(quantity);

              // First, we move the line to the position of the current cell (vertex 0) in
              // averaging direction and check whether this new point is inside the current cell
              dealii::Point<dim> translated_point   = quantity_ref_pressure->reference_point;
              translated_point[averaging_direction] = cell->vertex(0)[averaging_direction];

              // If the new point lies in the current cell, we have to take the current cell into
              // account
              dealii::Point<dim> const p_unit = find_unit_point(mapping, cell, translated_point);
              if(dealii::GeometryInfo<dim>::is_inside_unit_cell(p_unit, tolerance))
              {
                cells_and_ref_points_ref_pressure[line_iterator].push_back(
                  std::pair<typename dealii::DoFHandler<dim>::active_cell_iterator,
                            dealii::Point<dim>>(cell, p_unit));
              }
            }
          }
          ++line_iterator;
        }
      }
    }

    const unsigned          degree_dgq = dof_handler_velocity.get_fe().degree;
    dealii::QGauss<1>       gauss_1d(degree_dgq + 1);
    dealii::Quadrature<dim> quadrature_cell(gauss_1d);
    dealii::FE_DGQ<dim>     fe_dummy(0);
    dealii::FEValues<dim>   fe_values(mapping, fe_dummy, quadrature_cell, dealii::update_jacobians);
    unsigned int            n_cells = 0, n_points = 0;
    for(const auto & line_points : cells_and_ref_points)
      for(const auto & [_, pts] : line_points)
      {
        ++n_cells;
        n_points += pts.size();
      }
    jacobians_at_nodal_points.reinit(n_cells, quadrature_cell.size());
    inverse_jacobians_on_lines.clear();
    inverse_jacobians_on_lines.reserve(n_points);
    dealii::FEPointEvaluation<1, dim> fe_point_eval(mapping, fe_dummy, dealii::update_jacobians);
    unsigned int                      cell_index = 0;
    std::vector<dealii::Point<dim>>   points;
    for(const auto & line_points : cells_and_ref_points)
      for(const auto & [cell, pts] : line_points)
      {
        fe_values.reinit(typename dealii::Triangulation<dim>::cell_iterator(cell));
        for(unsigned int i = 0; i < quadrature_cell.size(); ++i)
          jacobians_at_nodal_points(cell_index, i) = fe_values.jacobian(i);
        ++cell_index;

        points.clear();
        for(const auto & a : pts)
          points.push_back(a.second);
        fe_point_eval.reinit(cell, points);
        for(unsigned int i = 0; i < pts.size(); ++i)
          inverse_jacobians_on_lines.push_back(fe_point_eval.jacobian(i).covariant_form());
      }

    dof_indices_on_cell.clear();
    dealii::internal::MatrixFreeFunctions::ShapeInfo<Number> shape_info(
      gauss_1d, dof_handler_velocity.get_fe());
    if(shape_info.element_type == dealii::internal::MatrixFreeFunctions::tensor_raviart_thomas)
    {
      const unsigned int n_n    = shape_info.data[0].fe_degree + 1;
      const unsigned int n_t    = shape_info.data[1].fe_degree + 1;
      const unsigned int n_cols = (n_n + 1) / 2;
      const unsigned int n_rows = 2;
      shape_values_eo_n.resize(n_cols * n_rows);
      for(unsigned int i = 0; i < n_cols; ++i)
        for(unsigned int j = 0; j < (n_n + 1) / 2; ++j)
        {
          if(i == n_cols - 1 && n_n % 2 == 1)
            shape_values_eo_n[2 * i][j] = 0.5 * shape_info.data[0].shape_values[i * n_n + j];
          else
            shape_values_eo_n[2 * i][j] =
              0.5 * (shape_info.data[0].shape_values[i * n_n + j] +
                     shape_info.data[0].shape_values[i * n_n + n_n - 1 - j]);
          shape_values_eo_n[2 * i + 1][j] =
            0.5 * (shape_info.data[0].shape_values[i * n_n + j] -
                   shape_info.data[0].shape_values[i * n_n + n_n - 1 - j]);
        }
      shape_values_eo_t.resize(n_cols * n_rows);
      for(unsigned int i = 0; i < (n_t + 1) / 2; ++i)
        for(unsigned int j = 0; j < (n_n + 1) / 2; ++j)
        {
          if(i == n_cols - 1 && n_t % 2 == 1)
            shape_values_eo_t[2 * i][j] = 0.5 * shape_info.data[1].shape_values[i * n_n + j];
          else
            shape_values_eo_t[2 * i][j] =
              0.5 * (shape_info.data[1].shape_values[i * n_n + j] +
                     shape_info.data[1].shape_values[i * n_n + n_n - 1 - j]);
          shape_values_eo_t[2 * i + 1][j] =
            0.5 * (shape_info.data[1].shape_values[i * n_n + j] -
                   shape_info.data[1].shape_values[i * n_n + n_n - 1 - j]);
        }
    }
    else
    {
      shape_values_eo_dgq = shape_info.data[0].shape_values_eo;
    }

    polynomials_nodal =
      dealii::Polynomials::generate_complete_Lagrange_basis(gauss_1d.get_points());

    // Initialize non-matching mapping info
    nonmatching_mapping_info = std::make_shared<dealii::NonMatching::MappingInfo<dim, dim, Number>>(
      mapping, dealii::update_values | dealii::update_jacobians | dealii::update_gradients);
    std::vector<typename dealii::DoFHandler<dim>::active_cell_iterator> cells;
    std::vector<std::vector<dealii::Point<dim>>>                        unit_points;
    for(unsigned int line_index = 0; line_index < cells_and_ref_points.size(); ++line_index)
    {
      for(const std::shared_ptr<Quantity> & quantity : data.lines[line_index]->quantities)
        if(quantity->type == QuantityType::Pressure)
        {
          for(const auto & cell_and_pts : cells_and_ref_points[line_index])
          {
            cells.push_back(cell_and_pts.first);
            unit_points.emplace_back();
            unit_points.back().resize(cell_and_pts.second.size() * gauss_1d.size());
            for(unsigned int p = 0, idx = 0; p < cell_and_pts.second.size(); ++p)
              for(unsigned int q = 0; q < gauss_1d.size(); ++q, ++idx)
                for(unsigned int d = 0; d < dim; ++d)
                  unit_points.back()[idx][d] = (d == averaging_direction) ?
                                                 gauss_1d.point(q)[0] :
                                                 cell_and_pts.second[p].second[d];
          }
        }
    }
    nonmatching_mapping_info->reinit_cells(cells, unit_points);
    evaluator_p = std::make_shared<dealii::FEPointEvaluation<1, dim, dim, Number>>(
      *nonmatching_mapping_info, dof_handler_pressure.get_fe());


    create_directories(data.directory, mpi_comm);
  }
}

template<int dim, typename Number>
void
LinePlotCalculatorStatisticsHomogeneous<dim, Number>::setup(
  LinePlotDataStatistics<dim> const &                        data_in,
  RTOperator::RaviartThomasOperatorBase<dim, Number> const & rt_operator)
{
  time_all          = 0;
  data              = data_in;
  this->rt_operator = &rt_operator;

  AssertThrow(Utilities::is_valid_timestep(
                data_in.time_control_data_statistics.write_preliminary_results_every_nth_time_step),
              dealii::ExcMessage("write_preliminary_results_every_nth_time_step has to be set."));
  time_control_statistics.setup(data_in.time_control_data_statistics);

  if(data_in.time_control_data_statistics.time_control_data.is_active)
  {
    AssertThrow(dim == 3, dealii::ExcMessage("Not implemented."));

    AssertThrow(data.lines.size() > 0, dealii::ExcMessage("Empty data"));

    global_points.resize(data.lines.size());
    cells_and_ref_points.resize(data.lines.size());
    cells_and_ref_points_ref_pressure.resize(data.lines.size());

    velocity_global.resize(data.lines.size());
    wall_shear_global.resize(data.lines.size());
    reynolds_global.resize(data.lines.size());
    pressure_global.resize(data.lines.size());
    reference_pressure_global.resize(data.lines.size());
    dissipation_global.resize(data.lines.size());
    grid_size_global.resize(data.lines.size());

    // make sure that line type is correct
    std::shared_ptr<LineHomogeneousAveraging<dim>> line_hom =
      std::dynamic_pointer_cast<LineHomogeneousAveraging<dim>>(data.lines[0]);
    AssertThrow(line_hom.get() != 0,
                dealii::ExcMessage("Invalid line type, expected LineHomogeneousAveraging<dim>"));
    averaging_direction = line_hom->averaging_direction;

    AssertThrow(averaging_direction == 0 or averaging_direction == 1 or averaging_direction == 2,
                dealii::ExcMessage("Take the average either in x, y or z-direction"));

    unsigned int line_iterator = 0;
    for(const std::shared_ptr<Line<dim>> & line : data.lines)
    {
      // make sure that line type is correct
      std::shared_ptr<LineHomogeneousAveraging<dim>> line_hom =
        std::dynamic_pointer_cast<LineHomogeneousAveraging<dim>>(line);

      AssertThrow(line_hom.get() != 0,
                  dealii::ExcMessage("Invalid line type, expected LineHomogeneousAveraging<dim>"));

      AssertThrow(averaging_direction == line_hom->averaging_direction,
                  dealii::ExcMessage("All lines must use the same averaging direction."));

      // Resize global variables for # of points on line
      if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
      {
        velocity_global[line_iterator].resize(line->n_points);
        pressure_global[line_iterator].resize(line->n_points);
        wall_shear_global[line_iterator].resize(line->n_points);
        reynolds_global[line_iterator].resize(line->n_points);
        dissipation_global[line_iterator].resize(line->n_points);
        grid_size_global[line_iterator].resize(line->n_points);
      }

      // initialize global_points: use equidistant points along line
      global_points[line_iterator].reserve(line->n_points);
      for(unsigned int i = 0; i < line->n_points; ++i)
      {
        dealii::Point<dim> point =
          line->begin + double(i) / double(line->n_points - 1) * (line->end - line->begin);
        if(line->manifold.get() != nullptr)
          point = line->manifold->push_forward(point);
        global_points[line_iterator].push_back(point);
      }
      ++line_iterator;
    }

    // Save all cells and corresponding points on unit cell
    // that are relevant for a given point along the line.

    // use a tolerance to check whether a point is inside the unit cell; we
    // also use this as bias to make sure exactly one cell finds points
    // located at the cell boundary
    double const tolerance = 1.e-8;

    const dealii::Triangulation<dim> & tria = dof_handler_velocity.get_triangulation();
    std::vector<unsigned int>          tmp_list_of_cells_to_evaluate;
    active_cell_index_to_evaluate_index.clear();
    active_cell_index_to_evaluate_index.resize(tria.n_active_cells(),
                                               dealii::numbers::invalid_unsigned_int);
    pressure_dof_indices_on_cell.resize(tria.n_active_cells(),
                                        dealii::numbers::invalid_unsigned_int);
    std::vector<dealii::types::global_dof_index> pressure_dof_indices(
      dof_handler_pressure.get_fe().dofs_per_cell);
    // For velocity and pressure quantities:
    for(unsigned int ic = 0; ic < rt_operator.get_cell_level_index().size(); ++ic)
      for(unsigned int il = 0; il < dealii::VectorizedArray<Number>::size(); ++il)
        if(rt_operator.get_cell_level_index()[ic][il][0] != dealii::numbers::invalid_unsigned_int)
        {
          typename dealii::DoFHandler<dim>::active_cell_iterator cell(
            &tria,
            rt_operator.get_cell_level_index()[ic][il][0],
            rt_operator.get_cell_level_index()[ic][il][1],
            &dof_handler_velocity);
          AssertThrow(cell->is_locally_owned(), dealii::ExcInternalError());

          bool         cell_was_not_yet_considered = true;
          unsigned int line_iterator               = 0;
          for(const std::shared_ptr<Line<dim>> & line : data.lines)
          {
            AssertThrow(line->quantities.size() > 0,
                        dealii::ExcMessage("No quantities specified for line."));

            bool velocity_has_to_be_evaluated = false;
            bool pressure_has_to_be_evaluated = false;
            for(const std::shared_ptr<Quantity> & quantity : line->quantities)
            {
              if(quantity->type == QuantityType::Velocity or
                 quantity->type == QuantityType::SkinFriction or
                 quantity->type == QuantityType::ReynoldsStresses)
              {
                velocity_has_to_be_evaluated = true;
              }

              if(quantity->type == QuantityType::Pressure)
                pressure_has_to_be_evaluated = true;
            }

            if(velocity_has_to_be_evaluated == true || pressure_has_to_be_evaluated == true)
            {
              bool found_a_point_on_this_cell = false;
              // cells and reference points for all points along a line
              for(unsigned int p = 0; p < line->n_points; ++p)
              {
                // First, we move the line to the position of the current cell (vertex 0) in
                // averaging direction and check whether this new point is inside the current cell
                dealii::Point<dim> translated_point   = global_points[line_iterator][p];
                translated_point[averaging_direction] = cell->vertex(0)[averaging_direction];

                // If the new point lies in the current cell, we have to take the current cell into
                // account
                dealii::Point<dim> const p_unit = find_unit_point(mapping, cell, translated_point);

                bool point_within_cell = true;
                for(unsigned int d = 0; d < dim; ++d)
                  if(d != averaging_direction)
                  {
                    if(p_unit[d] <= -tolerance && !cell->at_boundary(2 * d))
                      point_within_cell = false;
                    // bias to always consider point on one cell, should be
                    // stable also with multiple MPI ranks
                    if(p_unit[d] >= 1. - tolerance && !cell->at_boundary(2 * d + 1))
                      point_within_cell = false;
                  }

                if(point_within_cell)
                {
                  if(not found_a_point_on_this_cell)
                  {
                    cells_and_ref_points[line_iterator].emplace_back(
                      cell, std::vector<std::pair<unsigned int, dealii::Point<dim>>>());
                    found_a_point_on_this_cell = true;
                  }
                  cells_and_ref_points[line_iterator].back().second.emplace_back(
                    p, dealii::GeometryInfo<dim>::project_to_unit_cell(p_unit));
                }
              }
              if(velocity_has_to_be_evaluated && found_a_point_on_this_cell &&
                 cell_was_not_yet_considered)
              {
                active_cell_index_to_evaluate_index[cell->active_cell_index()] =
                  tmp_list_of_cells_to_evaluate.size();
                tmp_list_of_cells_to_evaluate.push_back(
                  ic * dealii::VectorizedArray<Number>::size() + il);
                cell_was_not_yet_considered = false;
              }
            }

            if(pressure_has_to_be_evaluated)
            {
              typename dealii::DoFHandler<dim>::active_cell_iterator cell_p(
                &tria,
                rt_operator.get_cell_level_index()[ic][il][0],
                rt_operator.get_cell_level_index()[ic][il][1],
                &dof_handler_pressure);
              cell_p->get_dof_indices(pressure_dof_indices);
              pressure_dof_indices_on_cell[cell_p->active_cell_index()] =
                dof_handler_pressure.locally_owned_dofs().index_within_set(pressure_dof_indices[0]);
            }
            ++line_iterator;
          }
        }
    list_of_cells_to_evaluate.resize(
      (tmp_list_of_cells_to_evaluate.size() + dealii::VectorizedArray<Number>::size() - 1) /
      dealii::VectorizedArray<Number>::size());
    for(unsigned int c = 0, i = 0; i < list_of_cells_to_evaluate.size(); ++i)
      for(unsigned int v = 0; v < dealii::VectorizedArray<Number>::size(); ++v, ++c)
        list_of_cells_to_evaluate[i][v] = tmp_list_of_cells_to_evaluate[std::min<unsigned int>(
          c, tmp_list_of_cells_to_evaluate.size() - 1)];
    evaluated_dg_values_on_cells.reinit(
      list_of_cells_to_evaluate.size() * dealii::VectorizedArray<Number>::size(),
      dealii::Utilities::pow(dof_handler_velocity.get_fe().degree + 1, dim));

    // Save all cells and corresponding points on unit cell that are relevant for a given point
    // along the line.
    for(auto const & cell : dof_handler_pressure.active_cell_iterators())
    {
      if(cell->is_locally_owned())
      {
        unsigned int line_iterator = 0;
        for(const std::shared_ptr<Line<dim>> & line : data.lines)
        {
          AssertThrow(line->quantities.size() > 0,
                      dealii::ExcMessage("No quantities specified for line."));

          // cells and reference points for reference pressure (only one point for each line)
          for(const std::shared_ptr<Quantity> & quantity : line->quantities)
          {
            // evaluate quantities that involve pressure
            if(quantity->type == QuantityType::PressureCoefficient)
            {
              std::shared_ptr<QuantityPressureCoefficient<dim>> quantity_ref_pressure =
                std::dynamic_pointer_cast<QuantityPressureCoefficient<dim>>(quantity);

              // First, we move the line to the position of the current cell (vertex 0) in
              // averaging direction and check whether this new point is inside the current cell
              dealii::Point<dim> translated_point   = quantity_ref_pressure->reference_point;
              translated_point[averaging_direction] = cell->vertex(0)[averaging_direction];

              // If the new point lies in the current cell, we have to take the current cell into
              // account
              dealii::Point<dim> const p_unit = find_unit_point(mapping, cell, translated_point);
              if(dealii::GeometryInfo<dim>::is_inside_unit_cell(p_unit, tolerance))
              {
                cells_and_ref_points_ref_pressure[line_iterator].push_back(
                  std::pair<typename dealii::DoFHandler<dim>::active_cell_iterator,
                            dealii::Point<dim>>(cell, p_unit));
              }
            }
          }
          ++line_iterator;
        }
      }
    }

    const unsigned      degree_dgq = dof_handler_velocity.get_fe().degree;
    dealii::QGauss<1>   gauss_1d(degree_dgq + 1);
    dealii::FE_DGQ<dim> fe_dummy(0);
    unsigned int        n_points = 0;
    for(const auto & line_points : cells_and_ref_points)
      for(const auto & [_, pts] : line_points)
      {
        n_points += pts.size();
      }
    inverse_jacobians_on_lines.clear();
    inverse_jacobians_on_lines.reserve(n_points);
    dealii::FEPointEvaluation<1, dim> fe_point_eval(mapping, fe_dummy, dealii::update_jacobians);
    std::vector<dealii::Point<dim>>   points;
    for(const auto & line_points : cells_and_ref_points)
      for(const auto & [cell, pts] : line_points)
      {
        points.clear();
        for(const auto & a : pts)
          points.push_back(a.second);
        fe_point_eval.reinit(cell, points);
        for(unsigned int i = 0; i < pts.size(); ++i)
          inverse_jacobians_on_lines.push_back(fe_point_eval.jacobian(i).covariant_form());
      }

    polynomials_nodal =
      dealii::Polynomials::generate_complete_Lagrange_basis(gauss_1d.get_points());

    // Initialize non-matching mapping info
    nonmatching_mapping_info = std::make_shared<dealii::NonMatching::MappingInfo<dim, dim, Number>>(
      mapping, dealii::update_values | dealii::update_jacobians | dealii::update_gradients);
    std::vector<typename dealii::DoFHandler<dim>::active_cell_iterator> cells;
    std::vector<std::vector<dealii::Point<dim>>>                        unit_points;
    for(unsigned int line_index = 0; line_index < cells_and_ref_points.size(); ++line_index)
    {
      for(const std::shared_ptr<Quantity> & quantity : data.lines[line_index]->quantities)
        if(quantity->type == QuantityType::Pressure)
        {
          for(const auto & cell_and_pts : cells_and_ref_points[line_index])
          {
            cells.push_back(cell_and_pts.first);
            unit_points.emplace_back();
            unit_points.back().resize(cell_and_pts.second.size() * gauss_1d.size());
            for(unsigned int p = 0, idx = 0; p < cell_and_pts.second.size(); ++p)
              for(unsigned int q = 0; q < gauss_1d.size(); ++q, ++idx)
                for(unsigned int d = 0; d < dim; ++d)
                  unit_points.back()[idx][d] = (d == averaging_direction) ?
                                                 gauss_1d.point(q)[0] :
                                                 cell_and_pts.second[p].second[d];
          }
        }
    }
    nonmatching_mapping_info->reinit_cells(cells, unit_points);
    evaluator_p = std::make_shared<dealii::FEPointEvaluation<1, dim, dim, Number>>(
      *nonmatching_mapping_info, dof_handler_pressure.get_fe());

    create_directories(data.directory, mpi_comm);
  }
}

template<int dim, typename Number>
void
LinePlotCalculatorStatisticsHomogeneous<dim, Number>::evaluate(
  VectorType const & velocity,
  VectorType const & pressure,
  double const       time_step_size_for_sampling)
{
  do_evaluate(velocity, pressure, time_step_size_for_sampling);
}

template<int dim, typename Number>
void
LinePlotCalculatorStatisticsHomogeneous<dim, Number>::write_output() const
{
  do_write_output();
}

template<int dim, typename Number>
void
LinePlotCalculatorStatisticsHomogeneous<dim, Number>::print_headline(
  std::ofstream &    f,
  unsigned int const number_of_samples) const
{
  f << "number of samples: N = " << number_of_samples << " representing a time span of "
    << accumulated_time << std::endl;
}

void
mpi_sum_at_root(double * data_ptr, const unsigned int size, const MPI_Comm mpi_comm)
{
  if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
  {
    int ierr = MPI_Reduce(MPI_IN_PLACE, data_ptr, size, MPI_DOUBLE, MPI_SUM, 0, mpi_comm);
    AssertThrowMPI(ierr);
  }
  else
  {
    int ierr = MPI_Reduce(data_ptr, data_ptr, size, MPI_DOUBLE, MPI_SUM, 0, mpi_comm);
    AssertThrowMPI(ierr);
  }
}


using namespace dealii;

template<int templ_stride_in, int templ_stride_out, typename Number>
void
apply_matrix_vector_vect_eo(const VectorizedArray<Number> * matrix,
                            const Number *                  in,
                            Number *                        out,
                            const int                       n_rows,
                            const int                       n_cols,
                            const int                       run_stride_in  = 0,
                            const int                       run_stride_out = 0)
{
  const unsigned int n_lanes    = VectorizedArray<Number>::size();
  const int          stride_in  = templ_stride_in > 0 ? templ_stride_in : run_stride_in;
  const int          stride_out = templ_stride_out > 0 ? templ_stride_out : run_stride_out;

  const int      n_actual_rows = (n_rows + 1) / 2;
  const int      n_chunks      = (n_actual_rows + n_lanes - 1) / n_lanes;
  const Number * in_back       = in + stride_in * (n_cols - 1);
  const int      n_actual_cols = (n_cols + 1) / 2;
  if(n_chunks == 1)
  {
    Number                  x_p   = (*in + *in_back);
    Number                  x_m   = (*in - *in_back);
    VectorizedArray<Number> sum_p = matrix[0] * x_p;
    VectorizedArray<Number> sum_m = matrix[1] * x_m;
    for(int i = 1; i < n_actual_cols; ++i)
    {
      in += stride_in;
      in_back -= stride_in;
      x_p = (*in + *in_back);
      x_m = (*in - *in_back);
      sum_p += matrix[2 * i] * x_p;
      sum_m += matrix[2 * i + 1] * x_m;
    }
    const VectorizedArray<Number> result_p = sum_p + sum_m;
    const VectorizedArray<Number> result_m = sum_p - sum_m;
    for(int i = 0; i < n_actual_rows; ++i, out += stride_out)
      *out = result_p[i];
    for(int i = n_rows / 2 - 1; i >= 0; --i, out += stride_out)
      *out = result_m[i];
  }
  else
    AssertThrow(false,
                ExcNotImplemented("Implement some loop unrolling n=" + std::to_string(n_rows)));
}



template<int dim, typename Number>
void
read_rt_cell_values(const unsigned int                                             degree_normal,
                    const Number *                                                 src_vector,
                    const dealii::AlignedVector<dealii::VectorizedArray<Number>> & matrix_n,
                    const dealii::AlignedVector<dealii::VectorizedArray<Number>> & matrix_t,
                    const dealii::ndarray<unsigned int, 2 * dim + 1> &             dof_indices,
                    std::vector<Number> &                                          tmp_array,
                    std::vector<dealii::Tensor<1, dim, Number>> &                  out)
{
  const unsigned int n_t                = degree_normal;
  const unsigned int n_n                = n_t + 1;
  const unsigned int dofs_per_face      = dealii::Utilities::pow(n_t, dim - 1);
  const unsigned int dofs_per_plane     = n_t * (n_t - 1);
  const unsigned int cell_dofs_per_comp = dofs_per_plane * (dim > 2 ? n_t : 1);

  tmp_array.resize(cell_dofs_per_comp + 2 * dofs_per_face + dealii::Utilities::pow(n_n, dim));
  Number * tmp2 = tmp_array.data() + cell_dofs_per_comp + 2 * dofs_per_face;

  for(unsigned int f = 0; f < 2; ++f)
  {
    const unsigned int idx = dof_indices[f];
    if(idx != dealii::numbers::invalid_unsigned_int)
      for(unsigned int i = 0; i < dofs_per_face; ++i)
        tmp_array[f * n_t + i * n_n] = src_vector[idx + i];
  }
  {
    unsigned int idx = dof_indices[2 * dim];
    for(unsigned int i = 0; i < dofs_per_face; ++i)
      for(unsigned int i_x = 1; i_x < n_t; ++i_x, ++idx)
        tmp_array[i_x + i * n_n] = src_vector[idx];
  }
  // perform interpolation in x direction, leave gaps in output to fit y interpolation
  for(unsigned int i_z = 0, i = 0; i_z < (dim > 2 ? n_t : 1); ++i_z)
  {
    for(unsigned int i_y = 0; i_y < n_t; ++i_y, ++i)
      apply_matrix_vector_vect_eo<1, 1>(
        matrix_n.data(), tmp_array.data() + i * n_n, tmp2 + (i_z * n_n + i_y) * n_n, n_n, n_n);

    // perform interpolation in y direction
    for(unsigned int i_x = 0; i_x < n_n; ++i_x)
      apply_matrix_vector_vect_eo<0, 0>(matrix_t.data(),
                                        tmp2 + i_z * n_n * n_n + i_x,
                                        tmp2 + i_z * n_n * n_n + i_x,
                                        n_n,
                                        n_t,
                                        n_n,
                                        n_n);
  }
  if(dim == 3)
    for(unsigned int i = 0; i < n_n * n_n; ++i)
      apply_matrix_vector_vect_eo<0, 0>(
        matrix_t.data(), tmp2 + i, &out[i][0], n_n, n_t, n_n * n_n, n_n * n_n * dim);

  // y component
  for(unsigned int f = 0; f < 2; ++f)
  {
    const unsigned int idx = dof_indices[2 + f];
    for(unsigned int i_z = 0, i = 0; i_z < (dim > 2 ? n_t : 1); ++i_z)
      for(unsigned int i_x = 0; i_x < n_t; ++i_x, ++i)
        tmp_array[f * n_t * n_t + i_z * n_t * n_n + i_x] = src_vector[idx + i];
  }
  {
    unsigned int idx = dof_indices[2 * dim] + cell_dofs_per_comp;
    for(unsigned int i_z = 0; i_z < (dim > 2 ? n_t : 1); ++i_z)
      for(unsigned int i_y = 1; i_y < n_t; ++i_y)
        for(unsigned int i_x = 0; i_x < n_t; ++i_x, ++idx)
          tmp_array[i_z * n_t * n_n + i_y * n_t + i_x] = src_vector[idx];
  }

  for(unsigned int i_z = 0, i = 0; i_z < (dim > 2 ? n_t : 1); ++i_z)
  {
    // perform interpolation in y direction
    for(unsigned int i_x = 0; i_x < n_t; ++i_x)
      apply_matrix_vector_vect_eo<0, 0>(matrix_n.data(),
                                        tmp_array.data() + i_z * n_t * n_n + i_x,
                                        tmp2 + i_z * n_n * n_n + i_x,
                                        n_n,
                                        n_n,
                                        n_t,
                                        n_n);

    // perform interpolation in x direction
    for(unsigned int i_y = 0; i_y < n_n; ++i_y, ++i)
    {
      apply_matrix_vector_vect_eo<1, 1>(matrix_t.data(), tmp2 + i * n_n, tmp2 + i * n_n, n_n, n_t);
    }
  }
  if(dim == 3)
    for(unsigned int i = 0; i < n_n * n_n; ++i)
      apply_matrix_vector_vect_eo<0, 0>(
        matrix_t.data(), tmp2 + i, &out[i][1], n_n, n_t, n_n * n_n, n_n * n_n * dim);

  // z component
  if constexpr(dim == 3)
  {
    for(unsigned int f = 0; f < 2; ++f)
    {
      const unsigned int idx = dof_indices[4 + f];
      for(unsigned int i = 0; i < dofs_per_face; ++i)
        tmp_array[f * n_t * dofs_per_face + i] = src_vector[idx + i];
    }
    {
      unsigned int idx = dof_indices[2 * dim] + 2 * cell_dofs_per_comp;
      for(unsigned int i_z = 1; i_z < n_t; ++i_z)
        for(unsigned int i_y = 0; i_y < n_t; ++i_y)
          for(unsigned int i_x = 0; i_x < n_t; ++i_x, ++idx)
            tmp_array[i_z * n_t * n_t + i_y * n_t + i_x] = src_vector[idx];
    }
    for(unsigned int i = 0; i < n_t * n_t; ++i)
      apply_matrix_vector_vect_eo<0, 0>(
        matrix_n.data(), tmp_array.data() + i, tmp2 + i, n_n, n_n, n_t * n_t, n_n * n_n);
    for(unsigned int i_z = 0; i_z < n_n; ++i_z)
    {
      for(unsigned int i_x = 0; i_x < n_t; ++i_x)
        apply_matrix_vector_vect_eo<0, 0>(matrix_t.data(),
                                          tmp2 + i_z * n_n * n_n + i_x,
                                          tmp2 + i_z * n_n * n_n + i_x,
                                          n_n,
                                          n_t,
                                          n_t,
                                          n_t);
      for(unsigned int i_y = 0; i_y < n_n; ++i_y)
        apply_matrix_vector_vect_eo<1, dim>(matrix_t.data(),
                                            tmp2 + i_z * n_n * n_n + i_y * n_t,
                                            &out[i_z * n_n * n_n + i_y * n_n][2],
                                            n_n,
                                            n_t);
    }
  }
}

template<int dim, typename Number>
void
LinePlotCalculatorStatisticsHomogeneous<dim, Number>::do_evaluate(
  VectorType const & velocity,
  VectorType const & pressure,
  double const       time_step_size_for_sampling)
{
  dealii::Timer time;
  using VectorizedArrayType = dealii::VectorizedArray<Number>;
  // increment number of samples
  number_of_samples++;
  accumulated_time += time_step_size_for_sampling;

  dealii::FiniteElement<dim> const & fe_u = dof_handler_velocity.get_fe();

  if(rt_operator != nullptr)
  {
    velocity.update_ghost_values();
    dealii::AlignedVector<dealii::Tensor<1, dim, VectorizedArrayType>> eval_field(
      evaluated_dg_values_on_cells.size(1));
    for(unsigned int i = 0; i < list_of_cells_to_evaluate.size(); ++i)
    {
      rt_operator->evaluate_field(velocity, list_of_cells_to_evaluate[i], eval_field);
      std::array<unsigned int, VectorizedArrayType::size()> store_indices;
      for(unsigned int j = 0; j < store_indices.size(); ++j)
        store_indices[j] = j * dim * eval_field.size();
      dealii::vectorized_transpose_and_store(
        false,
        dim * eval_field.size(),
        &eval_field[0][0],
        store_indices.data(),
        &evaluated_dg_values_on_cells(i * VectorizedArrayType::size(), 0)[0]);
    }
  }
  else if(dof_indices_on_cell.empty())
  {
    dof_indices_on_cell.reserve(jacobians_at_nodal_points.size(0));
    std::vector<dealii::types::global_dof_index> dof_indices(fe_u.dofs_per_cell);
    const bool is_rt_element = fe_u.get_name().find("RaviartThomas") != std::string::npos;
    for(const auto & line_points : cells_and_ref_points)
      for(const auto & [cell, _] : line_points)
      {
        cell->get_dof_indices(dof_indices);
        std::array<unsigned int, 2 * dim + 1> indices;
        std::fill(indices.begin(), indices.end(), dealii::numbers::invalid_unsigned_int);
        if(is_rt_element)
        {
          for(unsigned int f = 0; f < 2 * dim + 1; ++f)
            indices[f] =
              velocity.get_partitioner()->global_to_local(dof_indices[f * fe_u.dofs_per_face]);
        }
        else
          indices[0] = velocity.get_partitioner()->global_to_local(dof_indices[0]);
        dof_indices_on_cell.push_back(indices);
      }
  }

  unsigned int total_length = 0;
  for(auto const & line : data.lines)
    total_length += line->n_points;

  std::vector<double>                                  length_local(total_length);
  std::vector<dealii::Tensor<1, dim, double>>          velocity_local(total_length);
  std::vector<double>                                  wall_shear_local(total_length);
  std::vector<dealii::SymmetricTensor<2, dim, double>> reynolds_local(total_length);
  std::vector<double>                                  pressure_local(total_length);
  std::vector<double>                                  reference_pressure_local(data.lines.size());
  std::vector<double>                                  dissipation_local(total_length);
  std::vector<double>                                  grid_size_local(total_length);

  // use quadrature for averaging in homogeneous direction
  const unsigned int              n_q_points_1d = fe_u.degree + 1;
  dealii::QGauss<1>               gauss_1d(n_q_points_1d);
  std::vector<dealii::Point<dim>> points;

  std::vector<dealii::Tensor<1, dim, Number>> velocity_dgq_on_cell(
    jacobians_at_nodal_points.size(1));
  std::vector<Number>                                              tmp_array;
  std::array<dealii::ndarray<VectorizedArrayType, 2, dim - 1>, 20> shapes_2d;

  const unsigned int n_points_in_plane = dealii::Utilities::pow(n_q_points_1d, dim - 1);
  std::vector<dealii::Tensor<1, dim, Number>>          cell_averaged_velocity;
  std::vector<dealii::SymmetricTensor<2, dim, Number>> cell_averaged_reynolds;

  if(rt_operator == nullptr)
    velocity.update_ghost_values();
  if(dof_handler_pressure.get_fe().dofs_per_vertex > 0)
    pressure.update_ghost_values();

  unsigned int counter_cells_p   = 0;
  unsigned int counter_all_cells = 0;
  unsigned int counter_line      = 0;
  unsigned int offset_arrays     = 0;
  for(unsigned int index = 0; index < data.lines.size(); ++index)
  {
    Line<dim> & line = *data.lines[index];

    bool evaluate_velocity      = false;
    bool need_velocity_gradient = false;
    bool need_skin_friction     = false;
    bool need_dissipation       = false;

    for(const std::shared_ptr<Quantity> & quantity : line.quantities)
    {
      // evaluate quantities that involve velocity
      if(quantity->type == QuantityType::Velocity or quantity->type == QuantityType::SkinFriction or
         quantity->type == QuantityType::ReynoldsStresses or
         quantity->type == QuantityType::Dissipation)
      {
        evaluate_velocity = true;
      }
      if(quantity->type == QuantityType::SkinFriction or
         quantity->type == QuantityType::Dissipation)
      {
        need_velocity_gradient = true;
      }
      if(quantity->type == QuantityType::SkinFriction)
      {
        need_skin_friction = true;
      }
      if(quantity->type == QuantityType::Dissipation)
      {
        need_dissipation = true;
      }
    }

    if(need_dissipation)
    {
      bool has_skin_friction_quantity = false;
      for(const auto & quantity : line.quantities)
        if(quantity->type == QuantityType::SkinFriction)
          has_skin_friction_quantity = true;

      AssertThrow(has_skin_friction_quantity,
                  dealii::ExcMessage(
                    "Dissipation requires QuantitySkinFriction to provide viscosity."));
    }


    // Do we want to perform averaging on the cell with tensor product first
    // (leads to small aliasing errors for Reynolds stresses, but is faster),
    // and then perform interpolation from 2D data, or do we rather want to
    // evaluate on the plane including the averaging direction first and then
    // perform averaging (more accurate, but also more expensive).
    constexpr unsigned int evaluate_averaging_by_tensor_product = false;

    if(evaluate_velocity == true)
    {
      for(auto const & [cell, point_list] : cells_and_ref_points[index])
      {
        dealii::Tensor<1, dim, Number> * eval_ptr = nullptr;
        if(rt_operator == nullptr)
        {
          const std::array<unsigned int, 2 * dim + 1> cell_indices =
            dof_indices_on_cell[counter_all_cells];
          // RT elements have all entries set, DG elements only first
          if(cell_indices[1] != dealii::numbers::invalid_unsigned_int)
          {
            read_rt_cell_values(fe_u.degree,
                                velocity.begin(),
                                shape_values_eo_n,
                                shape_values_eo_t,
                                cell_indices,
                                tmp_array,
                                velocity_dgq_on_cell);
            for(unsigned int q = 0; q < velocity_dgq_on_cell.size(); ++q)
            {
              const dealii::Tensor<2, dim> jac = jacobians_at_nodal_points(counter_all_cells, q);
              velocity_dgq_on_cell[q]          = (jac * velocity_dgq_on_cell[q]) / determinant(jac);
            }
          }
          else
            for(unsigned int c = 0; c < dim; ++c)
            {
              dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_evenodd,
                                                       dim,
                                                       0,
                                                       0,
                                                       Number,
                                                       Number>
                eval(
                  shape_values_eo_dgq.data(), nullptr, nullptr, fe_u.degree + 1, fe_u.degree + 1);
              eval.template values<0, true, false>(velocity.begin() + cell_indices[0] +
                                                     c * velocity_dgq_on_cell.size(),
                                                   tmp_array.data());
              if constexpr(dim == 3)
                eval.template values<2, true, false>(tmp_array.data(), tmp_array.data());
              eval.template values<1, true, false>(tmp_array.data(), &velocity_dgq_on_cell[0][c]);
            }
          eval_ptr = velocity_dgq_on_cell.data();
        }
        else
        {
          eval_ptr = &evaluated_dg_values_on_cells(
            active_cell_index_to_evaluate_index[cell->active_cell_index()], 0);
          if constexpr(evaluate_averaging_by_tensor_product)
          {
            cell_averaged_velocity.resize(n_points_in_plane);
            cell_averaged_reynolds.resize(n_points_in_plane);
            double h_z =
              1.0 /
              inverse_jacobians_on_lines[counter_line][averaging_direction][averaging_direction];
            std::fill(cell_averaged_velocity.begin(),
                      cell_averaged_velocity.end(),
                      dealii::Tensor<1, dim>());
            std::fill(cell_averaged_reynolds.begin(),
                      cell_averaged_reynolds.end(),
                      dealii::SymmetricTensor<2, dim>());
            for(unsigned int iz = 0, i = 0; iz < n_q_points_1d; ++iz)
              for(unsigned int i1 = 0; i1 < cell_averaged_velocity.size(); ++i1, ++i)
              {
                const Number JxW = (h_z * gauss_1d.weight(iz));
                cell_averaged_velocity[i1] += eval_ptr[i] * JxW;
                for(unsigned int d = 0; d < dim; ++d)
                  for(unsigned int e = d; e < dim; ++e)
                    cell_averaged_reynolds[i1][d][e] += eval_ptr[i][d] * JxW * eval_ptr[i][e];
              }
          }
        }

        // perform averaging in homogeneous direction. Currently, some
        // directions are hardcoded, so we can only support the last direction
        // here
        AssertThrow(averaging_direction == dim - 1, dealii::ExcNotImplemented());
        const unsigned int n_lanes  = VectorizedArrayType::size();
        const unsigned int n_points = point_list.size();
        const unsigned int n_chunks = (n_points + n_lanes - 1) / n_lanes;
        for(unsigned int p1_v = 0; p1_v < n_chunks; ++p1_v)
        {
          dealii::Point<dim - 1, VectorizedArrayType> point_on_line;
          dealii::Tensor<2, dim, VectorizedArrayType> inv_jac;
          for(unsigned int d = 0; d < dim; ++d)
            inv_jac[d][d] = 1.0;
          for(unsigned int p1 = p1_v * n_lanes, v = 0;
              p1 < std::min((p1_v + 1) * n_lanes, n_points);
              ++p1, ++v)
          {
            for(unsigned int d = 0, c = 0; d < dim; ++d)
              if(d != averaging_direction)
                point_on_line[c++][v] = point_list[p1].second[d];
            dealii::Tensor<2, dim> const inv_jac_v = inverse_jacobians_on_lines[counter_line++];
            for(unsigned int d = 0; d < dim; ++d)
              for(unsigned int e = 0; e < dim; ++e)
                inv_jac[d][e][v] = inv_jac_v[d][e];
          }
          // bool                                    need_skin_friction = false;

          dealii::Tensor<1, dim, VectorizedArrayType> normal;
          dealii::Tensor<1, dim, VectorizedArrayType> tangent;

          Number viscosity = 0.0;
          if(need_skin_friction || need_dissipation)
          {
            bool found_skin_friction = false;
            for(const auto & quantity : line.quantities)
              if(quantity->type == QuantityType::SkinFriction)
              {
                auto q = std::dynamic_pointer_cast<QuantitySkinFriction<dim>>(quantity);
                AssertThrow(q, dealii::ExcInternalError());

                dealii::Tensor<2, dim, VectorizedArrayType> jac = invert(transpose(inv_jac));
                tangent                                         = jac * q->tangent_vector;
                tangent /= tangent.norm();

                if(averaging_direction == 2)
                {
                  normal[0] = tangent[1];
                  normal[1] = -tangent[0];
                }
                else
                  AssertThrow(false, dealii::ExcNotImplemented());

                viscosity           = q->viscosity;
                found_skin_friction = true;
              }
            AssertThrow(found_skin_friction,
                        dealii::ExcMessage(
                          "Dissipation/SkinFriction requires QuantitySkinFriction"));
          }


          VectorizedArrayType const det =
            1.0 / std::abs(inv_jac[averaging_direction][averaging_direction]);
          dealii::internal::compute_values_of_array(shapes_2d.data(),
                                                    polynomials_nodal,
                                                    point_on_line);

          const VectorizedArrayType                            length = det;
          dealii::Tensor<1, dim, VectorizedArrayType>          vel;
          dealii::SymmetricTensor<2, dim, VectorizedArrayType> reynolds;
          VectorizedArrayType                                  skin_friction = 0;
          VectorizedArrayType                                  dissipation   = 0.0;
          if constexpr(!evaluate_averaging_by_tensor_product)
          {
            dealii::Tensor<1, dim, VectorizedArrayType> velocity;
            dealii::Tensor<2, dim, VectorizedArrayType> velocity_gradient;
            for(unsigned int q1 = 0; q1 < n_q_points_1d; ++q1)
            {
              VectorizedArrayType const JxW = det * gauss_1d.weight(q1);
              if(need_velocity_gradient)
              {
                auto const val_grad =
                  dealii::internal::evaluate_tensor_product_value_and_gradient_shapes<
                    dim - 1,
                    dealii::Tensor<1, dim, Number>,
                    VectorizedArrayType>(shapes_2d.data(),
                                         polynomials_nodal.size(),
                                         eval_ptr + q1 * n_points_in_plane);
                velocity = val_grad[dim - 1];
                for(unsigned int d = 0; d < dim; ++d)
                {
                  for(unsigned int e = 0; e < dim - 1; ++e)
                    velocity_gradient[d][e] = val_grad[e][d];
                  velocity_gradient[d] = inv_jac * velocity_gradient[d];
                }

                if(need_dissipation)
                {
                  VectorizedArrayType epsilon_q = 0.0;
                  for(unsigned int i = 0; i < dim; ++i)
                    for(unsigned int j = 0; j < dim; ++j)
                      epsilon_q += velocity_gradient[i][j] * velocity_gradient[i][j];
                  epsilon_q *= viscosity;
                  dissipation += epsilon_q * JxW;
                }

                if(need_skin_friction)
                {
                  for(unsigned int d = 0; d < dim; ++d)
                    for(unsigned int e = 0; e < dim; ++e)
                      skin_friction += tangent[d] * velocity_gradient[d][e] * (normal[e] * JxW);
                }
              }
              else
                velocity = dealii::internal::evaluate_tensor_product_value_shapes<
                  dim - 1,
                  dealii::Tensor<1, dim, Number>,
                  VectorizedArrayType>(shapes_2d.data(),
                                       polynomials_nodal.size(),
                                       eval_ptr + q1 * n_points_in_plane);

              for(const std::shared_ptr<Quantity> & quantity : line.quantities)
              {
                if(quantity->type == QuantityType::Velocity)
                {
                  for(unsigned int d = 0; d < dim; ++d)
                    vel[d] += velocity[d] * JxW;
                }
                else if(quantity->type == QuantityType::ReynoldsStresses)
                {
                  for(unsigned int d = 0; d < dim; ++d)
                    for(unsigned int e = d; e < dim; ++e)
                      reynolds[d][e] += (velocity[d] * JxW) * velocity[e];
                }
                // else if(quantity->type == QuantityType::SkinFriction)
                // {
                //   for(unsigned int d = 0; d < dim; ++d)
                //     for(unsigned int e = 0; e < dim; ++e)
                //       skin_friction += tangent[d] * velocity_gradient[d][e] * (normal[e] * JxW);
                // }
              }
            }
          }
          else
          {
            if(need_velocity_gradient)
            {
              auto const val_grad =
                dealii::internal::evaluate_tensor_product_value_and_gradient_shapes<
                  dim - 1,
                  dealii::Tensor<1, dim, Number>,
                  VectorizedArrayType>(shapes_2d.data(),
                                       polynomials_nodal.size(),
                                       cell_averaged_velocity.data());
              vel = val_grad[dim - 1];
              dealii::Tensor<2, dim, VectorizedArrayType> grad;
              for(unsigned int d = 0; d < dim; ++d)
              {
                for(unsigned int e = 0; e < dim - 1; ++e)
                  grad[d][e] = val_grad[e][d];
                grad[d] = inv_jac * grad[d];
              }
              if(need_skin_friction)
              {
                for(unsigned int d = 0; d < dim; ++d)
                  for(unsigned int e = 0; e < dim; ++e)
                    skin_friction += tangent[d] * grad[d][e] * normal[e];
              }
            }
            else
              vel = dealii::internal::evaluate_tensor_product_value_shapes<
                dim - 1,
                dealii::Tensor<1, dim, Number>,
                VectorizedArrayType>(shapes_2d.data(),
                                     polynomials_nodal.size(),
                                     cell_averaged_velocity.data());
            reynolds = dealii::internal::evaluate_tensor_product_value_shapes<
              dim - 1,
              dealii::SymmetricTensor<2, dim, Number>,
              VectorizedArrayType>(shapes_2d.data(),
                                   polynomials_nodal.size(),
                                   cell_averaged_reynolds.data());
          }

          for(unsigned int p1 = p1_v * n_lanes, v = 0;
              p1 < std::min((p1_v + 1) * n_lanes, n_points);
              ++p1, ++v)
          {
            const unsigned int p = point_list[p1].first;
            // calculate integrals in homogeneous direction
            length_local[offset_arrays + p] += length[v];

            for(const std::shared_ptr<Quantity> & quantity : line.quantities)
            {
              if(quantity->type == QuantityType::Velocity)
              {
                for(unsigned int d = 0; d < dim; ++d)
                  velocity_local[offset_arrays + p][d] += vel[d][v];
              }
              else if(quantity->type == QuantityType::ReynoldsStresses)
              {
                for(unsigned int d = 0; d < dim; ++d)
                  for(unsigned int e = d; e < dim; ++e)
                    reynolds_local[offset_arrays + p][d][e] += reynolds[d][e][v];
              }
              else if(quantity->type == QuantityType::SkinFriction)
              {
                wall_shear_local[offset_arrays + p] += skin_friction[v];
              }
              else if(quantity->type == QuantityType::Dissipation)
              {
                dissipation_local[offset_arrays + p] += dissipation[v];
                VectorizedArrayType local_he =
                  std::pow(Number(1.0) / std::abs(determinant(inv_jac)), Number(1.0 / 3.0));

                grid_size_local[offset_arrays + p] += local_he[v] * det[v];
              }
            }
          }
        }
        ++counter_all_cells;
      }
    }

    for(const std::shared_ptr<Quantity> & quantity : line.quantities)
    {
      // evaluate quantities that involve velocity
      if(quantity->type == QuantityType::Pressure)
      {
        for(auto const & [cell, point_list] : cells_and_ref_points[index])
        {
          evaluator_p->reinit(counter_cells_p);
          ++counter_cells_p;
          evaluator_p->evaluate(dealii::ArrayView<const Number>(
                                  pressure.begin() +
                                    pressure_dof_indices_on_cell[cell->active_cell_index()],
                                  dof_handler_pressure.get_fe().dofs_per_cell),
                                dealii::EvaluationFlags::values);

          for(unsigned int p1 = 0, q = 0; p1 < point_list.size(); ++p1)
            for(unsigned int q1 = 0; q1 < gauss_1d.size(); ++q1, ++q)
            {
              unsigned int const p = point_list[p1].first;

              double det =
                std::abs(evaluator_p->jacobian(q)[averaging_direction][averaging_direction]);
              double JxW = det * gauss_1d.weight(q1);

              if(not evaluate_velocity)
                length_local[offset_arrays + p] += JxW;
              pressure_local[offset_arrays + p] += evaluator_p->get_value(q) * JxW;
            }
        }
      }
      if(quantity->type == QuantityType::PressureCoefficient)
      {
        double length_local   = 0.0;
        double pressure_local = 0.0;

        TYPE vector_cells_and_ref_points = cells_and_ref_points_ref_pressure[index];

        average_pressure_for_given_point(pressure,
                                         vector_cells_and_ref_points,
                                         length_local,
                                         pressure_local);

        // MPI communication
        length_local   = dealii::Utilities::MPI::sum(length_local, mpi_comm);
        pressure_local = dealii::Utilities::MPI::sum(pressure_local, mpi_comm);

        // averaging in space (over homogeneous direction)
        reference_pressure_global[index] +=
          (pressure_local / length_local) * time_step_size_for_sampling;
      }
    }
    offset_arrays += line.n_points;
  }

  velocity.zero_out_ghost_values();
  if(dof_handler_pressure.get_fe().dofs_per_vertex > 0)
    pressure.zero_out_ghost_values();

  offset_arrays = 0;
  for(unsigned int line = 0; line < data.lines.size(); ++line)
  {
    const unsigned int n_points_on_line = data.lines[line]->n_points;
    mpi_sum_at_root(length_local.data() + offset_arrays, n_points_on_line, mpi_comm);
    for(const std::shared_ptr<Quantity> & quantity : data.lines[line]->quantities)
    {
      // Cells are distributed over processors, therefore we need
      // to sum the contributions of every single processor.
      if(quantity->type == QuantityType::Velocity)
      {
        mpi_sum_at_root(&velocity_local[offset_arrays][0], n_points_on_line * dim, mpi_comm);

        if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
          for(unsigned int p = 0; p < n_points_on_line; ++p)
          {
            velocity_global[line][p] +=
              (velocity_local[offset_arrays + p] / length_local[offset_arrays + p]) *
              time_step_size_for_sampling;
          }
      }
      else if(quantity->type == QuantityType::ReynoldsStresses)
      {
        mpi_sum_at_root(&reynolds_local[offset_arrays][0][0],
                        n_points_on_line * dim * (dim + 1) / 2,
                        mpi_comm);

        if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
          for(unsigned int p = 0; p < n_points_on_line; ++p)
          {
            reynolds_global[line][p] +=
              (reynolds_local[offset_arrays + p] / length_local[offset_arrays + p]) *
              time_step_size_for_sampling;
          }
      }
      else if(quantity->type == QuantityType::SkinFriction)
      {
        mpi_sum_at_root(wall_shear_local.data() + offset_arrays, n_points_on_line, mpi_comm);

        if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
          for(unsigned int p = 0; p < n_points_on_line; ++p)
          {
            wall_shear_global[line][p] +=
              (wall_shear_local[offset_arrays + p] / length_local[offset_arrays + p]) *
              time_step_size_for_sampling;
          }
      }
      else if(quantity->type == QuantityType::Pressure)
      {
        mpi_sum_at_root(pressure_local.data() + offset_arrays, n_points_on_line, mpi_comm);

        // averaging in space (over homogeneous direction)
        if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
          for(unsigned int p = 0; p < n_points_on_line; ++p)
            pressure_global[line][p] +=
              (pressure_local[offset_arrays + p] / length_local[offset_arrays + p]) *
              time_step_size_for_sampling;
      }
      else if(quantity->type == QuantityType::Dissipation)
      {
        mpi_sum_at_root(dissipation_local.data() + offset_arrays, n_points_on_line, mpi_comm);
        mpi_sum_at_root(grid_size_local.data() + offset_arrays, n_points_on_line, mpi_comm);

        if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
          for(unsigned int p = 0; p < n_points_on_line; ++p)
          {
            dissipation_global[line][p] +=
              (dissipation_local[offset_arrays + p] / length_local[offset_arrays + p]) *
              time_step_size_for_sampling;
            grid_size_global[line][p] +=
              (grid_size_local[offset_arrays + p] / length_local[offset_arrays + p]) *
              time_step_size_for_sampling;
          }
      }
    }
    offset_arrays += n_points_on_line;
  }

  time_all += time.wall_time();
}

template<int dim, typename Number>
void
LinePlotCalculatorStatisticsHomogeneous<dim, Number>::average_pressure_for_given_point(
  VectorType const & pressure,
  TYPE const &       vector_cells_and_ref_points,
  double &           length_local,
  double &           pressure_local)
{
  dealii::FiniteElement<dim> const &           fe = dof_handler_pressure.get_fe();
  std::vector<double>                          pressure_on_cell(fe.dofs_per_cell);
  std::vector<dealii::types::global_dof_index> dof_indices(fe.dofs_per_cell);

  // use quadrature for averaging in homogeneous direction
  dealii::QGauss<1>                 gauss_1d(fe.degree + 1);
  std::vector<dealii::Point<dim>>   points(gauss_1d.size());  // 1D points
  std::vector<double>               weights(gauss_1d.size()); // 1D weights
  dealii::FEPointEvaluation<1, dim> evaluator(mapping,
                                              fe,
                                              dealii::update_values | dealii::update_jacobians |
                                                dealii::update_quadrature_points |
                                                dealii::update_gradients);

  for(typename TYPE::const_iterator cell_and_ref_point = vector_cells_and_ref_points.begin();
      cell_and_ref_point != vector_cells_and_ref_points.end();
      ++cell_and_ref_point)
  {
    dealii::Point<dim> const p_unit = cell_and_ref_point->second;

    // Find points for Gauss quadrature
    for(unsigned int i = 0; i < gauss_1d.size(); ++i)
      for(unsigned int d = 0; d < dim; ++d)
        points[i][d] = (d == averaging_direction) ? gauss_1d.point(i)[0] : p_unit[d];

    typename dealii::DoFHandler<dim>::active_cell_iterator const cell = cell_and_ref_point->first;
    evaluator.reinit(cell, points);
    cell->get_dof_indices(dof_indices);

    for(unsigned int j = 0; j < dof_indices.size(); ++j)
      pressure_on_cell[j] = pressure(dof_indices[j]);
    evaluator.evaluate(pressure_on_cell,
                       dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

    for(unsigned int q = 0; q < points.size(); ++q)
    {
      double const p = evaluator.get_value(q);

      double det = std::abs(evaluator.jacobian(q)[averaging_direction][averaging_direction]);
      double JxW = det * gauss_1d.weight(q);

      length_local += JxW;
      pressure_local += p * JxW;
    }
  }
}

template<int dim, typename Number>
void
LinePlotCalculatorStatisticsHomogeneous<dim, Number>::do_write_output() const
{
  if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
  {
    dealii::Timer      time;
    unsigned int const precision = data.precision;

    // Iterator for lines
    unsigned int line_iterator = 0;
    for(const std::shared_ptr<Line<dim>> & line : data.lines)
    {
      std::string filename_prefix = data.directory + line->name;

      for(const std::shared_ptr<Quantity> & quantity : line->quantities)
      {
        if(quantity->type == QuantityType::Velocity)
        {
          std::string   filename = filename_prefix + "_velocity" + ".txt";
          std::ofstream f;
          if(clear_files)
          {
            f.open(filename.c_str(), std::ios::trunc);
          }
          else
          {
            f.open(filename.c_str(), std::ios::app);
          }

          print_headline(f, number_of_samples);

          for(unsigned int d = 0; d < dim; ++d)
            f << std::setw(precision + 8) << std::left
              << "x_" + dealii::Utilities::int_to_string(d + 1);
          for(unsigned int d = 0; d < dim; ++d)
            f << std::setw(precision + 8) << std::left
              << "u_" + dealii::Utilities::int_to_string(d + 1);

          f << std::endl;

          // loop over all points
          for(unsigned int p = 0; p < line->n_points; ++p)
          {
            f << std::scientific << std::setprecision(precision);

            // write data
            for(unsigned int d = 0; d < dim; ++d)
              f << std::setw(precision + 8) << std::left << global_points[line_iterator][p][d];

            // write velocity and average over time
            for(unsigned int d = 0; d < dim; ++d)
              f << std::setw(precision + 8) << std::left
                << velocity_global[line_iterator][p][d] / accumulated_time;

            f << std::endl;
          }
          f.close();
        }

        if(quantity->type == QuantityType::ReynoldsStresses)
        {
          std::string   filename = filename_prefix + "_reynoldsstresses" + ".txt";
          std::ofstream f;
          if(clear_files)
          {
            f.open(filename.c_str(), std::ios::trunc);
          }
          else
          {
            f.open(filename.c_str(), std::ios::app);
          }

          print_headline(f, number_of_samples);

          for(unsigned int d = 0; d < dim; ++d)
            f << std::setw(precision + 8) << std::left
              << "x_" + dealii::Utilities::int_to_string(d + 1);

          for(unsigned int i = 0; i < dim; ++i)
          {
            for(unsigned int j = i; j < dim; ++j)
            {
              f << std::setw(precision + 8) << std::left
                << "u_" + dealii::Utilities::int_to_string(i + 1) + "u_" +
                     dealii::Utilities::int_to_string(j + 1);
            }
          }
          f << std::endl;

          // loop over all points
          for(unsigned int p = 0; p < line->n_points; ++p)
          {
            f << std::scientific << std::setprecision(precision);

            for(unsigned int d = 0; d < dim; ++d)
              f << std::setw(precision + 8) << std::left << global_points[line_iterator][p][d];

            for(unsigned int i = 0; i < dim; ++i)
            {
              for(unsigned int j = i; j < dim; ++j)
              {
                // equation <u_i' u_j'> = <u_i*u_j> - <u_i> * <u_j>
                f << std::setw(precision + 8) << std::left
                  << reynolds_global[line_iterator][p][i][j] / accumulated_time -
                       (velocity_global[line_iterator][p][i] / accumulated_time) *
                         (velocity_global[line_iterator][p][j] / accumulated_time);
              }
            }

            f << std::endl;
          }
          f.close();
        }

        if(quantity->type == QuantityType::SkinFriction)
        {
          std::shared_ptr<QuantitySkinFriction<dim>> averaging_quantity =
            std::dynamic_pointer_cast<QuantitySkinFriction<dim>>(quantity);

          std::string   filename = filename_prefix + "_wall_shear_stress" + ".txt";
          std::ofstream f;
          if(clear_files)
          {
            f.open(filename.c_str(), std::ios::trunc);
          }
          else
          {
            f.open(filename.c_str(), std::ios::app);
          }

          print_headline(f, number_of_samples);

          for(unsigned int d = 0; d < dim; ++d)
            f << std::setw(precision + 8) << std::left
              << "x_" + dealii::Utilities::int_to_string(d + 1);

          f << std::setw(precision + 8) << std::left << "tau_w" << std::endl;

          // loop over all points
          for(unsigned int p = 0; p < line->n_points; ++p)
          {
            f << std::scientific << std::setprecision(precision);

            // write data
            for(unsigned int d = 0; d < dim; ++d)
              f << std::setw(precision + 8) << std::left << global_points[line_iterator][p][d];

            // tau_w -> C_f = tau_w / (1/2 rho u²)
            double const viscosity = averaging_quantity->viscosity;
            f << std::setw(precision + 8) << std::left
              << viscosity * wall_shear_global[line_iterator][p] / accumulated_time;

            f << std::endl;
          }
          f.close();
        }

        if(quantity->type == QuantityType::Pressure or
           quantity->type == QuantityType::PressureCoefficient)
        {
          std::string   filename = filename_prefix + "_pressure" + ".txt";
          std::ofstream f;

          if(clear_files)
          {
            f.open(filename.c_str(), std::ios::trunc);
          }
          else
          {
            f.open(filename.c_str(), std::ios::app);
          }

          print_headline(f, number_of_samples);

          for(unsigned int d = 0; d < dim; ++d)
            f << std::setw(precision + 8) << std::left
              << "x_" + dealii::Utilities::int_to_string(d + 1);

          f << std::setw(precision + 8) << std::left << "p";

          if(quantity->type == QuantityType::PressureCoefficient)
            f << std::setw(precision + 8) << std::left << "p-p_ref";

          f << std::endl;

          for(unsigned int p = 0; p < line->n_points; ++p)
          {
            f << std::scientific << std::setprecision(precision);

            for(unsigned int d = 0; d < dim; ++d)
              f << std::setw(precision + 8) << std::left << global_points[line_iterator][p][d];

            f << std::setw(precision + 8) << std::left
              << pressure_global[line_iterator][p] / accumulated_time;

            if(quantity->type == QuantityType::PressureCoefficient)
            {
              // p - p_ref -> C_p = (p - p_ref) / (1/2 rho u²)
              f << std::left
                << (pressure_global[line_iterator][p] - reference_pressure_global[line_iterator]) /
                     accumulated_time;
            }
            f << std::endl;
          }
          f.close();
        }

        if(quantity->type == QuantityType::Dissipation)
        {
          std::string   filename = filename_prefix + "_dissipation" + ".txt";
          std::ofstream f;
          if(clear_files)
          {
            f.open(filename.c_str(), std::ios::trunc);
          }
          else
          {
            f.open(filename.c_str(), std::ios::app);
          }

          print_headline(f, number_of_samples);

          for(unsigned int d = 0; d < dim; ++d)
            f << std::setw(precision + 8) << std::left
              << "x_" + dealii::Utilities::int_to_string(d + 1);

          f << std::setw(precision + 8) << std::left << "epsilon";
          f << std::setw(precision + 8) << std::left << "h_e" << std::endl;

          // loop over all points
          for(unsigned int p = 0; p < line->n_points; ++p)
          {
            f << std::scientific << std::setprecision(precision);

            // write data
            for(unsigned int d = 0; d < dim; ++d)
              f << std::setw(precision + 8) << std::left << global_points[line_iterator][p][d];

            // write dissipation and average over time
            f << std::setw(precision + 8) << std::left
              << dissipation_global[line_iterator][p] / accumulated_time;

            // write grid size and average over time
            f << std::setw(precision + 8) << std::left
              << grid_size_global[line_iterator][p] / accumulated_time;

            f << std::endl;
          }
          f.close();
        }
      }
      ++line_iterator;
    }
    const_cast<double &>(time_all) += time.wall_time();
    std::cout << "Accumulated " << number_of_samples
              << " samples on lines in a compute time of t = " << time_all << " s" << std::endl;
  }
}

template class LinePlotCalculatorStatisticsHomogeneous<2, float>;
template class LinePlotCalculatorStatisticsHomogeneous<3, float>;

template class LinePlotCalculatorStatisticsHomogeneous<2, double>;
template class LinePlotCalculatorStatisticsHomogeneous<3, double>;

} // namespace IncNS
} // namespace ExaDG
