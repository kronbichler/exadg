/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2025 by Martin Kronbichler, Shubham Goswami,
 *  Richard Schussnig
 *
 *  This file is dual-licensed under the Apache-2.0 with LLVM Exception (see
 *  https://spdx.org/licenses/Apache-2.0.html and
 *  https://spdx.org/licenses/LLVM-exception.html) and the GNU General Public
 *  License as published by the Free Software Foundation, either version 3 of
 *  the License, or (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License in the top-level LICENSE file for
 *  more details.
 *  ______________________________________________________________________
 */

#pragma once

#include <deal.II/base/floating_point_comparator.h>
#include <deal.II/base/memory_space_data.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/tensor_product_matrix.h>
#include <deal.II/matrix_free/evaluation_kernels.h>
#include <deal.II/matrix_free/evaluation_kernels_hanging_nodes.h>
#include <deal.II/matrix_free/evaluation_template_factory.h>
#include <deal.II/matrix_free/hanging_nodes_internal.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tensor_product_kernels.h>
#include <deal.II/matrix_free/vector_access_internal.h>
#include <deal.II/matrix_free/vector_data_exchange.h>

#include <map>
#include <numeric>
#include <random>
#include <set>

#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/extruded_mapping_info.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/face_integral_helper.h>
#include <exadg/solvers_and_preconditioners/preconditioners/scratch_matrix_vector_product.h>


namespace LaplaceOperator
{
using namespace dealii;

template<int n_q_points_1d, int dim, typename Number>
void
compute_cell_lapl(const internal::MatrixFreeFunctions::UnivariateShapeData<Number> & shape_data,
                  const Extruded::MappingInfo<dim, Number> &                         mapping_info,
                  const unsigned int                                                 cell,
                  const Number                                                       factor_mass,
                  const Number                                                       factor_lapl,
                  const VectorizedArray<Number> *                                    quad_values,
                  VectorizedArray<Number> *                                          out_values)
{
  constexpr unsigned int n_q_points_2d = n_q_points_1d * n_q_points_1d;
  constexpr unsigned int n_q_points    = Utilities::pow(n_q_points_1d, dim);
  constexpr unsigned int n_lanes       = VectorizedArray<Number>::size();

  VectorizedArray<Number> grad_y[n_q_points];
  VectorizedArray<Number> grad_x[Utilities::pow(n_q_points_1d, dim - 1)];
  VectorizedArray<Number> grad_z[n_q_points_1d];

  std::array<unsigned int, n_lanes> shifted_data_indices;
  for(unsigned int v = 0; v < n_lanes; ++v)
    shifted_data_indices[v] = mapping_info.mapping_data_index[cell][v] * 4;

  // the extent in z direction can differ between cells on adaptively refined
  // meshes, so load it per lane
  [[maybe_unused]] VectorizedArray<Number> h_z, h_z_inverse;
  if constexpr(dim == 3)
  {
    std::array<unsigned int, n_lanes> geometry_indices;
    for(unsigned int v = 0; v < n_lanes; ++v)
      geometry_indices[v] = mapping_info.mapping_data_index[cell][v] / n_q_points_2d;
    h_z.gather(mapping_info.h_z_values.data(), geometry_indices.data());
    h_z_inverse = Number(1) / h_z;
  }

  constexpr unsigned int nn = n_q_points_1d;

  // Create a local copy to a stack variable of the shape gradients, such that
  // the compiler can decide on the most optimal placement of variables in
  // registers, rather than having to consider potential aliasing between the
  // 'shape_gradients_collocation_eo' variable and the other variables used
  // during sum factorization.
  VectorizedArray<Number> shape_grad[nn * ((nn + 1) / 2)];
  AssertDimension(shape_data.shape_gradients_collocation_eo.size(), nn * ((nn + 1) / 2));
  for(unsigned int i = 0; i < nn * ((nn + 1) / 2); ++i)
    shape_grad[i] = shape_data.shape_gradients_collocation_eo[i];

  for(unsigned int i1 = 0; i1 < (dim == 3 ? nn : 1); ++i1)
    for(unsigned int i0 = 0; i0 < nn; ++i0)
      internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                            internal::EvaluatorQuantity::gradient,
                                            nn,
                                            nn,
                                            nn,
                                            Utilities::pow(nn, dim - 1),
                                            true,
                                            false>(shape_grad,
                                                   quad_values + (i1 * n_q_points_2d + i0),
                                                   grad_y + (i1 * nn + i0));

  for(unsigned int qy = 0, q1 = 0; qy < nn; ++qy)
  {
    for(unsigned int i0 = 0; i0 < (dim == 3 ? nn : 1); ++i0)
      internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                            internal::EvaluatorQuantity::gradient,
                                            nn,
                                            nn,
                                            1,
                                            (dim == 3 ? nn : 1),
                                            true,
                                            false>(shape_grad,
                                                   quad_values + (qy * nn + i0 * n_q_points_2d),
                                                   grad_x + i0);

    for(unsigned int qx = 0; qx < nn; ++qx, ++q1)
    {
      Tensor<2, 2, VectorizedArray<Number>> inv_jac_xy;
      vectorized_load_and_transpose(4,
                                    &mapping_info.inv_jacobians_xy[q1][0][0],
                                    shifted_data_indices.data(),
                                    &inv_jac_xy[0][0]);
      VectorizedArray<Number> JxW_xy;
      JxW_xy.gather(&mapping_info.cell_JxW_xy[q1], mapping_info.mapping_data_index[cell].data());

      if constexpr(dim == 2)
      {
        const VectorizedArray<Number> val     = quad_values[q1];
        const VectorizedArray<Number> grad[2] = {grad_x[qx], grad_y[q1]};

        VectorizedArray<Number> grad_real[dim];
        for(unsigned int d = 0; d < dim; ++d)
          grad_real[d] = inv_jac_xy[d][0] * grad[0] + inv_jac_xy[d][1] * grad[1];

        for(unsigned int e = 0; e < dim; ++e)
        {
          grad_real[e] *= JxW_xy * factor_lapl;
        }

        grad_x[qx]     = inv_jac_xy[0][0] * grad_real[0] + inv_jac_xy[1][0] * grad_real[1];
        grad_y[q1]     = inv_jac_xy[0][1] * grad_real[0] + inv_jac_xy[1][1] * grad_real[1];
        out_values[q1] = val * (JxW_xy * factor_mass);
      }
      else // now to dim == 3
      {
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::gradient,
                                              nn,
                                              nn,
                                              nn * nn,
                                              1,
                                              true,
                                              false>(shape_grad, quad_values + q1, grad_z);

        const VectorizedArray<Number> JxW_h_z = JxW_xy * h_z;
        for(unsigned int qz = 0, q = q1; qz < nn; ++qz, q += n_q_points_2d)
        {
          const VectorizedArray<Number> val     = quad_values[q];
          const VectorizedArray<Number> grad[3] = {grad_x[qz + qx * nn],
                                                   grad_y[qz * nn + qy * nn * nn + qx],
                                                   grad_z[qz]};

          VectorizedArray<Number> grad_real[dim];
          for(unsigned int d = 0; d < 2; ++d)
            grad_real[d] = inv_jac_xy[d][0] * grad[0] + inv_jac_xy[d][1] * grad[1];
          grad_real[2] = h_z_inverse * grad[2];

          const Number                  weight_z     = mapping_info.quad_weights_z[qz];
          const VectorizedArray<Number> factor_deriv = (JxW_h_z * factor_lapl) * weight_z;
          const VectorizedArray<Number> factor_ma    = (JxW_h_z * factor_mass) * weight_z;

          out_values[q] = val * factor_ma;
          for(unsigned int e = 0; e < dim; ++e)
          {
            grad_real[e] *= factor_deriv;
          }

          grad_x[qz + qx * nn] = inv_jac_xy[0][0] * grad_real[0] + inv_jac_xy[1][0] * grad_real[1];
          grad_y[qz * nn + qy * nn * nn + qx] =
            inv_jac_xy[0][1] * grad_real[0] + inv_jac_xy[1][1] * grad_real[1];
          grad_z[qz] = h_z_inverse * grad_real[2];
        }

        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::gradient,
                                              nn,
                                              nn,
                                              1,
                                              nn * nn,
                                              false,
                                              true>(shape_grad, grad_z, out_values + q1);
      }
    }

    for(unsigned int i0 = 0; i0 < (dim == 3 ? nn : 1); ++i0)
      internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                            internal::EvaluatorQuantity::gradient,
                                            nn,
                                            nn,
                                            (dim == 3 ? nn : 1),
                                            1,
                                            false,
                                            true>(shape_grad,
                                                  grad_x + i0,
                                                  out_values + qy * nn + i0 * n_q_points_2d);
  }

  for(unsigned int i1 = 0; i1 < (dim == 3 ? nn : 1); ++i1)
    for(unsigned int i0 = 0; i0 < nn; ++i0)
      internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                            internal::EvaluatorQuantity::gradient,
                                            nn,
                                            nn,
                                            Utilities::pow(nn, dim - 1),
                                            nn,
                                            false,
                                            true>(shape_grad,
                                                  grad_y + i1 * nn + i0,
                                                  out_values + i1 * n_q_points_2d + i0);
}



// Diagonal of the cell integrals of compute_cell_lapl() for the tensor
// product basis with shape_data.fe_degree + 1 = n_q_points_1d functions per
// direction, in lexicographic order. Rather than applying the cell operator
// to unit vectors, we use that the diagonal entry of basis function
// phi_i0(x) phi_i1(y) phi_i2(z) only involves the squares phi^2, phi phi'
// and phi'^2 of the 1D functions: the xy part is the contraction of the
// metric terms at the xy quadrature points with tables of these 1D products
// (sum factorization in x, then y), and the geometry does not depend on z,
// so the z part is a weighted 1D integral. If given, z_face_terms[i2] is
// added to the factor of the integral of JxW_xy phi_i0^2 phi_i1^2 (this is
// the form of the face integrals on faces with normal in z direction).
template<int n_q_points_1d, int dim, typename Number>
void
compute_cell_lapl_diagonal(
  const internal::MatrixFreeFunctions::UnivariateShapeData<Number> & shape_data,
  const Extruded::MappingInfo<dim, Number> &                         mapping_info,
  const unsigned int                                                 cell,
  const Number                                                       factor_mass,
  const Number                                                       factor_lapl,
  const VectorizedArray<Number> *                                    z_face_terms,
  VectorizedArray<Number> *                                          diagonal)
{
  static_assert(dim == 2 || dim == 3, "Only implemented for dim = 2, 3");
  constexpr unsigned int nn            = n_q_points_1d;
  constexpr unsigned int n_q_points_2d = nn * nn;
  constexpr unsigned int n_lanes       = VectorizedArray<Number>::size();
  AssertDimension(shape_data.fe_degree + 1, nn);
  AssertDimension(shape_data.n_q_points_1d, nn);

  // 1D products of shape functions at the quadrature points: [0] phi phi,
  // [1] phi phi', [2] phi' phi'
  Number products[3][nn * nn];
  for(unsigned int i = 0; i < nn; ++i)
    for(unsigned int q = 0; q < nn; ++q)
    {
      const Number value      = shape_data.shape_values[i * nn + q];
      const Number derivative = shape_data.shape_gradients[i * nn + q];
      products[0][i * nn + q] = value * value;
      products[1][i * nn + q] = value * derivative;
      products[2][i * nn + q] = derivative * derivative;
    }

  std::array<unsigned int, n_lanes> shifted_data_indices;
  for(unsigned int v = 0; v < n_lanes; ++v)
    shifted_data_indices[v] = mapping_info.mapping_data_index[cell][v] * 4;

  // coefficients at the xy quadrature points of the terms phi'phi' x phi phi,
  // phi phi' x phi phi' and phi phi x phi'phi' of the xy gradient part as
  // well as of the term phi phi x phi phi, the latter without factors
  VectorizedArray<Number>                  coefficients[4][n_q_points_2d];
  [[maybe_unused]] VectorizedArray<Number> h_z, h_z_inverse;
  if constexpr(dim == 3)
  {
    std::array<unsigned int, n_lanes> geometry_indices;
    for(unsigned int v = 0; v < n_lanes; ++v)
      geometry_indices[v] = mapping_info.mapping_data_index[cell][v] / n_q_points_2d;
    h_z.gather(mapping_info.h_z_values.data(), geometry_indices.data());
    h_z_inverse = Number(1) / h_z;
  }
  for(unsigned int q1 = 0; q1 < n_q_points_2d; ++q1)
  {
    Tensor<2, 2, VectorizedArray<Number>> inv_jac_xy;
    vectorized_load_and_transpose(4,
                                  &mapping_info.inv_jacobians_xy[q1][0][0],
                                  shifted_data_indices.data(),
                                  &inv_jac_xy[0][0]);
    VectorizedArray<Number> JxW_xy;
    JxW_xy.gather(&mapping_info.cell_JxW_xy[q1], mapping_info.mapping_data_index[cell].data());

    VectorizedArray<Number> factor = JxW_xy * factor_lapl;
    if constexpr(dim == 3)
      factor *= h_z;
    // metric tensor inv_jac_xy^T inv_jac_xy
    const VectorizedArray<Number> metric_00 =
      inv_jac_xy[0][0] * inv_jac_xy[0][0] + inv_jac_xy[1][0] * inv_jac_xy[1][0];
    const VectorizedArray<Number> metric_01 =
      inv_jac_xy[0][0] * inv_jac_xy[0][1] + inv_jac_xy[1][0] * inv_jac_xy[1][1];
    const VectorizedArray<Number> metric_11 =
      inv_jac_xy[0][1] * inv_jac_xy[0][1] + inv_jac_xy[1][1] * inv_jac_xy[1][1];
    coefficients[0][q1] = factor * metric_00;
    coefficients[1][q1] = (Number(2) * factor) * metric_01;
    coefficients[2][q1] = factor * metric_11;
    coefficients[3][q1] = JxW_xy;
  }

  // sum factorization in x direction, then in y direction, for the four
  // terms, with the pairs (x table, y table) of the products
  constexpr unsigned int  table_x[4] = {2, 1, 0, 0};
  constexpr unsigned int  table_y[4] = {0, 1, 2, 0};
  VectorizedArray<Number> tmp[4][n_q_points_2d];
  for(unsigned int t = 0; t < 4; ++t)
    for(unsigned int qy = 0; qy < nn; ++qy)
      for(unsigned int i0 = 0; i0 < nn; ++i0)
      {
        VectorizedArray<Number> sum = coefficients[t][qy * nn] * products[table_x[t]][i0 * nn];
        for(unsigned int qx = 1; qx < nn; ++qx)
          sum += coefficients[t][qy * nn + qx] * products[table_x[t]][i0 * nn + qx];
        tmp[t][qy * nn + i0] = sum;
      }
  VectorizedArray<Number> xy_gradient[n_q_points_2d], xy_value[n_q_points_2d];
  for(unsigned int i1 = 0; i1 < nn; ++i1)
    for(unsigned int i0 = 0; i0 < nn; ++i0)
    {
      VectorizedArray<Number> sum[4];
      for(unsigned int t = 0; t < 4; ++t)
      {
        sum[t] = tmp[t][i0] * products[table_y[t]][i1 * nn];
        for(unsigned int qy = 1; qy < nn; ++qy)
          sum[t] += tmp[t][qy * nn + i0] * products[table_y[t]][i1 * nn + qy];
      }
      xy_gradient[i1 * nn + i0] = sum[0] + sum[1] + sum[2];
      xy_value[i1 * nn + i0]    = sum[3];
    }

  if constexpr(dim == 2)
  {
    for(unsigned int i = 0; i < n_q_points_2d; ++i)
      diagonal[i] = xy_gradient[i] + factor_mass * xy_value[i];
  }
  else
  {
    for(unsigned int i2 = 0; i2 < nn; ++i2)
    {
      Number z_value = 0, z_derivative = 0;
      for(unsigned int qz = 0; qz < nn; ++qz)
      {
        z_value += mapping_info.quad_weights_z[qz] * products[0][i2 * nn + qz];
        z_derivative += mapping_info.quad_weights_z[qz] * products[2][i2 * nn + qz];
      }
      VectorizedArray<Number> factor_value =
        (factor_mass * z_value) * h_z + (factor_lapl * z_derivative) * h_z_inverse;
      if(z_face_terms != nullptr)
        factor_value += z_face_terms[i2];
      for(unsigned int i = 0; i < n_q_points_2d; ++i)
        diagonal[i2 * n_q_points_2d + i] = z_value * xy_gradient[i] + factor_value * xy_value[i];
    }
  }
}


// retrieve the coarsening sequence for p multigrid, defined in a central
// place: pick the next coarser degree as half the previous degree, but for
// small degrees we do smaller steps to avoid too aggressive coarsening (i.e.,
// we do steps like 2-1, 3-2-1, 4-2-1, 5-3-2-1, 6-4-2-1, 7-5-3-2-1,
// 8-5-3-2-1 etc)
constexpr unsigned int
get_coarser_fe_degree(const unsigned int degree)
{
  if(degree < 4)
    return degree - 1;
  else if(degree < 8)
    return degree - 2;
  else
    return degree - 3;
}



// Order of the cell batches of a MatrixFree object in the loop of
// LaplaceOperatorFE::vmult_scratch(): the fraction 'split' of the interior
// cell batches, then the cell batches with an unknown not owned by this
// process (after resolving the identity constraints of periodic faces), then
// the remaining interior cell batches, or the original order for 'split' < 0
template<int dim, typename Number, typename OtherNumber>
std::vector<unsigned int>
compute_fe_loop_order(const MatrixFree<dim, Number> &        matrix_free,
                      const AffineConstraints<OtherNumber> & constraints,
                      const double                           split)
{
  const unsigned int        n_batches = matrix_free.n_cell_batches();
  std::vector<unsigned int> order(n_batches);
  std::iota(order.begin(), order.end(), 0u);
  if(split < 0)
    return order;
  const IndexSet &                     owned = matrix_free.get_dof_handler().locally_owned_dofs();
  std::vector<types::global_dof_index> dof_indices(
    matrix_free.get_dof_handler().get_fe().dofs_per_cell);
  std::vector<unsigned int> interior, boundary;
  for(unsigned int cell = 0; cell < n_batches; ++cell)
  {
    bool touches_ghosts = false;
    for(unsigned int v = 0;
        v < matrix_free.n_active_entries_per_cell_batch(cell) && !touches_ghosts;
        ++v)
    {
      matrix_free.get_cell_iterator(cell, v)->get_dof_indices(dof_indices);
      for(types::global_dof_index index : dof_indices)
      {
        const auto line = constraints.get_constraint_entries(index);
        if(line != nullptr && line->size() == 1 && (*line)[0].second == OtherNumber(1.0))
          index = (*line)[0].first;
        if(!owned.is_element(index) && !constraints.is_constrained(index))
          touches_ghosts = true;
      }
    }
    (touches_ghosts ? boundary : interior).push_back(cell);
  }
  if(boundary.empty())
    return order;
  const unsigned int n_first =
    static_cast<unsigned int>(std::min(1., split) * interior.size() + 0.5);
  order.clear();
  order.insert(order.end(), interior.begin(), interior.begin() + n_first);
  order.insert(order.end(), boundary.begin(), boundary.end());
  order.insert(order.end(), interior.begin() + n_first, interior.end());
  return order;
}



// Renumber the locally owned DoFs of a continuous FE_Q space for the loop of
// LaplaceOperatorFE::vmult_scratch() (as RTOperator's numbering in
// Helper::renumber_dofs_in_cell_loop_order()): the entities (vertices,
// lines, quads, hexes) in the order of their last write in the loop of the
// cell batches of compute_fe_loop_order(), such that the ranges of the
// vector passed to 'after_loop' are contiguous, then the entities read by
// cells of other processes, whose result is only complete after the
// exchange, and finally the constrained DoFs. The DoFs of each entity stay
// contiguous and in their order. The cell batches are those of a MatrixFree
// object with the settings of LaplaceOperatorFE::reinit() and the given
// 'cell_vectorization_category'. The constraints need to be recomputed
// after the renumbering.
template<int dim, typename OtherNumber>
void
renumber_fe_dofs_in_loop_order(DoFHandler<dim> &                      dof_handler,
                               const AffineConstraints<OtherNumber> & constraints,
                               const std::vector<unsigned int> &      cell_vectorization_category,
                               const double                           split)
{
  const FiniteElement<dim> &                            fe     = dof_handler.get_fe();
  const unsigned int                                    degree = fe.degree;
  typename MatrixFree<dim, OtherNumber>::AdditionalData mf_data;
  mf_data.cell_vectorization_category          = cell_vectorization_category;
  mf_data.cell_vectorization_categories_strict = false;
  mf_data.overlap_communication_computation    = false;
  mf_data.initialize_mapping                   = false;
  MatrixFree<dim, OtherNumber> matrix_free;
  matrix_free.reinit(MappingQ1<dim>(), dof_handler, constraints, QGauss<1>(degree + 1), mf_data);
  const std::vector<unsigned int> order = compute_fe_loop_order(matrix_free, constraints, split);

  const IndexSet &   owned   = dof_handler.locally_owned_dofs();
  const unsigned int n_owned = owned.n_elements();
  const auto         resolve = [&](types::global_dof_index index) {
    const auto line = constraints.get_constraint_entries(index);
    if(line != nullptr && line->size() == 1 && (*line)[0].second == OtherNumber(1.0))
      index = (*line)[0].first;
    return index;
  };

  // entities of a cell: start within the cell's DoFs (hierarchic order) and
  // number of DoFs
  std::vector<std::pair<unsigned int, unsigned int>> entities;
  for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
    entities.emplace_back(v * fe.n_dofs_per_vertex(), fe.n_dofs_per_vertex());
  if(degree > 1)
  {
    for(unsigned int l = 0; l < GeometryInfo<dim>::lines_per_cell; ++l)
      entities.emplace_back(fe.get_first_line_index() + l * fe.n_dofs_per_line(),
                            fe.n_dofs_per_line());
    if constexpr(dim == 3)
    {
      for(unsigned int q = 0; q < GeometryInfo<dim>::quads_per_cell; ++q)
        entities.emplace_back(fe.get_first_quad_index(q), fe.n_dofs_per_quad(q));
      entities.emplace_back(fe.get_first_hex_index(), fe.n_dofs_per_hex());
    }
    else
      entities.emplace_back(fe.get_first_quad_index(0), fe.n_dofs_per_quad(0));
  }

  // first DoF of an entity, i.e., the smallest index of its DoFs, which
  // form a contiguous range (in the order of the cell, the DoFs on some
  // faces are permuted, see LaplaceOperatorFE::reinit())
  std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
  const auto entity_first = [&](const unsigned int start, const unsigned int n) {
    types::global_dof_index first = resolve(dof_indices[start]);
    types::global_dof_index last  = first;
    for(unsigned int i = 1; i < n; ++i)
    {
      const types::global_dof_index index = resolve(dof_indices[start + i]);
      first                               = std::min(first, index);
      last                                = std::max(last, index);
    }
    AssertThrow(last - first + 1 == n, ExcMessage("The DoFs of each entity need to be contiguous"));
    return first;
  };

  // last position in the loop writing each locally owned entity, given by
  // the local index of its first DoF
  std::vector<unsigned int> last_writer(n_owned, numbers::invalid_unsigned_int);
  for(unsigned int position = 0; position < order.size(); ++position)
    for(unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(order[position]); ++v)
    {
      matrix_free.get_cell_iterator(order[position], v)->get_dof_indices(dof_indices);
      for(const auto & [start, n] : entities)
      {
        const types::global_dof_index first = entity_first(start, n);
        if(owned.is_element(first) && !constraints.is_constrained(first))
          last_writer[owned.index_within_set(first)] = position;
      }
    }

  // entities read by cells of other processes
  std::vector<unsigned char> is_imported(n_owned, 0);
  for(const auto & cell : dof_handler.active_cell_iterators())
    if(cell->is_ghost())
    {
      cell->get_dof_indices(dof_indices);
      for(const auto & [start, n] : entities)
      {
        const types::global_dof_index first = entity_first(start, n);
        if(owned.is_element(first))
          is_imported[owned.index_within_set(first)] = 1;
      }
    }

  std::vector<types::global_dof_index> new_numbers(n_owned, numbers::invalid_dof_index);
  std::vector<std::pair<unsigned int, unsigned int>> imported_entities;
  types::global_dof_index                            counter = 0;
  for(unsigned int position = 0; position < order.size(); ++position)
    for(unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(order[position]); ++v)
    {
      matrix_free.get_cell_iterator(order[position], v)->get_dof_indices(dof_indices);
      for(const auto & [start, n] : entities)
      {
        const types::global_dof_index first = entity_first(start, n);
        if(!owned.is_element(first) || constraints.is_constrained(first))
          continue;
        const unsigned int local = owned.index_within_set(first);
        if(last_writer[local] != position || new_numbers[local] != numbers::invalid_dof_index)
          continue;
        if(is_imported[local])
        {
          imported_entities.emplace_back(local, n);
          for(unsigned int i = 0; i < n; ++i)
            new_numbers[local + i] = numbers::invalid_dof_index - 1;
        }
        else
          for(unsigned int i = 0; i < n; ++i)
            new_numbers[local + i] = counter++;
      }
    }
  for(const auto & [local, n] : imported_entities)
    for(unsigned int i = 0; i < n; ++i)
      new_numbers[local + i] = counter++;
  for(types::global_dof_index & number : new_numbers)
    if(number == numbers::invalid_dof_index)
      number = counter++;
  AssertThrow(counter == n_owned, ExcInternalError());

  std::vector<types::global_dof_index> new_global(n_owned);
  for(unsigned int i = 0; i < n_owned; ++i)
    new_global[i] = owned.nth_index_in_set(new_numbers[i]);
  dof_handler.renumber_dofs(new_global);
}



template<int dim, typename Number = double>
class LaplaceOperatorFE : public EnableObserverPointer
{
public:
  static constexpr unsigned int n_lanes = VectorizedArray<Number>::size();
  using VectorType                      = LinearAlgebra::distributed::Vector<Number>;

  LaplaceOperatorFE() = default;

  template<typename OtherNumber>
  void
  reinit(const Mapping<dim> &                   mapping,
         const DoFHandler<dim> &                dof_handler,
         const AffineConstraints<OtherNumber> & constraints,
         const std::vector<unsigned int> &      cell_vectorization_category,
         const Quadrature<1> &                  quadrature,
         const MPI_Comm                         communicator_shared = MPI_COMM_SELF)
  {
    (void)communicator_shared; // see set_shared_memory()
    this->dof_handler                                   = &dof_handler;
    const FiniteElement<dim> &                       fe = dof_handler.get_fe();
    typename MatrixFree<dim, Number>::AdditionalData mf_data;
    mf_data.cell_vectorization_category          = cell_vectorization_category;
    mf_data.cell_vectorization_categories_strict = false;
    mf_data.overlap_communication_computation    = false;
    mf_data.initialize_mapping                   = false;
    matrix_free.reinit(mapping, dof_handler, constraints, quadrature, mf_data);

    const unsigned int fe_degree = fe.degree;
    compressed_dof_indices.clear();
    const unsigned int stored_dofs_per_direction = fe_degree == 1 ? 2 : 3;
    const unsigned int n_stored_dofs             = Utilities::pow(stored_dofs_per_direction, dim);
    compressed_dof_indices.resize(n_stored_dofs * n_lanes * matrix_free.n_cell_batches(),
                                  numbers::invalid_unsigned_int);
    all_indices_unconstrained.clear();
    all_indices_unconstrained.resize(n_stored_dofs * matrix_free.n_cell_batches(), 1);

    const unsigned int                  n_components = fe.n_components();
    const Utilities::MPI::Partitioner & part = *matrix_free.get_dof_info().vector_partitioner;

    const std::vector<unsigned int> renumber_lex =
      FETools::hierarchic_to_lexicographic_numbering<dim>(2);

    // Write the compressed indices of one cell, given by the dof indices in
    // hierarchic order, to the positions 'out[i * n_lanes]' for the
    // 'n_stored_dofs' entities of the cell
    const auto compress_indices = [&](const std::vector<types::global_dof_index> & dof_indices,
                                      unsigned int *                               out) {
      // Low degree -> just pick the stored dofs
      if(fe_degree == 1)
      {
        for(unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i)
        {
          for(unsigned int c = 0; c < n_components; ++c)
            AssertThrow(dof_indices[i * n_components + c] == dof_indices[i * n_components] + c,
                        ExcMessage("Expected contiguous numbering"));
          out[i * n_lanes] = constraints.is_constrained(dof_indices[i * n_components]) ?
                               numbers::invalid_unsigned_int :
                               part.global_to_local(dof_indices[i * n_components]);
        }
        return;
      }

      // Else store the compressed form with 3^dim dofs per cell
      unsigned int cc = 0, cf = 0;
      for(unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell;
          ++i, ++cc, cf += n_components)
      {
        if(!constraints.is_constrained(dof_indices[cf]))
          out[renumber_lex[cc] * n_lanes] = part.global_to_local(dof_indices[cf]);
        for(unsigned int c = 0; c < n_components; ++c)
          AssertThrow(dof_indices[cf + c] == dof_indices[cf] + c,
                      ExcMessage("Expected contiguous numbering"));
      }

      for(unsigned int line = 0; line < GeometryInfo<dim>::lines_per_cell; ++line)
      {
        const unsigned int size = fe_degree - 1;
        if(!constraints.is_constrained(dof_indices[cf]))
        {
          for(unsigned int i = 0; i < size; ++i)
            for(unsigned int c = 0; c < n_components; ++c)
              AssertThrow(dof_indices[cf + c * size + i] == dof_indices[cf] + i * n_components + c,
                          ExcMessage("Expected contiguous numbering"));
          out[renumber_lex[cc] * n_lanes] = part.global_to_local(dof_indices[cf]);
        }
        ++cc;
        cf += size * n_components;
      }
      for(unsigned int quad = 0; quad < GeometryInfo<dim>::quads_per_cell; ++quad)
      {
        const unsigned int size = (fe_degree - 1) * (fe_degree - 1);
        if(!constraints.is_constrained(dof_indices[cf]))
        {
          // switch order x-z for y faces in 3D to lexicographic layout
          if(dim == 3 && (quad == 2 || quad == 3))
            for(unsigned int i1 = 0, i = 0; i1 < fe_degree - 1; ++i1)
              for(unsigned int i0 = 0; i0 < fe_degree - 1; ++i0, ++i)
                for(unsigned int c = 0; c < n_components; ++c)
                {
                  AssertThrow(dof_indices[cf + c * size + i0 * (fe_degree - 1) + i1] ==
                                dof_indices[cf] + i * n_components + c,
                              ExcMessage("Expected contiguous numbering"));
                }
          else
            for(unsigned int i = 0; i < size; ++i)
              for(unsigned int c = 0; c < n_components; ++c)
                AssertThrow(dof_indices[cf + c * size + i] ==
                              dof_indices[cf] + i * n_components + c,
                            ExcMessage("Expected contiguous numbering"));
          out[renumber_lex[cc] * n_lanes] = part.global_to_local(dof_indices[cf]);
        }
        ++cc;
        cf += size * n_components;
      }
      for(unsigned int hex = 0; hex < GeometryInfo<dim>::hexes_per_cell; ++hex)
      {
        const unsigned int size = (fe_degree - 1) * (fe_degree - 1) * (fe_degree - 1);
        if(!constraints.is_constrained(dof_indices[cf]))
        {
          for(unsigned int i = 0; i < size; ++i)
            for(unsigned int c = 0; c < n_components; ++c)
              AssertThrow(dof_indices[cf + c * size + i] == dof_indices[cf] + i * n_components + c,
                          ExcMessage("Expected contiguous numbering"));
          out[renumber_lex[cc] * n_lanes] = part.global_to_local(dof_indices[cf]);
        }
        ++cc;
        cf += size * n_components;
      }
      AssertThrow(cc == Utilities::pow(3, dim),
                  ExcMessage("Expected 3^dim dofs, got " + std::to_string(cc)));
      AssertThrow(cf == dof_indices.size(),
                  ExcMessage("Expected (fe_degree+1)^dim dofs, got " + std::to_string(cf)));
    };

    // Replace the indices of DoFs constrained to a single other DoF with
    // weight 1 (periodicity) by the index of that DoF
    const auto resolve_identity_constraints =
      [&](std::vector<types::global_dof_index> & dof_indices) {
        for(types::global_dof_index & i : dof_indices)
        {
          const auto line = constraints.get_constraint_entries(i);
          if(line != nullptr && line->size() == 1 && (*line)[0].second == Number(1.0))
            i = (*line)[0].first;
        }
      };

    // Hanging nodes: MatrixFree has detected the cells with hanging nodes,
    // whose constraint kind we store per cell batch. On those cells, the
    // indices of the constrained DoFs on the refined side get replaced by the
    // ones of the coarse side via the same algorithm as in MatrixFree, such
    // that the cell kernel can apply the interpolation of
    // FEEvaluationHangingNodesFactory to the values read from those indices.
    const auto & hn_masks = matrix_free.get_dof_info().hanging_node_constraint_masks;
    hanging_node_masks.clear();
    batch_has_hanging_nodes.clear();
    positions_replaced_by_hanging_nodes.clear();
    if(std::any_of(hn_masks.begin(), hn_masks.end(), [](const auto mask) {
         return mask != internal::MatrixFreeFunctions::unconstrained_compressed_constraint_kind;
       }))
    {
      AssertThrow(n_components == 1,
                  ExcNotImplemented("Hanging nodes only implemented for scalar elements"));
      AssertDimension(hn_masks.size(), matrix_free.n_cell_batches() * n_lanes);
      hanging_node_masks.resize(matrix_free.n_cell_batches());
      batch_has_hanging_nodes.resize(matrix_free.n_cell_batches(), 0);
      for(unsigned int c = 0; c < matrix_free.n_cell_batches(); ++c)
        for(unsigned int v = 0; v < n_lanes; ++v)
        {
          hanging_node_masks[c][v] = hn_masks[c * n_lanes + v];
          if(hn_masks[c * n_lanes + v] !=
             internal::MatrixFreeFunctions::unconstrained_compressed_constraint_kind)
            batch_has_hanging_nodes[c] = 1;
        }
    }
    std::unique_ptr<internal::MatrixFreeFunctions::HangingNodes<dim>> hanging_nodes;
    if(!hanging_node_masks.empty())
      hanging_nodes = std::make_unique<internal::MatrixFreeFunctions::HangingNodes<dim>>(
        dof_handler.get_triangulation());
    const std::vector<std::vector<unsigned int>> lexicographic_numbering{
      {matrix_free.get_shape_info().lexicographic_numbering}};
    const std::vector<std::vector<bool>> hn_component_mask{{true}};

    std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
    std::vector<types::global_dof_index> dof_indices_lex(fe.dofs_per_cell);
    std::vector<types::global_dof_index> own_dof_indices(fe.dofs_per_cell);
    std::vector<unsigned int>            own_indices(n_stored_dofs * n_lanes);
    for(unsigned int c = 0; c < matrix_free.n_cell_batches(); ++c)
    {
      for(unsigned int l = 0; l < matrix_free.n_active_entries_per_cell_batch(c); ++l)
      {
        const auto cell = matrix_free.get_cell_iterator(c, l);
        cell->get_dof_indices(dof_indices);
        unsigned int * out = compressed_dof_indices.data() + n_stored_dofs * n_lanes * c + l;

        if(!hanging_node_masks.empty() &&
           hanging_node_masks[c][l] !=
             internal::MatrixFreeFunctions::unconstrained_compressed_constraint_kind)
        {
          // indices without the hanging-node replacement, to identify the
          // entities that are not owned by the cell
          own_dof_indices = dof_indices;
          resolve_identity_constraints(own_dof_indices);
          std::fill(own_indices.begin(), own_indices.end(), numbers::invalid_unsigned_int);
          compress_indices(own_dof_indices, own_indices.data());

          const internal::MatrixFreeFunctions::ConstraintKinds refinement_configuration =
            hanging_nodes->compute_refinement_configuration(cell);
          AssertThrow(internal::MatrixFreeFunctions::compress(refinement_configuration, dim) ==
                        hanging_node_masks[c][l],
                      ExcInternalError("Hanging-node configuration does not match MatrixFree"));
          const std::vector<unsigned int> & lex = lexicographic_numbering[0];
          for(unsigned int i = 0; i < dof_indices.size(); ++i)
            dof_indices_lex[i] = dof_indices[lex[i]];
          hanging_nodes->update_dof_indices(cell,
                                            {},
                                            lexicographic_numbering,
                                            hn_component_mask,
                                            refinement_configuration,
                                            dof_indices_lex);
          for(unsigned int i = 0; i < dof_indices.size(); ++i)
            dof_indices[lex[i]] = dof_indices_lex[i];

          resolve_identity_constraints(dof_indices);
          compress_indices(dof_indices, out);

          // Record the positions that have been replaced. Note that the
          // own index of a hanging DoF can be valid, namely when the
          // hanging-node constraint is an identity to a single DoF of the
          // coarse side (e.g. the midpoint of a coarse edge for even
          // degrees), which gets resolved like the periodicity constraints.
          // Dropping such a position is still correct for the transfer
          // operators, because that DoF is also visited by the cell on the
          // coarse side that actually owns it.
          for(unsigned int i = 0; i < n_stored_dofs; ++i)
            if(own_indices[i * n_lanes] != out[i * n_lanes])
              positions_replaced_by_hanging_nodes.push_back(n_stored_dofs * n_lanes * c +
                                                            i * n_lanes + l);
        }
        else
        {
          resolve_identity_constraints(dof_indices);
          compress_indices(dof_indices, out);
        }
      }

      for(unsigned int i = 0; i < n_stored_dofs; ++i)
        for(unsigned int v = 0; v < n_lanes; ++v)
          if(compressed_dof_indices[n_stored_dofs * (n_lanes * c) + i * n_lanes + v] ==
             numbers::invalid_unsigned_int)
            all_indices_unconstrained[n_stored_dofs * c + i] = 0;
    }

    cell_level_index.resize(matrix_free.n_cell_batches());
    for(unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
      for(unsigned int lane = 0; lane < n_lanes; ++lane)
        if(lane >= matrix_free.n_active_entries_per_cell_batch(cell))
          cell_level_index[cell][lane] = {
            {numbers::invalid_unsigned_int, numbers::invalid_unsigned_int}};
        else
        {
          const auto iter                 = matrix_free.get_cell_iterator(cell, lane);
          cell_level_index[cell][lane][0] = iter->level();
          cell_level_index[cell][lane][1] = iter->index();
        }

    mapping_info.reinit(mapping, quadrature, dof_handler.get_triangulation(), cell_level_index);

    factor_laplace = 1.0;
    factor_mass    = 0.0;

    // overlap of the communication with the interior cell batches, see
    // set_loop_settings()
    setup_loop_order(overlap_split);
    setup_shared_memory();
    setup_scratch();
  }

  // Vectors of the operator; with the shared-memory exchange of
  // set_shared_memory(), they are allocated in the shared memory of the
  // node, which vmult_scratch() needs for the source vector
  void
  initialize_dof_vector(VectorType & vec) const
  {
    if(comm_sm)
      vec.reinit(matrix_free.get_dof_info().vector_partitioner, *comm_sm);
    else
      matrix_free.initialize_dof_vector(vec);
  }

  void
  vmult(VectorType & dst, const VectorType & src) const
  {
    Timer timer;
    matrix_free.cell_loop(&LaplaceOperatorFE::local_apply, this, dst, src, true);
    for(unsigned int i : matrix_free.get_constrained_dofs(0))
      dst.local_element(i) = src.local_element(i);
    timings[t_vmult_count] += 1;
    timings[t_vmult_total] += timer.wall_time();
  }

  // Print the setup of the data exchange of the FE operator per process:
  // the neighboring processes in the same and in other shared-memory
  // domains (nodes), the size of the data per ghost update (received ghost
  // entries; the compress sends the same amount back) and of the locally
  // owned entries sent to the other processes (and received in the
  // compress), split by same and other nodes, the share of cell batches
  // touching ghost entries, and, with the block-Jacobi preconditioner of
  // setup_block_batches(), the share of block batches run after the cell
  // loop and of the unknowns with point Jacobi. Must be called on all
  // processes.
  void
  print_exchange_statistics(const std::string & label) const
  {
    const MPI_Comm comm = dof_handler->get_mpi_communicator();
    const auto &   part = *matrix_free.get_dof_info().vector_partitioner;

    // ranks of the own shared-memory domain
    std::vector<int> node_ranks;
    {
      MPI_Comm  comm_node;
      const int rank = Utilities::MPI::this_mpi_process(comm);
      int ierr = MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &comm_node);
      AssertThrowMPI(ierr);
      node_ranks.resize(Utilities::MPI::n_mpi_processes(comm_node));
      ierr = MPI_Allgather(&rank, 1, MPI_INT, node_ranks.data(), 1, MPI_INT, comm_node);
      AssertThrowMPI(ierr);
      ierr = MPI_Comm_free(&comm_node);
      AssertThrowMPI(ierr);
      std::sort(node_ranks.begin(), node_ranks.end());
    }
    const auto on_node = [&](const unsigned int rank) {
      return std::binary_search(node_ranks.begin(), node_ranks.end(), static_cast<int>(rank));
    };

    std::set<unsigned int> neighbors;
    std::array<double, 2>  ghost_bytes = {}, import_bytes = {};
    for(const auto & [rank, n] : part.ghost_targets())
    {
      neighbors.insert(rank);
      ghost_bytes[on_node(rank) ? 0 : 1] += n * sizeof(Number);
    }
    for(const auto & [rank, n] : part.import_targets())
    {
      neighbors.insert(rank);
      import_bytes[on_node(rank) ? 0 : 1] += n * sizeof(Number);
    }
    unsigned int n_neighbors_on_node = 0;
    for(const unsigned int rank : neighbors)
      if(on_node(rank))
        ++n_neighbors_on_node;

    // cell batches with ghost entries
    const unsigned int n_stored  = Utilities::pow(dof_handler->get_fe().degree == 1 ? 2 : 3, dim);
    const unsigned int n_owned   = part.locally_owned_size();
    const unsigned int n_batches = matrix_free.n_cell_batches();
    unsigned int       n_batches_with_ghosts = 0;
    for(unsigned int cell = 0; cell < n_batches; ++cell)
    {
      bool has_ghost = false;
      for(unsigned int i = 0; i < n_stored * n_lanes && !has_ghost; ++i)
      {
        const unsigned int index = compressed_dof_indices[cell * n_stored * n_lanes + i];
        has_ghost                = index != numbers::invalid_unsigned_int && index >= n_owned;
      }
      if(has_ghost)
        ++n_batches_with_ghosts;
    }

    if(Utilities::MPI::this_mpi_process(comm) == 0)
      std::cout << "Data exchange of FE Laplace operator <"
                << (std::is_same_v<Number, double> ? "double" : "float") << "> " << label
                << ", degree " << dof_handler->get_fe().degree << ", per process:" << std::endl;
    Helper::print_time(n_neighbors_on_node, "Neighbors on same node", comm);
    Helper::print_time(neighbors.size() - n_neighbors_on_node, "Neighbors on other nodes", comm);
    Helper::print_time(ghost_bytes[0] / 1024., "Ghosts from same node [KiB]", comm);
    Helper::print_time(ghost_bytes[1] / 1024., "Ghosts from other nodes [KiB]", comm);
    Helper::print_time(import_bytes[0] / 1024., "Owned to same node [KiB]", comm);
    Helper::print_time(import_bytes[1] / 1024., "Owned to other nodes [KiB]", comm);
    Helper::print_time(n_batches > 0 ? double(n_batches_with_ghosts) / n_batches : 0.,
                       "Cell batches with ghosts",
                       comm);
    if(Utilities::MPI::max(has_block_batches() ? 1 : 0, comm) == 1)
    {
      const unsigned int n_block_batches = block_batch_order.size();
      const unsigned int n_after_loop =
        has_block_batches() ? block_batch_index[n_batches + 1] - block_batch_index[n_batches] : 0;
      Helper::print_time(n_block_batches > 0 ? double(n_after_loop) / n_block_batches : 0.,
                         "Block batches after loop",
                         comm);
      Helper::print_time(n_owned > 0 ? double(leftover_indices.size()) / n_owned : 0.,
                         "Unknowns with point Jacobi",
                         comm);
    }
    if(Utilities::MPI::this_mpi_process(comm) == 0)
      std::cout << std::endl;
  }

  // Print the collected timings of the phases of vmult_scratch(), which is
  // used by the Chebyshev smoother, and of the plain vmult(). Must be called
  // on all processes of the communicator of the DoFHandler.
  void
  print_timings(const std::string & label) const
  {
    const MPI_Comm comm      = dof_handler->get_mpi_communicator();
    const double   n_scratch = Utilities::MPI::max(timings[t_scratch_count], comm);
    const double   n_vmult   = Utilities::MPI::max(timings[t_vmult_count], comm);
    std::cout << std::defaultfloat << std::setprecision(3);
    if(n_scratch > 0)
    {
      const double count      = std::max(1., timings[t_scratch_count]);
      const double total_time = Utilities::MPI::sum(timings[t_scratch_total] / count, comm) /
                                Utilities::MPI::n_mpi_processes(comm);
      if(Utilities::MPI::this_mpi_process(comm) == 0)
        std::cout << "Collected timings for FE Laplace operator <"
                  << (std::is_same_v<Number, double> ? "double" : "float") << "> " << label
                  << ", degree " << matrix_free.get_shape_info().data[0].fe_degree << ", "
                  << dof_handler->n_dofs() << " DoFs, vmult_scratch in "
                  << static_cast<unsigned long>(n_scratch)
                  << " evaluations [t_total=" << total_time * n_scratch << "s]" << std::endl;
      Helper::print_time(timings[t_before_loop] / count,
                         "before_loop prior to ghosts",
                         comm,
                         total_time);
      Helper::print_time(timings[t_ghost_start] / count,
                         "Ghost send + prepare scratch",
                         comm,
                         total_time);
      Helper::print_time(timings[t_ghost_wait] / count, "Wait for ghosts", comm, total_time);
      const double cells_total =
        timings[t_cell_loop] + timings[t_cells_boundary] + timings[t_cells_interior_2];
      Helper::print_time(timings[t_cell_loop] / count, "Interior cells, part 1", comm, total_time);
      Helper::print_time(timings[t_cells_boundary] / count, "Boundary cells", comm, total_time);
      Helper::print_time(timings[t_cells_interior_2] / count,
                         "Interior cells, part 2",
                         comm,
                         total_time);
      Helper::print_time_per_node(cells_total / count, "All cells, node avg", comm, total_time);
      Helper::print_time(timings[t_compress_send] / count,
                         "Compress start + zero ghosts",
                         comm,
                         total_time);
      Helper::print_time(timings[t_compress_wait] / count, "Compress wait", comm, total_time);
      Helper::print_time(timings[t_finish_ranges] / count,
                         "after_loop on MPI ranges",
                         comm,
                         total_time);
      if(Utilities::MPI::max(timings[t_block_interface], comm) > 0)
        Helper::print_time(timings[t_block_interface] / count,
                           "Interface blocks (block Jacobi)",
                           comm,
                           total_time);
      Helper::print_time(timings[t_scratch_total] / count, "Total vmult_scratch", comm, total_time);
    }
    if(n_vmult > 0)
    {
      if(Utilities::MPI::this_mpi_process(comm) == 0 && n_scratch == 0)
        std::cout << "Collected timings for FE Laplace operator <"
                  << (std::is_same_v<Number, double> ? "double" : "float") << "> " << label
                  << ", degree " << matrix_free.get_shape_info().data[0].fe_degree << ", "
                  << dof_handler->n_dofs() << " DoFs" << std::endl;
      Helper::print_time(timings[t_vmult_total] / std::max(1., timings[t_vmult_count]),
                         "vmult (" + std::to_string(static_cast<unsigned long>(n_vmult)) + "x)",
                         comm);
    }
    if((n_scratch > 0 || n_vmult > 0) && Utilities::MPI::this_mpi_process(comm) == 0)
      std::cout << std::endl;
  }

  void
  vmult(
    VectorType &                                                        dst,
    const VectorType &                                                  src,
    const std::function<void(const unsigned int, const unsigned int)> & operation_before_loop,
    const std::function<void(const unsigned int, const unsigned int)> & operation_after_loop) const
  {
    matrix_free.cell_loop(
      &LaplaceOperatorFE::local_apply, this, dst, src, operation_before_loop, operation_after_loop);
  }

  // Variant of the matrix-vector product that accumulates the result into
  // a small scratch array instead of a global vector, see
  // ExaDG::ScratchMatrixVectorProduct for the format of the ranges passed
  // to 'after_loop'. Like for MatrixFree::cell_loop(), the constrained
  // entries, which are the only ones not written by the cells, are set to
  // the entries of src.
  template<typename AfterLoop>
  void
  vmult_scratch(const VectorType & src, const AfterLoop & after_loop) const
  {
    vmult_scratch(src, nullptr, after_loop);
  }

  // Variant of vmult_scratch() that also runs 'before_loop(begin, end)' on
  // the locally owned entries of src before their first read, i.e., before
  // the ghost exchange for the entries sent to other processes, before the
  // first cell batch reading them, and before the result of the entries is
  // passed to 'after_loop' or set from src for the constrained entries. Each
  // entry passes through 'before_loop' exactly once. On processes with
  // fewer locally owned entries than set by set_min_size_for_pre_list(),
  // 'before_loop' runs on all entries before the loop.
  template<typename BeforeLoop, typename AfterLoop>
  void
  vmult_scratch(const VectorType & src,
                const BeforeLoop & before_loop,
                const AfterLoop &  after_loop) const
  {
    do_vmult_scratch(src, before_loop, after_loop, nullptr);
  }

  // Implementation of vmult_scratch(), with 'after_batch(position,
  // scratch_array)' run after each step of the loop if given. The cell
  // batches run in the order of setup_loop_order(): the ghost update of src
  // is finished after the first part of the interior cell batches, and the
  // compress is started after the cell batches touching ghost entries, so
  // that both overlap with the interior cell batches.
  template<typename BeforeLoop, typename AfterLoop, typename AfterBatch>
  void
  do_vmult_scratch(const VectorType & src,
                   const BeforeLoop & before_loop,
                   const AfterLoop &  after_loop,
                   const AfterBatch & after_batch) const
  {
    Timer      total_timer, timer;
    const auto record = [&](const unsigned int phase) {
      timings[phase] += timer.wall_time();
      timer.restart();
    };
    constexpr bool              has_before_loop = !std::is_same_v<BeforeLoop, std::nullptr_t>;
    const bool                  src_has_ghosts  = src.has_ghost_elements();
    const unsigned int          n_cell_batches  = matrix_free.n_cell_batches();
    const bool                  use_pre_list    = !cell_loop_pre_list_index.empty();
    [[maybe_unused]] const auto run_pre_list    = [&](const unsigned int index) {
      if constexpr(has_before_loop)
        for(unsigned int range = cell_loop_pre_list_index[index];
            range < cell_loop_pre_list_index[index + 1];
            ++range)
          before_loop(cell_loop_pre_list[range].first, cell_loop_pre_list[range].second);
    };
    if constexpr(has_before_loop)
    {
      // the ghost values of src on other processes would not see the
      // changes by 'before_loop'
      AssertThrow(!src_has_ghosts,
                  ExcMessage("The source vector must not have ghost values with before_loop"));
      if(use_pre_list)
        run_pre_list(n_cell_batches);
      else
        before_loop(0, src.locally_owned_size());
    }
    record(t_before_loop);

    // with the shared-memory exchange, the ghost update is always run, as
    // it also orders the compress steps of successive products, see
    // ScratchMatrixVectorProduct::set_shared_memory_exchange()
    const bool update_ghosts       = sm_exchange || !src_has_ghosts;
    const auto ghost_update_finish = [&]() {
      if(sm_exchange)
        sm_ghost_update_finish(src);
      else if(!src_has_ghosts)
        src.update_ghost_values_finish();
    };
    if(sm_exchange)
      sm_ghost_update_start(src);
    else if(!src_has_ghosts)
      src.update_ghost_values_start();
    scratch.prepare_loop();
    record(t_ghost_start);
    if(n_before_ghosts == 0 && update_ghosts)
      ghost_update_finish();
    record(t_ghost_wait);

    const auto copy_constrained =
      [&](const unsigned int begin, const unsigned int end, Number * values) {
        std::copy(src.begin() + begin, src.begin() + end, values);
      };
    const MPI_Comm     comm   = dof_handler->get_mpi_communicator();
    const unsigned int degree = matrix_free.get_shape_info().data[0].fe_degree;
    scratch.cell_loop(
      [&](const unsigned int position, Number * scratch_array) {
        if constexpr(has_before_loop)
          if(use_pre_list)
            run_pre_list(position);
        const unsigned int                                       cell = loop_order[position];
        typename ExaDG::ScratchMatrixVectorProduct<Number>::View dst{scratch_array};
        if(degree == 1)
          apply_cell_operation<1>(cell, src, scratch_dof_indices, dst);
        else if(degree == 2)
          apply_cell_operation<2>(cell, src, scratch_dof_indices, dst);
        else if(degree == 3)
          apply_cell_operation<3>(cell, src, scratch_dof_indices, dst);
        else if(degree == 4)
          apply_cell_operation<4>(cell, src, scratch_dof_indices, dst);
#ifndef DEBUG
        else if(degree == 5)
          apply_cell_operation<5>(cell, src, scratch_dof_indices, dst);
        else if(degree == 6)
          apply_cell_operation<6>(cell, src, scratch_dof_indices, dst);
        else if(degree == 7)
          apply_cell_operation<7>(cell, src, scratch_dof_indices, dst);
        else if(degree == 8)
          apply_cell_operation<8>(cell, src, scratch_dof_indices, dst);
#endif
        else
          AssertThrow(false, ExcMessage("Degree " + std::to_string(degree) + " not instantiated"));
      },
      after_loop,
      copy_constrained,
      [&](const unsigned int position, const Number * scratch_array) {
        if constexpr(!std::is_same_v<AfterBatch, std::nullptr_t>)
          after_batch(position, scratch_array);
        if(position + 1 == n_before_ghosts)
        {
          record(t_cell_loop);
          ghost_update_finish();
          record(t_ghost_wait);
        }
        if(position + 1 == n_before_compress && n_before_compress < n_cell_batches)
        {
          record(t_cells_boundary);
          scratch.start_compress();
          record(t_compress_send);
        }
        // let MPI progress with the messages in flight during the interior
        // cell batches
        if(mpi_progress_interval > 0 && position % mpi_progress_interval == 0 &&
           (position < n_before_ghosts || position >= n_before_compress))
        {
          int flag;
          MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &flag, MPI_STATUS_IGNORE);
        }
      });
    if(n_before_compress < n_cell_batches)
      record(t_cells_interior_2);
    else
      record(n_before_ghosts > 0 ? t_cells_boundary : t_cell_loop);

    scratch.finish_loop(
      after_loop,
      [&]() {
        if(!src_has_ghosts)
          src.zero_out_ghost_values();
      },
      copy_constrained,
      &timings[t_compress_send]);

    timings[t_scratch_count] += 1;
    timings[t_scratch_total] += total_timer.wall_time();
  }

  // Exchange of vmult_scratch() within the shared-memory domain (node), to
  // be set before reinit(): if enabled, the ghost update reads the entries
  // owned by the processes of the same node directly from their memory, and
  // the owners add the contributions to their entries in the compress
  // directly from the memory of the writing processes, using the zero-byte
  // signals of deal.II's VectorDataExchange::Full. Only the processes on
  // other nodes exchange messages. The vectors of initialize_dof_vector()
  // are then allocated in shared memory, and the source vectors of
  // vmult_scratch() must be such vectors. With 'ranks_per_node' > 0, the
  // node is split into groups of that many consecutive ranks, to emulate
  // several nodes on one machine.
  void
  set_shared_memory(const bool use_shared_memory, const unsigned int ranks_per_node = 0)
  {
    shared_memory           = use_shared_memory;
    emulated_ranks_per_node = ranks_per_node;
  }

  // Settings of the loop of vmult_scratch(), to be set before reinit():
  // the fraction 'split' of the interior cell batches run before the cell
  // batches touching ghost entries (negative for the original order without
  // overlap of the communication), and the interval of cell batches between
  // calls to MPI_Iprobe in the interior cell batches (0 to disable)
  void
  set_loop_settings(const double split, const unsigned int progress_interval)
  {
    overlap_split         = split;
    mpi_progress_interval = progress_interval;
  }

  // For the block-Jacobi preconditioner of setup_block_batches(), to be set
  // before that function: the blocks of the cell loop compute the residual
  // from the result of the operator still in the scratch array instead of
  // reading the residual stored by 'after_loop'. This avoids the residual
  // vector but extends the lifetime of the entities in the scratch array
  // and was slower in tests.
  void
  set_block_from_scratch(const bool from_scratch)
  {
    block_from_scratch = from_scratch;
  }

  // Order of the cell batches in vmult_scratch(), see do_vmult_scratch():
  // first the fraction 'split' of the interior cell batches, which do not
  // touch ghost entries, then the cell batches touching ghost entries, then
  // the remaining interior cell batches, each part in the original order.
  // With 'split' < 0, the original order without overlap. Must be called
  // before setup_scratch().
  void
  setup_loop_order(const double split)
  {
    const unsigned int n_batches = matrix_free.n_cell_batches();
    loop_order.resize(n_batches);
    std::iota(loop_order.begin(), loop_order.end(), 0u);
    n_before_ghosts   = 0;
    n_before_compress = n_batches;
    if(split < 0)
      return;
    const unsigned int n_stored = Utilities::pow(dof_handler->get_fe().degree == 1 ? 2 : 3, dim);
    const unsigned int n_owned =
      matrix_free.get_dof_info().vector_partitioner->locally_owned_size();
    std::vector<unsigned int> interior, boundary;
    for(unsigned int cell = 0; cell < n_batches; ++cell)
    {
      bool touches_ghosts = false;
      for(unsigned int i = 0; i < n_stored * n_lanes && !touches_ghosts; ++i)
      {
        const unsigned int index = compressed_dof_indices[cell * n_stored * n_lanes + i];
        touches_ghosts           = index != numbers::invalid_unsigned_int && index >= n_owned;
      }
      (touches_ghosts ? boundary : interior).push_back(cell);
    }
    const unsigned int n_first =
      static_cast<unsigned int>(std::min(1., split) * interior.size() + 0.5);
    loop_order.clear();
    loop_order.insert(loop_order.end(), interior.begin(), interior.begin() + n_first);
    loop_order.insert(loop_order.end(), boundary.begin(), boundary.end());
    loop_order.insert(loop_order.end(), interior.begin() + n_first, interior.end());
    n_before_ghosts   = n_first;
    n_before_compress = n_first + boundary.size();
    // without cell batches touching ghost entries, there is nothing to
    // overlap with
    if(boundary.empty())
    {
      n_before_ghosts   = 0;
      n_before_compress = n_batches;
    }
  }

  // Set up the communicators and the exchange object of the shared-memory
  // exchange, see set_shared_memory(). The exchange runs on duplicates of
  // the communicators, so its messages cannot match others.
  void
  setup_shared_memory()
  {
    sm_exchange.reset();
    comm_sm.reset();
    if(!shared_memory)
    {
      scratch.set_shared_memory_exchange(nullptr, MPI_COMM_SELF);
      return;
    }
    const auto make_comm = [](const MPI_Comm comm) {
      return std::shared_ptr<MPI_Comm>(new MPI_Comm(comm), [](MPI_Comm * c) {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if(!finalized)
          MPI_Comm_free(c);
        delete c;
      });
    };
    const MPI_Comm     comm = dof_handler->get_mpi_communicator();
    const unsigned int rank = Utilities::MPI::this_mpi_process(comm);
    MPI_Comm           node;
    int ierr = MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    AssertThrowMPI(ierr);
    if(emulated_ranks_per_node > 0)
    {
      MPI_Comm group;
      ierr = MPI_Comm_split(node, rank / emulated_ranks_per_node, rank, &group);
      AssertThrowMPI(ierr);
      ierr = MPI_Comm_free(&node);
      AssertThrowMPI(ierr);
      node = group;
    }
    comm_sm = make_comm(node);
    MPI_Comm dup;
    ierr = MPI_Comm_dup(comm, &dup);
    AssertThrowMPI(ierr);
    comm_exchange = make_comm(dup);
    ierr          = MPI_Comm_dup(node, &dup);
    AssertThrowMPI(ierr);
    comm_exchange_sm = make_comm(dup);

    const auto & part = *matrix_free.get_dof_info().vector_partitioner;
    const auto   part_exchange =
      std::make_shared<Utilities::MPI::Partitioner>(part.locally_owned_range(),
                                                    part.ghost_indices(),
                                                    *comm_exchange);
    sm_exchange =
      std::make_shared<internal::MatrixFreeFunctions::VectorDataExchange::Full>(part_exchange,
                                                                                *comm_exchange_sm);
    ghost_temporary.resize_fast(sm_exchange->n_import_indices());
    n_sm_processes = Utilities::MPI::n_mpi_processes(*comm_sm);
    scratch.set_shared_memory_exchange(sm_exchange, *comm_sm);
  }

  void
  sm_ghost_update_start(const VectorType & src) const
  {
    AssertThrow(src.shared_vector_data().size() == n_sm_processes,
                ExcMessage("The source vector must be allocated in shared memory, see "
                           "initialize_dof_vector()"));
    const unsigned int n_owned = src.locally_owned_size();
    Number *           ghosts  = const_cast<Number *>(src.begin()) + n_owned;
    sm_exchange->export_to_ghosted_array_start(
      ghost_channel,
      ArrayView<const Number>(src.begin(), n_owned),
      src.shared_vector_data(),
      ArrayView<Number>(ghosts, sm_exchange->n_ghost_indices()),
      ArrayView<Number>(ghost_temporary.data(), ghost_temporary.size()),
      ghost_requests);
  }

  void
  sm_ghost_update_finish(const VectorType & src) const
  {
    const unsigned int n_owned = src.locally_owned_size();
    Number *           ghosts  = const_cast<Number *>(src.begin()) + n_owned;
    sm_exchange->export_to_ghosted_array_finish(ArrayView<const Number>(src.begin(), n_owned),
                                                src.shared_vector_data(),
                                                ArrayView<Number>(ghosts,
                                                                  sm_exchange->n_ghost_indices()),
                                                ghost_requests);
  }

  // Set the minimal number of locally owned entries for which the variant
  // of vmult_scratch() with 'before_loop' runs it within the cell loop,
  // close to the first read of the entries. Below, the overhead of the
  // lists would not pay off, as the vectors stay in cache anyway.
  void
  set_min_size_for_pre_list(const unsigned int min_size)
  {
    min_size_for_pre_list = min_size;
    if(dof_handler != nullptr)
      setup_pre_list();
  }

  types::global_dof_index
  m() const
  {
    return matrix_free.get_dof_handler().n_dofs();
  }

  Number
  el(const types::global_dof_index, const types::global_dof_index) const
  {
    AssertThrow(false, ExcNotImplemented());
    return 0;
  }

  const MatrixFree<dim, Number> &
  get_matrix_free() const
  {
    return matrix_free;
  }

  void
  compute_inverse_diagonal(VectorType & vector) const
  {
    matrix_free.initialize_dof_vector(vector);
    unsigned int dummy = 0;
    matrix_free.cell_loop(&LaplaceOperatorFE::local_compute_diagonal, this, vector, dummy);
    for(unsigned int i : matrix_free.get_constrained_dofs(0))
      vector.local_element(i) = 1;
    for(Number & a : vector)
    {
      AssertThrow(a > 0,
                  ExcInternalError(
                    "Error FE degree " + std::to_string(dof_handler->get_fe().degree) + " size " +
                    std::to_string(dof_handler->n_dofs()) + " a = " + std::to_string(a)));
      a = Number(1.) / a;
    }
  }

  // --- Block-Jacobi preconditioner with fast diagonalization ---
  //
  // The unknowns are split into non-overlapping blocks, one per cell: a
  // cell owns the unknowns at the nodes (i_0, i_1, i_2) of its tensor-
  // product numbering with i_d in [s_d, degree], where s_d = 1 if the cell
  // has a neighbor (or a periodic neighbor) on its lower face in direction
  // d, whose block contains the nodes with i_d = 0, and s_d = 0 at a
  // boundary. The unknowns of periodic nodes are those of the master DoF
  // the constraints map them to. The block of the operator is approximated
  // by the block of a Cartesian surrogate as for LaplaceOperatorDG: per
  // direction, the 1D mass and stiffness matrices of the continuous
  // element on the cell with the extent h_d, restricted to the owned
  // nodes, plus the contribution of the neighbor on the upper face (if
  // any) to the shared node, using the same extent h_d, and inverted by
  // the fast diagonalization method. On a Cartesian mesh with uniform
  // extents, the surrogate block is the exact block of the operator.
  //
  // This is a basic implementation applied on whole vectors cell by cell,
  // to assess the quality of the smoother. Returns false (and sets up
  // nothing) if the mesh has hanging nodes or constraints other than
  // periodicity. Uses the current factor_mass and factor_laplace.
  template<typename OtherNumber>
  bool
  setup_block_inverse(const AffineConstraints<OtherNumber> & constraints)
  {
    block_start.clear();
    block_sizes.clear();
    block_indices.clear();
    block_eigenvectors.clear();
    block_eigenvalues.clear();
    if(Utilities::MPI::max(has_hanging_nodes() ? 1 : 0, dof_handler->get_mpi_communicator()) == 1)
      return false;

    const unsigned int degree        = matrix_free.get_shape_info().data[0].fe_degree;
    const unsigned int nn            = degree + 1;
    const unsigned int n_q_points_1d = matrix_free.get_shape_info().data[0].n_q_points_1d;
    const unsigned int n_q_points_2d = n_q_points_1d * n_q_points_1d;
    const auto &       shape_data    = matrix_free.get_shape_info().data[0];
    AssertDimension(mapping_info.quad_weights_z.size(), n_q_points_1d);

    // 1D mass and stiffness matrices of the element on the unit interval in
    // lexicographic numbering
    std::vector<double> mass_ref(nn * nn), stiffness_ref(nn * nn);
    for(unsigned int i = 0; i < nn; ++i)
      for(unsigned int j = 0; j < nn; ++j)
        for(unsigned int q = 0; q < n_q_points_1d; ++q)
        {
          const double w = mapping_info.quad_weights_z[q];
          mass_ref[i * nn + j] += w * shape_data.shape_values[i * n_q_points_1d + q] *
                                  shape_data.shape_values[j * n_q_points_1d + q];
          stiffness_ref[i * nn + j] += w * shape_data.shape_gradients[i * n_q_points_1d + q] *
                                       shape_data.shape_gradients[j * n_q_points_1d + q];
        }

    const std::vector<unsigned int> hierarchic_to_lexicographic =
      FETools::hierarchic_to_lexicographic_numbering<dim>(degree);
    const Utilities::MPI::Partitioner &  part = *matrix_free.get_dof_info().vector_partitioner;
    std::vector<types::global_dof_index> dof_indices(dof_handler->get_fe().dofs_per_cell),
      lexicographic_indices(dof_indices.size());
    std::vector<double> mass, laplace, eigenvalues, eigenvectors;
    bool                supported = true;

    block_start.push_back(0);
    for(unsigned int cell = 0; cell < matrix_free.n_cell_batches() && supported; ++cell)
      for(unsigned int lane = 0; lane < matrix_free.n_active_entries_per_cell_batch(cell); ++lane)
      {
        const auto cell_it = matrix_free.get_cell_iterator(cell, lane);
        cell_it->get_dof_indices(dof_indices);
        for(unsigned int i = 0; i < dof_indices.size(); ++i)
          lexicographic_indices[hierarchic_to_lexicographic[i]] = dof_indices[i];

        // index ranges of the owned nodes
        std::array<unsigned int, dim> start, size;
        std::array<bool, dim>         has_upper_neighbor;
        for(unsigned int d = 0; d < dim; ++d)
        {
          const auto has_neighbor = [&](const unsigned int face) {
            return !cell_it->at_boundary(face) || cell_it->has_periodic_neighbor(face);
          };
          start[d]              = has_neighbor(2 * d) ? 1 : 0;
          size[d]               = nn - start[d];
          has_upper_neighbor[d] = has_neighbor(2 * d + 1);
        }
        for(unsigned int d = 0; d < dim; ++d)
          block_sizes.push_back(size[d]);

        // unknowns of the block, with periodic constraints resolved
        for(unsigned int i2 = start[dim - 1]; i2 < nn; ++i2)
          for(unsigned int i1 = start[1]; i1 < nn; ++i1)
            for(unsigned int i0 = start[0]; i0 < nn; ++i0)
            {
              types::global_dof_index index = lexicographic_indices[(i2 * nn + i1) * nn + i0];
              if(constraints.is_constrained(index))
              {
                const auto * entries = constraints.get_constraint_entries(index);
                if(!constraints.is_identity_constrained(index) || entries == nullptr ||
                   entries->size() != 1 || std::abs(entries->front().second - 1.) > 1e-12)
                {
                  supported = false;
                  break;
                }
                index = entries->front().first;
              }
              AssertThrow(part.in_local_range(index) || part.is_ghost_entry(index),
                          ExcMessage("Block unknown neither owned nor ghost"));
              block_indices.push_back(part.global_to_local(index));
            }

        // extents of the Cartesian surrogate as in LaplaceOperatorDG
        std::array<double, dim> h;
        {
          const unsigned int base = mapping_info.mapping_data_index[cell][lane];
          double             h0 = 0, h1 = 0, weight_sum = 0;
          for(unsigned int q = 0; q < n_q_points_2d; ++q)
          {
            const double w   = mapping_info.quad_weights_xy[q];
            const auto & jac = mapping_info.jacobians_xy[base + q];
            h0 += w * std::sqrt(jac[0][0] * jac[0][0] + jac[1][0] * jac[1][0]);
            h1 += w * std::sqrt(jac[0][1] * jac[0][1] + jac[1][1] * jac[1][1]);
            weight_sum += w;
          }
          h[0] = h0 / weight_sum;
          h[1] = h1 / weight_sum;
          if constexpr(dim == 3)
            h[2] = mapping_info.h_z_values[base / n_q_points_2d];
        }

        // 1D matrices of the owned nodes and their eigendecomposition
        for(unsigned int d = 0; d < dim; ++d)
        {
          const unsigned int n = size[d];
          mass.assign(n * n, 0.);
          laplace.assign(n * n, 0.);
          for(unsigned int a = 0; a < n; ++a)
            for(unsigned int b = 0; b < n; ++b)
            {
              const unsigned int ia = a + start[d], ib = b + start[d];
              mass[a * n + b]    = h[d] * mass_ref[ia * nn + ib];
              laplace[a * n + b] = factor_laplace / h[d] * stiffness_ref[ia * nn + ib];
            }
          if(has_upper_neighbor[d])
          {
            mass[n * n - 1] += h[d] * mass_ref[0];
            laplace[n * n - 1] += factor_laplace / h[d] * stiffness_ref[0];
          }
          eigenvalues.resize(n);
          eigenvectors.resize(n * n);
          dealii::internal::TensorProductMatrixSymmetricSum::spectral_assembly<double>(
            mass.data(), laplace.data(), n, n, eigenvalues.data(), eigenvectors.data());
          for(unsigned int i = 0; i < n * n; ++i)
            block_eigenvectors.push_back(eigenvectors[i]);
          for(unsigned int i = 0; i < n; ++i)
            block_eigenvalues.push_back(eigenvalues[i] + factor_mass / dim);
        }
        block_start.push_back(block_indices.size());
      }

    if(Utilities::MPI::min(supported ? 1 : 0, dof_handler->get_mpi_communicator()) == 0)
    {
      block_start.clear();
      block_sizes.clear();
      block_indices.clear();
      block_eigenvectors.clear();
      block_eigenvalues.clear();
      return false;
    }
    return true;
  }

  // Variant of setup_block_inverse() with blocks on patches of 2^dim cells,
  // the children of a cell of the next coarser level: a patch has the nodes
  // (I_0, I_1, I_2) with I_d in [0, 2 degree] and owns those with I_d in
  // [s_d, 2 degree], with s_d as for single cells. The 1D matrices of the
  // surrogate are those of two elements, with the extents of the lower and
  // upper element in direction d averaged over the children at that
  // position, plus the contribution of the neighbor patch on the upper face
  // to the shared node with the extent of the upper element. Returns false
  // if a child is missing, not active, not locally owned, or if the mesh
  // has hanging nodes or constraints other than periodicity.
  template<typename OtherNumber>
  bool
  setup_patch_block_inverse(const AffineConstraints<OtherNumber> & constraints)
  {
    block_start.clear();
    block_sizes.clear();
    block_indices.clear();
    block_eigenvectors.clear();
    block_eigenvalues.clear();
    const MPI_Comm comm = dof_handler->get_mpi_communicator();
    if(Utilities::MPI::max(has_hanging_nodes() ? 1 : 0, comm) == 1)
      return false;

    const unsigned int degree        = matrix_free.get_shape_info().data[0].fe_degree;
    const unsigned int nn            = degree + 1;
    const unsigned int n_patch       = 2 * degree + 1;
    const unsigned int n_q_points_1d = matrix_free.get_shape_info().data[0].n_q_points_1d;
    const unsigned int n_q_points_2d = n_q_points_1d * n_q_points_1d;
    const auto &       shape_data    = matrix_free.get_shape_info().data[0];

    std::vector<double> mass_ref(nn * nn), stiffness_ref(nn * nn);
    for(unsigned int i = 0; i < nn; ++i)
      for(unsigned int j = 0; j < nn; ++j)
        for(unsigned int q = 0; q < n_q_points_1d; ++q)
        {
          const double w = mapping_info.quad_weights_z[q];
          mass_ref[i * nn + j] += w * shape_data.shape_values[i * n_q_points_1d + q] *
                                  shape_data.shape_values[j * n_q_points_1d + q];
          stiffness_ref[i * nn + j] += w * shape_data.shape_gradients[i * n_q_points_1d + q] *
                                       shape_data.shape_gradients[j * n_q_points_1d + q];
        }

    // extents of the Cartesian surrogate of each locally owned cell
    std::vector<std::array<double, dim>> cell_extents(
      dof_handler->get_triangulation().n_active_cells());
    for(unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
      for(unsigned int lane = 0; lane < matrix_free.n_active_entries_per_cell_batch(cell); ++lane)
      {
        const unsigned int base = mapping_info.mapping_data_index[cell][lane];
        double             h0 = 0, h1 = 0, weight_sum = 0;
        for(unsigned int q = 0; q < n_q_points_2d; ++q)
        {
          const double w   = mapping_info.quad_weights_xy[q];
          const auto & jac = mapping_info.jacobians_xy[base + q];
          h0 += w * std::sqrt(jac[0][0] * jac[0][0] + jac[1][0] * jac[1][0]);
          h1 += w * std::sqrt(jac[0][1] * jac[0][1] + jac[1][1] * jac[1][1]);
          weight_sum += w;
        }
        std::array<double, dim> & h =
          cell_extents[matrix_free.get_cell_iterator(cell, lane)->active_cell_index()];
        h[0] = h0 / weight_sum;
        h[1] = h1 / weight_sum;
        if constexpr(dim == 3)
          h[2] = mapping_info.h_z_values[base / n_q_points_2d];
      }

    const std::vector<unsigned int> hierarchic_to_lexicographic =
      FETools::hierarchic_to_lexicographic_numbering<dim>(degree);
    const Utilities::MPI::Partitioner &  part = *matrix_free.get_dof_info().vector_partitioner;
    std::vector<types::global_dof_index> dof_indices(dof_handler->get_fe().dofs_per_cell);
    std::vector<std::vector<types::global_dof_index>> child_indices(
      GeometryInfo<dim>::max_children_per_cell,
      std::vector<types::global_dof_index>(dof_indices.size()));
    std::vector<double> mass, laplace, eigenvalues, eigenvectors;
    bool                supported = true;

    block_start.push_back(0);
    for(const auto & cell : dof_handler->active_cell_iterators())
    {
      if(!cell->is_locally_owned())
        continue;
      if(cell->level() == 0 ||
         cell->parent()->n_children() != GeometryInfo<dim>::max_children_per_cell)
      {
        supported = false;
        break;
      }
      const auto parent = cell->parent();
      // each patch is handled by its first child
      if(parent->child(0)->index() != cell->index())
        continue;

      std::array<std::array<double, 2>, dim> h_patch = {};
      for(unsigned int c = 0; c < GeometryInfo<dim>::max_children_per_cell; ++c)
      {
        const auto child = parent->child(c);
        if(!child->is_active() || !child->is_locally_owned())
        {
          supported = false;
          break;
        }
        child->get_dof_indices(dof_indices);
        for(unsigned int i = 0; i < dof_indices.size(); ++i)
          child_indices[c][hierarchic_to_lexicographic[i]] = dof_indices[i];
        for(unsigned int d = 0; d < dim; ++d)
          h_patch[d][(c >> d) & 1] += cell_extents[child->active_cell_index()][d] /
                                      (GeometryInfo<dim>::max_children_per_cell / 2);
      }
      if(!supported)
        break;

      // index ranges of the owned nodes, with the boundary information of
      // the faces of the first (lower) and last (upper) child
      std::array<unsigned int, dim> start, size;
      std::array<bool, dim>         has_upper_neighbor;
      const auto                    first_child = parent->child(0);
      const auto last_child = parent->child(GeometryInfo<dim>::max_children_per_cell - 1);
      for(unsigned int d = 0; d < dim; ++d)
      {
        start[d] =
          !first_child->at_boundary(2 * d) || first_child->has_periodic_neighbor(2 * d) ? 1 : 0;
        size[d] = n_patch - start[d];
        has_upper_neighbor[d] =
          !last_child->at_boundary(2 * d + 1) || last_child->has_periodic_neighbor(2 * d + 1);
      }
      for(unsigned int d = 0; d < dim; ++d)
        block_sizes.push_back(size[d]);

      // unknowns of the patch, from the child containing the node
      for(unsigned int i2 = start[dim - 1]; i2 < n_patch && supported; ++i2)
        for(unsigned int i1 = start[1]; i1 < n_patch && supported; ++i1)
          for(unsigned int i0 = start[0]; i0 < n_patch; ++i0)
          {
            const std::array<unsigned int, 3> patch_index = {{i0, i1, i2}};
            unsigned int                      child = 0, local = 0, stride = 1;
            for(unsigned int d = 0; d < dim; ++d)
            {
              const unsigned int c = std::min(patch_index[d] / degree, 1u);
              child += c << d;
              local += (patch_index[d] - c * degree) * stride;
              stride *= nn;
            }
            types::global_dof_index index = child_indices[child][local];
            if(constraints.is_constrained(index))
            {
              const auto * entries = constraints.get_constraint_entries(index);
              if(!constraints.is_identity_constrained(index) || entries == nullptr ||
                 entries->size() != 1 || std::abs(entries->front().second - 1.) > 1e-12)
              {
                supported = false;
                break;
              }
              index = entries->front().first;
            }
            AssertThrow(part.in_local_range(index) || part.is_ghost_entry(index),
                        ExcMessage("Block unknown neither owned nor ghost"));
            block_indices.push_back(part.global_to_local(index));
          }
      if(!supported)
        break;

      // 1D matrices of the two elements on the owned nodes
      for(unsigned int d = 0; d < dim; ++d)
      {
        std::vector<double> mass_full(n_patch * n_patch), laplace_full(n_patch * n_patch);
        for(unsigned int e = 0; e < 2; ++e)
          for(unsigned int a = 0; a < nn; ++a)
            for(unsigned int b = 0; b < nn; ++b)
            {
              const unsigned int ia = a + e * degree, ib = b + e * degree;
              mass_full[ia * n_patch + ib] += h_patch[d][e] * mass_ref[a * nn + b];
              laplace_full[ia * n_patch + ib] +=
                factor_laplace / h_patch[d][e] * stiffness_ref[a * nn + b];
            }
        if(has_upper_neighbor[d])
        {
          mass_full[n_patch * n_patch - 1] += h_patch[d][1] * mass_ref[0];
          laplace_full[n_patch * n_patch - 1] += factor_laplace / h_patch[d][1] * stiffness_ref[0];
        }
        const unsigned int n = size[d];
        mass.resize(n * n);
        laplace.resize(n * n);
        for(unsigned int a = 0; a < n; ++a)
          for(unsigned int b = 0; b < n; ++b)
          {
            mass[a * n + b]    = mass_full[(a + start[d]) * n_patch + b + start[d]];
            laplace[a * n + b] = laplace_full[(a + start[d]) * n_patch + b + start[d]];
          }
        eigenvalues.resize(n);
        eigenvectors.resize(n * n);
        dealii::internal::TensorProductMatrixSymmetricSum::spectral_assembly<double>(
          mass.data(), laplace.data(), n, n, eigenvalues.data(), eigenvectors.data());
        for(unsigned int i = 0; i < n * n; ++i)
          block_eigenvectors.push_back(eigenvectors[i]);
        for(unsigned int i = 0; i < n; ++i)
          block_eigenvalues.push_back(eigenvalues[i] + factor_mass / dim);
      }
      block_start.push_back(block_indices.size());
    }

    if(Utilities::MPI::min(supported ? 1 : 0, comm) == 0)
    {
      block_start.clear();
      block_sizes.clear();
      block_indices.clear();
      block_eigenvectors.clear();
      block_eigenvalues.clear();
      return false;
    }
    return true;
  }

  bool
  has_block_inverse() const
  {
    return !block_start.empty();
  }

  unsigned int
  n_blocks() const
  {
    return block_start.empty() ? 0 : block_start.size() - 1;
  }

  // --- Vectorized block-Jacobi preconditioner embedded in the cell loop ---
  //
  // The same blocks and Cartesian surrogates as setup_block_inverse(), but
  // restricted to the locally owned unknowns and grouped into batches of
  // n_lanes blocks of arbitrary cells with the same index ranges
  // ("signature"), with their compressed indices gathered from the cell
  // batches. Owned unknowns whose block belongs to a cell of another
  // process ("leftovers") get the point-Jacobi preconditioner. Thus, the
  // preconditioner needs no communication; it agrees with the global
  // block-Jacobi preconditioner of setup_block_inverse() in serial.
  //
  // In vmult_scratch_with_block_update(), each batch runs right after the
  // cell batch T_b at which the result of the operator on all its unknowns
  // has been passed to 'after_loop', reading the residual stored by
  // 'after_loop' for the chunks flagged in get_block_residual_chunks().
  // Optionally (set_block_from_scratch()), the batches compute the
  // residual from the result of the operator still available in the
  // scratch array instead, with artificial writes of the entities of the
  // blocks by cell batch T_b in the scratch layout, determined by a
  // fixed-point iteration. The batches with unknowns whose result is only
  // complete after the exchange with other processes ("interface batches")
  // and the leftovers run after the loop. Lanes without a block
  // repeat the first lane and are masked out. Returns false (and sets up
  // nothing) if the mesh has hanging nodes or the degree is 1. Uses the
  // current factor_mass and factor_laplace.
  bool
  setup_block_batches()
  {
    block_batch_start.clear();
    block_batch_fe_indices.clear();
    block_batch_scratch_indices.clear();
    block_batch_unconstrained.clear();
    block_batch_data_offset.clear();
    block_batch_data.clear();
    block_batch_mask.clear();
    block_batch_index.clear();
    block_batch_order.clear();
    leftover_indices.clear();
    leftover_inverse_diagonal.clear();
    block_residual_chunks.clear();
    const MPI_Comm     comm   = dof_handler->get_mpi_communicator();
    const unsigned int degree = matrix_free.get_shape_info().data[0].fe_degree;
    // the blocks of the cell loop read the residual stored by 'after_loop'
    // (default), or the result of the operator from the scratch array, see
    // set_block_from_scratch()
    setup_scratch();
    if(Utilities::MPI::max(has_hanging_nodes() || degree < 2 ? 1 : 0, comm) == 1)
      return false;

    const unsigned int nn            = degree + 1;
    const unsigned int n_stored      = Utilities::pow(3, dim);
    const unsigned int n_q_points_1d = matrix_free.get_shape_info().data[0].n_q_points_1d;
    const unsigned int n_q_points_2d = n_q_points_1d * n_q_points_1d;
    const auto &       shape_data    = matrix_free.get_shape_info().data[0];
    const unsigned int n_batches     = matrix_free.n_cell_batches();
    const unsigned int n_owned =
      matrix_free.get_dof_info().vector_partitioner->locally_owned_size();
    constexpr unsigned int chunk = ExaDG::ScratchMatrixVectorProduct<Number>::chunk_size;

    std::vector<double> mass_ref(nn * nn), stiffness_ref(nn * nn);
    for(unsigned int i = 0; i < nn; ++i)
      for(unsigned int j = 0; j < nn; ++j)
        for(unsigned int q = 0; q < n_q_points_1d; ++q)
        {
          const double w = mapping_info.quad_weights_z[q];
          mass_ref[i * nn + j] += w * shape_data.shape_values[i * n_q_points_1d + q] *
                                  shape_data.shape_values[j * n_q_points_1d + q];
          stiffness_ref[i * nn + j] += w * shape_data.shape_gradients[i * n_q_points_1d + q] *
                                       shape_data.shape_gradients[j * n_q_points_1d + q];
        }

    // cell batch after which each locally owned entry of the result of
    // vmult_scratch() is passed to 'after_loop', n_batches if only after
    // the exchange
    std::vector<unsigned int> finished_after(n_owned, n_batches);
    const auto                compute_finished_after = [&]() {
      std::fill(finished_after.begin(), finished_after.end(), n_batches);
      scratch.for_each_range_finished_in_cell_loop(
        [&](const unsigned int cell, const unsigned int begin, const unsigned int end) {
          for(unsigned int i = begin; i < end; ++i)
            finished_after[i] = cell;
        });
    };
    compute_finished_after();

    struct BlockInfo
    {
      unsigned int                  cell, lane;
      std::array<unsigned int, dim> start;
      std::vector<unsigned int>     owned_entities;
      std::vector<unsigned int>     unknowns;
      std::vector<double>           data;
    };
    // blocks per signature, key[0] = 1 for the interface blocks
    std::map<std::array<unsigned int, dim + 1>, std::vector<BlockInfo>> blocks;
    std::vector<unsigned char>                                          covered(n_owned, 0);
    bool                                                                supported = true;
    std::vector<double> mass, laplace, eigenvalues, eigenvectors;
    const auto          entity_class = [&](const unsigned int i) {
      return i == 0 ? 0u : (i == degree ? 2u : 1u);
    };

    for(unsigned int cell = 0; cell < n_batches; ++cell)
      for(unsigned int lane = 0; lane < matrix_free.n_active_entries_per_cell_batch(cell); ++lane)
      {
        const auto            cell_it = matrix_free.get_cell_iterator(cell, lane);
        BlockInfo             block;
        std::array<bool, dim> has_upper_neighbor;
        block.cell = cell;
        block.lane = lane;
        for(unsigned int d = 0; d < dim; ++d)
        {
          const auto has_neighbor = [&](const unsigned int face) {
            return !cell_it->at_boundary(face) || cell_it->has_periodic_neighbor(face);
          };
          block.start[d]        = has_neighbor(2 * d) ? 1 : 0;
          has_upper_neighbor[d] = has_neighbor(2 * d + 1);
        }
        for(unsigned int e = 0; e < n_stored; ++e)
        {
          bool owned = true;
          for(unsigned int d = 0, j = e; d < dim; ++d, j /= 3)
            if(j % 3 == 0 && block.start[d] == 1)
              owned = false;
          if(owned)
            block.owned_entities.push_back(e);
        }

        // unknowns of the owned nodes via the entities of the compressed
        // indices; interface blocks have unknowns not owned by this process
        // or only complete after the exchange
        bool interface = false;
        for(unsigned int i2 = block.start[dim - 1]; i2 < nn; ++i2)
          for(unsigned int i1 = block.start[1]; i1 < nn; ++i1)
            for(unsigned int i0 = block.start[0]; i0 < nn; ++i0)
            {
              const std::array<unsigned int, 3> node   = {{i0, i1, i2}};
              unsigned int                      entity = 0, offset = 0, stride_e = 1, stride_o = 1;
              for(unsigned int d = 0; d < dim; ++d)
              {
                const unsigned int c = entity_class(node[d]);
                entity += c * stride_e;
                stride_e *= 3;
                if(c == 1)
                {
                  offset += (node[d] - 1) * stride_o;
                  stride_o *= degree - 1;
                }
              }
              const unsigned int start_index =
                compressed_dof_indices[(cell * n_stored + entity) * n_lanes + lane];
              if(start_index == numbers::invalid_unsigned_int)
              {
                supported = false;
                continue;
              }
              const unsigned int index = start_index + offset;
              if(index >= n_owned || finished_after[index] == n_batches)
                interface = true;
              if(index < n_owned)
              {
                block.unknowns.push_back(index);
                covered[index] = 1;
              }
            }

        // Cartesian surrogate as in setup_block_inverse()
        std::array<double, dim> h;
        {
          const unsigned int base = mapping_info.mapping_data_index[cell][lane];
          double             h0 = 0, h1 = 0, weight_sum = 0;
          for(unsigned int q = 0; q < n_q_points_2d; ++q)
          {
            const double w   = mapping_info.quad_weights_xy[q];
            const auto & jac = mapping_info.jacobians_xy[base + q];
            h0 += w * std::sqrt(jac[0][0] * jac[0][0] + jac[1][0] * jac[1][0]);
            h1 += w * std::sqrt(jac[0][1] * jac[0][1] + jac[1][1] * jac[1][1]);
            weight_sum += w;
          }
          h[0] = h0 / weight_sum;
          h[1] = h1 / weight_sum;
          if constexpr(dim == 3)
            h[2] = mapping_info.h_z_values[base / n_q_points_2d];
        }
        std::vector<double> values_data;
        for(unsigned int d = 0; d < dim; ++d)
        {
          const unsigned int n = nn - block.start[d];
          mass.assign(n * n, 0.);
          laplace.assign(n * n, 0.);
          for(unsigned int a = 0; a < n; ++a)
            for(unsigned int b = 0; b < n; ++b)
            {
              const unsigned int ia = a + block.start[d], ib = b + block.start[d];
              mass[a * n + b]    = h[d] * mass_ref[ia * nn + ib];
              laplace[a * n + b] = factor_laplace / h[d] * stiffness_ref[ia * nn + ib];
            }
          if(has_upper_neighbor[d])
          {
            mass[n * n - 1] += h[d] * mass_ref[0];
            laplace[n * n - 1] += factor_laplace / h[d] * stiffness_ref[0];
          }
          eigenvalues.resize(n);
          eigenvectors.resize(n * n);
          dealii::internal::TensorProductMatrixSymmetricSum::spectral_assembly<double>(
            mass.data(), laplace.data(), n, n, eigenvalues.data(), eigenvectors.data());
          block.data.insert(block.data.end(), eigenvectors.begin(), eigenvectors.end());
          for(unsigned int i = 0; i < n; ++i)
            values_data.push_back(eigenvalues[i] + factor_mass / dim);
        }
        block.data.insert(block.data.end(), values_data.begin(), values_data.end());

        std::array<unsigned int, dim + 1> key;
        key[0] = interface ? 1 : 0;
        for(unsigned int d = 0; d < dim; ++d)
          key[d + 1] = block.start[d];
        blocks[key].push_back(std::move(block));
      }

    if(Utilities::MPI::min(supported ? 1 : 0, comm) == 0)
    {
      setup_scratch();
      return false;
    }

    // leftovers: owned, unconstrained unknowns not in any local block
    {
      std::vector<unsigned char> constrained(n_owned, 0);
      for(const unsigned int i : matrix_free.get_constrained_dofs(0))
        if(i < n_owned)
          constrained[i] = 1;
      VectorType inverse_diagonal;
      compute_inverse_diagonal(inverse_diagonal);
      for(unsigned int i = 0; i < n_owned; ++i)
        if(!covered[i] && !constrained[i])
        {
          leftover_indices.push_back(i);
          leftover_inverse_diagonal.push_back(inverse_diagonal.local_element(i));
        }
    }

    // batches of the blocks of each signature in the order of the cell
    // batch they get ready after, the interface ones in the order of cells
    struct BatchInfo
    {
      std::vector<const BlockInfo *> members;
      unsigned int                   time;
      bool                           interface;
    };
    std::vector<BatchInfo> batches;
    for(auto & [key, list] : blocks)
    {
      std::vector<unsigned int> ready(list.size(), 0);
      for(unsigned int i = 0; i < list.size(); ++i)
        for(const unsigned int u : list[i].unknowns)
          ready[i] = std::max(ready[i], finished_after[u]);
      std::vector<unsigned int> order(list.size());
      std::iota(order.begin(), order.end(), 0u);
      std::stable_sort(order.begin(), order.end(), [&](const unsigned int a, const unsigned int b) {
        return ready[a] < ready[b];
      });
      for(unsigned int first = 0; first < list.size(); first += n_lanes)
      {
        BatchInfo batch;
        batch.interface = key[0] == 1;
        batch.time      = 0;
        for(unsigned int v = first; v < std::min<unsigned int>(first + n_lanes, list.size()); ++v)
        {
          batch.members.push_back(&list[order[v]]);
          batch.time = std::max(batch.time, ready[order[v]]);
        }
        if(batch.interface)
          batch.time = n_batches;
        batches.push_back(std::move(batch));
      }
    }

    // fixed point of the times of the batches and the scratch layout with
    // the artificial writes of the entities of the blocks at those times:
    // the extended lifetimes can delay the chunks passed to 'after_loop'
    std::vector<std::vector<unsigned int>> batches_at(n_batches);
    std::vector<unsigned int>              scratch_indices;
    for(unsigned int iteration = 0;; ++iteration)
    {
      AssertThrow(iteration < 100, ExcInternalError("No fixed point of the block schedule"));
      for(auto & list : batches_at)
        list.clear();
      for(unsigned int b = 0; b < batches.size(); ++b)
        if(!batches[b].interface)
          batches_at[batches[b].time].push_back(b);

      setup_scratch(
        [&](const unsigned int cell, const auto & add) {
          if(!block_from_scratch)
            return;
          for(const unsigned int b : batches_at[cell])
            for(const BlockInfo * block : batches[b].members)
              for(const unsigned int e : block->owned_entities)
              {
                unsigned int length = 1;
                for(unsigned int d = 0, j = e; d < dim; ++d, j /= 3)
                  if(j % 3 == 1)
                    length *= degree - 1;
                add(compressed_dof_indices[(block->cell * n_stored + e) * n_lanes + block->lane],
                    length);
              }
        },
        [&]() {
          compute_finished_after();
          scratch_indices.resize(batches.size() * n_stored * n_lanes);
          for(unsigned int b = 0; b < batches.size(); ++b)
            for(unsigned int e = 0; e < n_stored; ++e)
              for(unsigned int v = 0; v < n_lanes; ++v)
              {
                const BlockInfo * block = batches[b].members[v < batches[b].members.size() ? v : 0];
                scratch_indices[(b * n_stored + e) * n_lanes + v] = scratch.scratch_index(
                  compressed_dof_indices[(block->cell * n_stored + e) * n_lanes + block->lane]);
              }
        });

      if(!block_from_scratch)
        break;
      bool changed = false;
      for(BatchInfo & batch : batches)
        if(!batch.interface)
          for(const BlockInfo * block : batch.members)
            for(const unsigned int u : block->unknowns)
              if(finished_after[u] > batch.time)
              {
                AssertThrow(finished_after[u] < n_batches, ExcInternalError());
                batch.time = finished_after[u];
                changed    = true;
              }
      if(!changed)
        break;
    }

    // store the batches in the order of execution
    std::vector<std::vector<unsigned int>> order_after(n_batches + 1);
    for(unsigned int b = 0; b < batches.size(); ++b)
      order_after[batches[b].time].push_back(b);
    for(unsigned int b = 0; b < batches.size(); ++b)
    {
      const BatchInfo &              batch  = batches[b];
      const unsigned int             n_fill = batch.members.size();
      std::array<unsigned char, dim> start;
      for(unsigned int d = 0; d < dim; ++d)
        start[d] = batch.members[0]->start[d];
      block_batch_start.push_back(start);
      for(unsigned int e = 0; e < n_stored; ++e)
      {
        bool all_valid = true;
        for(unsigned int v = 0; v < n_lanes; ++v)
        {
          const BlockInfo *  block = batch.members[v < n_fill ? v : 0];
          const unsigned int index =
            compressed_dof_indices[(block->cell * n_stored + e) * n_lanes + block->lane];
          block_batch_fe_indices.push_back(index);
          block_batch_scratch_indices.push_back(scratch_indices[(b * n_stored + e) * n_lanes + v]);
          all_valid = all_valid && index != numbers::invalid_unsigned_int;
        }
        block_batch_unconstrained.push_back(all_valid ? 1 : 0);
      }
      block_batch_data_offset.push_back(block_batch_data.size());
      const unsigned int data_size = batch.members[0]->data.size();
      const unsigned int offset    = block_batch_data.size();
      block_batch_data.resize_fast(offset + data_size);
      for(unsigned int i = 0; i < data_size; ++i)
        for(unsigned int v = 0; v < n_lanes; ++v)
          block_batch_data[offset + i][v] = batch.members[v < n_fill ? v : 0]->data[i];
      VectorizedArray<Number> mask = 0.;
      for(unsigned int v = 0; v < n_fill; ++v)
        mask[v] = 1.;
      block_batch_mask.push_back(mask);
    }
    block_batch_index.push_back(0);
    for(const auto & list : order_after)
    {
      block_batch_order.insert(block_batch_order.end(), list.begin(), list.end());
      block_batch_index.push_back(block_batch_order.size());
    }

    // chunks of the residual needed after the loop
    block_residual_chunks.assign(n_owned / chunk + 1, 0);
    for(const BatchInfo & batch : batches)
      if(batch.interface)
        for(const BlockInfo * block : batch.members)
          for(const unsigned int u : block->unknowns)
            block_residual_chunks[u / chunk] = 1;
    for(const unsigned int i : leftover_indices)
      block_residual_chunks[i / chunk] = 1;
    if(!block_from_scratch)
      for(const BatchInfo & batch : batches)
        for(const BlockInfo * block : batch.members)
          for(const unsigned int u : block->unknowns)
            block_residual_chunks[u / chunk] = 1;

    matrix_free.initialize_dof_vector(block_input);
    return true;
  }

  bool
  has_block_batches() const
  {
    return !block_batch_index.empty();
  }

  // Flags per chunk of ScratchMatrixVectorProduct::chunk_size locally owned
  // entries whether the residual must be stored for the block updates after
  // the loop in vmult_scratch_with_block_update()
  const std::vector<unsigned char> &
  get_block_residual_chunks() const
  {
    return block_residual_chunks;
  }

  // Apply the vectorized block-Jacobi preconditioner dst = P^{-1} src on
  // whole vectors. Entries not part of any block or leftover, i.e.,
  // constrained ones, are set to zero.
  void
  apply_block_inverse_vectorized(VectorType & dst, const VectorType & src) const
  {
    Assert(has_block_batches(), ExcMessage("Call setup_block_batches() first"));
    // the input with zero ghost entries, as the unknowns of other processes
    // are dropped from the blocks
    block_input.copy_locally_owned_data_from(src);
    dst = 0;
    for(const unsigned int b : block_batch_order)
      dispatch_block_batch<false>(b, nullptr, block_input, dst, Number(1.));
    for(unsigned int k = 0; k < leftover_indices.size(); ++k)
      dst.local_element(leftover_indices[k]) +=
        leftover_inverse_diagonal[k] * src.local_element(leftover_indices[k]);
    dst.zero_out_ghost_values();
  }

  // Variant of vmult_scratch() that, after each cell batch, adds
  // factor * P^{-1} (rhs - A src) on the blocks of the batches scheduled
  // there into 'target', with A src taken from the scratch array. The
  // interface batches and the leftovers run after the loop, with the
  // residual stored by 'after_loop' in 'residual' for the chunks flagged by
  // get_block_residual_chunks(). The entries of 'target' must already hold
  // their final values apart from this addition when the blocks run, which
  // holds when 'after_loop' writes them.
  template<typename AfterLoop>
  void
  vmult_scratch_with_block_update(const VectorType & src,
                                  const AfterLoop &  after_loop,
                                  const VectorType & rhs,
                                  const VectorType & residual,
                                  VectorType &       target,
                                  const Number       factor) const
  {
    Assert(has_block_batches(), ExcMessage("Call setup_block_batches() first"));
    Timer              total_timer;
    const unsigned int n_batches = matrix_free.n_cell_batches();
    do_vmult_scratch(
      src, nullptr, after_loop, [&](const unsigned int position, const Number * scratch_array) {
        for(unsigned int i = block_batch_index[position]; i < block_batch_index[position + 1]; ++i)
          if(block_from_scratch)
            dispatch_block_batch<true>(block_batch_order[i], scratch_array, rhs, target, factor);
          else
            dispatch_block_batch<false>(block_batch_order[i], nullptr, residual, target, factor);
      });
    // do_vmult_scratch() counts the product, so only add the time below
    timings[t_scratch_count] -= 1;
    timings[t_scratch_total] -= total_timer.wall_time();

    // interface batches and leftovers; the blocks also write into ghost
    // entries of 'target', which get discarded
    Timer interface_timer;
    for(unsigned int i = block_batch_index[n_batches]; i < block_batch_index[n_batches + 1]; ++i)
      dispatch_block_batch<false>(block_batch_order[i], nullptr, residual, target, factor);
    for(unsigned int k = 0; k < leftover_indices.size(); ++k)
      target.local_element(leftover_indices[k]) +=
        factor * leftover_inverse_diagonal[k] * residual.local_element(leftover_indices[k]);
    target.zero_out_ghost_values();
    timings[t_block_interface] += interface_timer.wall_time();

    timings[t_scratch_count] += 1;
    timings[t_scratch_total] += total_timer.wall_time();
  }

  // Local vector indices of the unknowns of a block, in lexicographic order
  // of the owned nodes
  ArrayView<const unsigned int>
  get_block_indices(const unsigned int block) const
  {
    return ArrayView<const unsigned int>(block_indices.data() + block_start[block],
                                         block_start[block + 1] - block_start[block]);
  }

  // Apply the block-Jacobi preconditioner dst = P^{-1} src. Entries that are
  // not part of any block, i.e., constrained ones, are set to zero.
  void
  apply_block_inverse(VectorType & dst, const VectorType & src) const
  {
    Assert(has_block_inverse(), ExcMessage("Call setup_block_inverse() first"));
    const bool src_has_ghosts = src.has_ghost_elements();
    if(!src_has_ghosts)
      src.update_ghost_values();
    dst = 0;
    dst.zero_out_ghost_values();

    std::vector<double> values, tmp;
    unsigned int        offset_vectors = 0, offset_values = 0;
    for(unsigned int block = 0; block < n_blocks(); ++block)
    {
      const auto                    indices = get_block_indices(block);
      std::array<unsigned int, dim> size;
      for(unsigned int d = 0; d < dim; ++d)
        size[d] = block_sizes[block * dim + d];
      values.resize(indices.size());
      tmp.resize(indices.size());
      for(unsigned int i = 0; i < indices.size(); ++i)
        values[i] = src.local_element(indices[i]);

      std::array<const Number *, dim> eigenvectors, eigenvalues;
      for(unsigned int d = 0; d < dim; ++d)
      {
        eigenvectors[d] = block_eigenvectors.data() + offset_vectors;
        eigenvalues[d]  = block_eigenvalues.data() + offset_values;
        offset_vectors += size[d] * size[d];
        offset_values += size[d];
      }

      // values = S (Lambda^{-1} (S^T values))
      for(unsigned int d = 0; d < dim; ++d)
        apply_1d_matrix(eigenvectors[d], true, d, size, values.data(), tmp.data());
      for(unsigned int i2 = 0, c = 0; i2 < (dim == 3 ? size[2] : 1); ++i2)
        for(unsigned int i1 = 0; i1 < size[1]; ++i1)
          for(unsigned int i0 = 0; i0 < size[0]; ++i0, ++c)
          {
            const double sum = eigenvalues[0][i0] + eigenvalues[1][i1] +
                               (dim == 3 ? double(eigenvalues[dim - 1][i2]) : 0.);
            values[c] = sum > 1e-12 * std::abs(eigenvalues[0][size[0] - 1]) ? values[c] / sum : 0.;
          }
      for(unsigned int d = dim; d-- > 0;)
        apply_1d_matrix(eigenvectors[d], false, d, size, values.data(), tmp.data());

      for(unsigned int i = 0; i < indices.size(); ++i)
        dst.local_element(indices[i]) += values[i];
    }
    dst.compress(VectorOperation::add);
    if(!src_has_ghosts)
      src.zero_out_ghost_values();
  }

  // Add the cell-local values of the given cell batch into dst. On cells
  // with hanging nodes, the cell-local values are first multiplied by the
  // transpose of the hanging-node interpolation, unless apply_hanging_nodes
  // is false (e.g. for the diagonal). Note that dof_values get overwritten
  // in that case.
  template<int degree, bool apply_hanging_nodes = true>
  void
  distribute_local_to_global(const unsigned int        cell,
                             VectorizedArray<Number> * dof_values,
                             VectorType &              dst) const
  {
    if constexpr(apply_hanging_nodes)
      apply_hanging_node_constraints<degree>(cell, true, dof_values);
    scatter_dof_values<degree>(cell, compressed_dof_indices, dof_values, dst);
  }

  // Read the cell-local values of the given cell batch from src, i.e., the
  // values including the hanging-node interpolation on cells with hanging
  // nodes
  template<int degree>
  void
  read_dof_values(const unsigned int        cell,
                  const VectorType &        src,
                  VectorizedArray<Number> * dof_values) const
  {
    if constexpr(degree <= 2)
    {
      constexpr unsigned int dofs_per_cell = Utilities::pow(degree + 1, dim);
      const unsigned int *   dof_indices =
        compressed_dof_indices.data() + cell * dofs_per_cell * n_lanes;
      const unsigned char * unconstrained = all_indices_unconstrained.data() + cell * dofs_per_cell;
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        if(unconstrained[i])
          for(unsigned int v = 0; v < n_lanes; ++v)
            dof_values[i][v] = src.local_element(dof_indices[i * n_lanes + v]);
        else
        {
          dof_values[i] = 0.;
          for(unsigned int v = 0; v < n_lanes; ++v)
            if(dof_indices[i * n_lanes + v] != numbers::invalid_unsigned_int)
              dof_values[i][v] = src.local_element(dof_indices[i * n_lanes + v]);
        }
    }
    else
      read_dof_values_compressed<degree, degree + 1, 1>(
        src, compressed_dof_indices, all_indices_unconstrained, cell, {}, true, dof_values);

    apply_hanging_node_constraints<degree>(cell, false, dof_values);
  }

  // Apply the operator on the given cell batch and return the result in the
  // nodal basis at all (degree+1)^dim nodes of the cell, without the
  // transpose of the hanging-node interpolation and without summation into a
  // global vector. The values of src are read including the hanging-node
  // interpolation and constraints, so src must have its ghost values
  // updated. This is the cell contribution to a residual that gets
  // restricted cell by cell, see the transfer classes of the Poisson
  // multigrid.
  template<int degree>
  void
  compute_cell_operation_nodal(const unsigned int        cell,
                               const VectorType &        src,
                               VectorizedArray<Number> * values) const
  {
    if(degree <= 2 || (!batch_has_hanging_nodes.empty() && batch_has_hanging_nodes[cell]))
    {
      read_dof_values<degree>(cell, src, values);
      apply_cell_operation_local<degree>(cell, values);
    }
    else if constexpr(degree > 2)
      compute_cell_operation_nodal_fast<degree>(cell, src, values);
  }

  // Part of compute_cell_operation_nodal() for cells without hanging nodes
  // and degree > 2
  template<int degree>
  void
  compute_cell_operation_nodal_fast(const unsigned int        cell,
                                    const VectorType &        src,
                                    VectorizedArray<Number> * values) const
  {
    // fast path: interleave the first step of sum factorization with the
    // read operation
    const internal::MatrixFreeFunctions::UnivariateShapeData<Number> & shape_data =
      matrix_free.get_shape_info().data[0];
    read_dof_values_compressed<degree, degree + 1, 1>(src,
                                                      compressed_dof_indices,
                                                      all_indices_unconstrained,
                                                      cell,
                                                      shape_data.shape_values_eo,
                                                      false,
                                                      values);
    AssertThrow(factor_laplace != 0, ExcNotImplemented());
    compute_cell_lapl<degree + 1>(
      shape_data, mapping_info, cell, factor_mass, factor_laplace, values, values);
    dealii::internal::FEEvaluationImplBasisChange<
      dealii::internal::evaluate_evenodd,
      dealii::internal::EvaluatorQuantity::value,
      dim,
      degree + 1,
      degree + 1>::do_backward(1, shape_data.shape_values_eo, false, values, values);
  }

  // Read the nodal values of a single cell, given by the cell batch and
  // lane, from src (with ghost values) into out in lexicographic order,
  // without the hanging-node interpolation, entity by entity with
  // contiguous runs in src. The lanes must not have hanging nodes.
  template<int degree>
  void
  read_cell_values_scalar(const unsigned int cell,
                          const unsigned int lane,
                          const VectorType & src,
                          Number *           out) const
  {
    constexpr unsigned int n_stored = Utilities::pow(degree == 1 ? 2 : 3, dim);
    if(scalar_entity_start.empty())
      setup_scalar_read<degree>();
    const unsigned int * indices = compressed_dof_indices.data() + cell * n_stored * n_lanes + lane;
    const Number *       src_ptr = src.begin();
    for(unsigned int i = 0; i < n_stored; ++i)
    {
      const unsigned int index = indices[i * n_lanes];
      const unsigned int begin = scalar_entity_start[i], end = scalar_entity_start[i + 1];
      if(index == numbers::invalid_unsigned_int)
        for(unsigned int k = begin; k < end; ++k)
          out[scalar_entity_node[k]] = 0;
      else
        for(unsigned int k = begin; k < end; ++k)
          out[scalar_entity_node[k]] = src_ptr[index + k - begin];
    }
  }

  // Set up the node of each entry of the entities for
  // read_cell_values_scalar(), by reading a vector holding its own indices
  // with the compressed read
  template<int degree>
  void
  setup_scalar_read() const
  {
    constexpr unsigned int    n_stored      = Utilities::pow(degree == 1 ? 2 : 3, dim);
    constexpr unsigned int    dofs_per_cell = Utilities::pow(degree + 1, dim);
    std::vector<unsigned int> length(n_stored, 1);
    if(degree > 1)
      for(unsigned int i = 0; i < n_stored; ++i)
        for(unsigned int d = 0, j = i; d < dim; ++d, j /= 3)
          if(j % 3 == 1)
            length[i] *= degree - 1;
    scalar_entity_start.assign(n_stored + 1, 0);
    for(unsigned int i = 0; i < n_stored; ++i)
      scalar_entity_start[i + 1] = scalar_entity_start[i] + length[i];
    AssertDimension(scalar_entity_start.back(), dofs_per_cell);
    scalar_entity_node.assign(dofs_per_cell, numbers::invalid_unsigned_int);
    if constexpr(degree <= 2)
    {
      for(unsigned int i = 0; i < n_stored; ++i)
        scalar_entity_node[i] = i;
    }
    else
    {
      const unsigned int max_length = Utilities::pow(degree - 1, dim);
      VectorType         indices_as_values;
      indices_as_values.reinit(n_stored * max_length);
      for(unsigned int k = 0; k < n_stored * max_length; ++k)
        indices_as_values(k) = k;
      std::vector<unsigned int> indices(n_stored * n_lanes);
      for(unsigned int i = 0; i < n_stored; ++i)
        for(unsigned int v = 0; v < n_lanes; ++v)
          indices[i * n_lanes + v] = i * max_length;
      const std::vector<unsigned char> unconstrained(n_stored, 1);
      VectorizedArray<Number>          values[dofs_per_cell];
      read_dof_values_compressed<degree, degree + 1, 1>(
        indices_as_values, indices, unconstrained, 0, {}, true, values);
      for(unsigned int node = 0; node < dofs_per_cell; ++node)
      {
        const unsigned int value = static_cast<unsigned int>(values[node][0]);
        const unsigned int i = value / max_length, k = value % max_length;
        AssertIndexRange(k, length[i]);
        scalar_entity_node[scalar_entity_start[i] + k] = node;
      }
    }
    for(const unsigned int node : scalar_entity_node)
      AssertThrow(node != numbers::invalid_unsigned_int, ExcInternalError());
  }

  const std::vector<unsigned int> &
  get_compressed_dof_indices() const
  {
    return compressed_dof_indices;
  }

  // Return the compressed indices where the entities that are replaced by
  // the coarse side of a hanging-node constraint are set to invalid, i.e.,
  // only those DoFs that are actually owned by the cell (in the sense of the
  // unconstrained finite element space) are kept. This is the index set
  // needed for the fine side of transfer operators.
  std::vector<unsigned int>
  get_compressed_dof_indices_without_hanging_nodes() const
  {
    std::vector<unsigned int> indices = compressed_dof_indices;
    for(const unsigned int i : positions_replaced_by_hanging_nodes)
      indices[i] = numbers::invalid_unsigned_int;
    return indices;
  }

  bool
  has_hanging_nodes() const
  {
    return !hanging_node_masks.empty();
  }

  // Return the hanging-node constraint kind of all lanes of the cell batches,
  // empty if the mesh does not have hanging nodes
  const std::vector<
    std::array<internal::MatrixFreeFunctions::compressed_constraint_kind, n_lanes>> &
  get_hanging_node_masks() const
  {
    return hanging_node_masks;
  }

private:
  // Apply the hanging-node interpolation (transpose = false) or its
  // transpose to the cell-local values of a cell batch, doing nothing on
  // batches without hanging nodes
  template<int degree>
  void
  apply_hanging_node_constraints(const unsigned int        cell,
                                 const bool                transpose,
                                 VectorizedArray<Number> * dof_values) const
  {
    if(!batch_has_hanging_nodes.empty() && batch_has_hanging_nodes[cell])
      internal::FEEvaluationImplHangingNodes<dim, VectorizedArray<Number>>::template run<degree,
                                                                                         Number>(
        1, matrix_free.get_shape_info(), transpose, hanging_node_masks[cell], dof_values);
  }

  // Same as above for a single lane of a cell batch with values of type
  // Number, used by the transfer operators
  template<int degree>
  void
  apply_hanging_node_constraints(const unsigned int cell,
                                 const unsigned int lane,
                                 const bool         transpose,
                                 Number *           dof_values) const
  {
    if(batch_has_hanging_nodes.empty() ||
       hanging_node_masks[cell][lane] ==
         internal::MatrixFreeFunctions::unconstrained_compressed_constraint_kind)
      return;

    // broadcast the values to all lanes with the same mask, as the kernel
    // works on full cell batches
    const unsigned int dofs_per_cell = matrix_free.get_shape_info().dofs_per_component_on_cell;
    AlignedVector<VectorizedArray<Number>> values(dofs_per_cell);
    for(unsigned int i = 0; i < dofs_per_cell; ++i)
      values[i] = dof_values[i];
    std::array<internal::MatrixFreeFunctions::compressed_constraint_kind, n_lanes> mask;
    mask.fill(hanging_node_masks[cell][lane]);
    internal::FEEvaluationImplHangingNodes<dim, VectorizedArray<Number>>::template run<degree,
                                                                                       Number>(
      1, matrix_free.get_shape_info(), transpose, mask, values.data());
    for(unsigned int i = 0; i < dofs_per_cell; ++i)
      dof_values[i] = values[i][0];
  }

  // Add the values of a cell batch into dst at the positions given by
  // dst_indices, in the format of compressed_dof_indices
  template<int degree, typename DstVectorType>
  void
  scatter_dof_values(const unsigned int                cell,
                     const std::vector<unsigned int> & dst_indices,
                     VectorizedArray<Number> *         dof_values,
                     DstVectorType &                   dst) const
  {
    if constexpr(degree <= 2)
    {
      constexpr unsigned int dofs_per_cell = Utilities::pow(degree + 1, dim);
      const unsigned int *   dof_indices   = dst_indices.data() + cell * dofs_per_cell * n_lanes;
      const unsigned char * unconstrained = all_indices_unconstrained.data() + cell * dofs_per_cell;
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        if(unconstrained[i])
          for(unsigned int v = 0; v < n_lanes; ++v)
            dst.local_element(dof_indices[i * n_lanes + v]) += dof_values[i][v];
        else
        {
          for(unsigned int v = 0; v < n_lanes; ++v)
            if(dof_indices[i * n_lanes + v] != numbers::invalid_unsigned_int)
              dst.local_element(dof_indices[i * n_lanes + v]) += dof_values[i][v];
        }
    }
    else
      distribute_local_to_global_compressed<degree, degree + 1, 1>(
        dst, dst_indices, all_indices_unconstrained, cell, {}, true, dof_values);
  }

  // Apply the operator on the cell-local values of a cell batch, given and
  // returned in the nodal basis in values
  template<int degree>
  void
  apply_cell_operation_local(const unsigned int cell_index, VectorizedArray<Number> * values) const
  {
    const internal::MatrixFreeFunctions::UnivariateShapeData<Number> & shape_data =
      matrix_free.get_shape_info().data[0];
    dealii::internal::FEEvaluationImplBasisChange<dealii::internal::evaluate_evenodd,
                                                  dealii::internal::EvaluatorQuantity::value,
                                                  dim,
                                                  degree + 1,
                                                  degree + 1>::do_forward(1,
                                                                          shape_data
                                                                            .shape_values_eo,
                                                                          values,
                                                                          values);
    AssertThrow(factor_laplace != 0, ExcNotImplemented());
    compute_cell_lapl<degree + 1>(
      shape_data, mapping_info, cell_index, factor_mass, factor_laplace, values, values);
    dealii::internal::FEEvaluationImplBasisChange<
      dealii::internal::evaluate_evenodd,
      dealii::internal::EvaluatorQuantity::value,
      dim,
      degree + 1,
      degree + 1>::do_backward(1, shape_data.shape_values_eo, false, values, values);
  }

  // Set up the scratch array of vmult_scratch() with the vertices, lines,
  // quads and hexes of the element as entities
  // Set up the scratch array of vmult_scratch(). Optionally, the cell batch
  // 'cell' additionally writes the entities given by
  // 'extra_entities(cell, add)', which keeps them alive in the scratch
  // array until that cell batch, and 'before_release' runs before the setup
  // data of the scratch array is released, e.g. to query scratch_index().
  void
  setup_scratch(const std::function<void(const unsigned int,
                                         const typename ExaDG::ScratchMatrixVectorProduct<
                                           Number>::EntityFunction &)> & extra_entities = {},
                const std::function<void()> &                            before_release = {})
  {
    const unsigned int fe_degree        = dof_handler->get_fe().degree;
    const unsigned int n_stored_per_dir = fe_degree == 1 ? 2 : 3;
    const unsigned int n_stored         = Utilities::pow(n_stored_per_dir, dim);

    // length of the entities in lexicographic order of the stored indices,
    // given by the number of directions in the interior of the cell
    std::vector<unsigned int> entity_length(n_stored, 1);
    if(fe_degree > 1)
      for(unsigned int i = 0; i < n_stored; ++i)
        for(unsigned int d = 0, j = i; d < dim; ++d, j /= 3)
          if(j % 3 == 1)
            entity_length[i] *= fe_degree - 1;

    const unsigned int n_cell_batches = matrix_free.n_cell_batches();
    scratch.reinit(
      matrix_free.get_dof_info().vector_partitioner,
      n_cell_batches,
      [&](const unsigned int position, const auto & add) {
        const unsigned int cell = loop_order[position];
        for(unsigned int i = 0; i < n_stored; ++i)
          for(unsigned int v = 0; v < n_lanes; ++v)
            add(compressed_dof_indices[(cell * n_stored + i) * n_lanes + v], entity_length[i]);
        if(extra_entities)
          extra_entities(position, add);
      },
      true);

    scratch_dof_indices = compressed_dof_indices;
    for(unsigned int & idx : scratch_dof_indices)
      idx = scratch.scratch_index(idx);
    if(before_release)
      before_release();
    scratch.release_setup_data();
    setup_pre_list();
  }

  // Set up the ranges of locally owned entries for 'before_loop' in
  // vmult_scratch(), in units of 'pre_list_chunk_size' entries: the
  // entries sent to other processes are processed before the ghost
  // exchange (index n_cell_batches), all others before the first cell batch
  // reading them or the cell batch after which their result is passed on
  // (for the constrained entries, which no cell reads).
  //
  // This follows DoFInfo::compute_vector_zero_access_pattern() in deal.II,
  // but the pre list of MatrixFree cannot be used here: it puts the entries
  // not read by any cell at the last partition, as the cell loop of
  // MatrixFree sets the constrained entries only at the end, whereas the
  // scratch loop sets them when finalizing their chunk, which can be much
  // earlier. Furthermore, the partitions of MatrixFree span several cell
  // batches, so the lists here are closer to the first read.
  void
  setup_pre_list()
  {
    cell_loop_pre_list_index.clear();
    cell_loop_pre_list.clear();
    const std::shared_ptr<const Utilities::MPI::Partitioner> & partitioner =
      matrix_free.get_dof_info().vector_partitioner;
    const unsigned int n_owned = partitioner->locally_owned_size();
    if(n_owned < min_size_for_pre_list)
      return;

    const unsigned int        fe_degree        = dof_handler->get_fe().degree;
    const unsigned int        n_stored_per_dir = fe_degree == 1 ? 2 : 3;
    const unsigned int        n_stored         = Utilities::pow(n_stored_per_dir, dim);
    std::vector<unsigned int> entity_length(n_stored, 1);
    if(fe_degree > 1)
      for(unsigned int i = 0; i < n_stored; ++i)
        for(unsigned int d = 0, j = i; d < dim; ++d, j /= 3)
          if(j % 3 == 1)
            entity_length[i] *= fe_degree - 1;

    constexpr unsigned int    chunk_size     = pre_list_chunk_size;
    const unsigned int        n_cell_batches = matrix_free.n_cell_batches();
    std::vector<unsigned int> first_touch((n_owned + chunk_size - 1) / chunk_size,
                                          numbers::invalid_unsigned_int);
    const auto                touch =
      [&](const unsigned int cell, const unsigned int begin, const unsigned int end) {
        for(unsigned int j = begin / chunk_size; j <= (end - 1) / chunk_size; ++j)
          first_touch[j] = std::min(first_touch[j], cell);
      };

    // reads by the cells, which coincide with the entities they write, at
    // their position in the loop
    for(unsigned int position = 0; position < n_cell_batches; ++position)
      for(unsigned int i = 0; i < n_stored; ++i)
        for(unsigned int v = 0; v < n_lanes; ++v)
        {
          const unsigned int idx =
            compressed_dof_indices[(loop_order[position] * n_stored + i) * n_lanes + v];
          if(idx < n_owned)
            touch(position, idx, idx + entity_length[i]);
        }

    // results passed on in the cell loop, including the constrained entries
    scratch.for_each_range_finished_in_cell_loop(
      [&](const unsigned int cell, const unsigned int begin, const unsigned int end) {
        if(end > begin)
          touch(cell, begin, end);
      });

    // entries sent to other processes, and those only passed on after the
    // cell loop without being read by a cell
    for(const auto & [begin, end] : partitioner->import_indices())
      for(unsigned int j = begin / chunk_size; end > begin && j <= (end - 1) / chunk_size; ++j)
        first_touch[j] = n_cell_batches;
    for(unsigned int & cell : first_touch)
      if(cell == numbers::invalid_unsigned_int)
        cell = n_cell_batches;

    std::map<unsigned int, std::vector<unsigned int>> chunks_by_cell;
    for(unsigned int j = 0; j < first_touch.size(); ++j)
      chunks_by_cell[first_touch[j]].push_back(j);
    Helper::convert_map_to_range_list(n_cell_batches + 1,
                                      chunk_size,
                                      chunks_by_cell,
                                      cell_loop_pre_list_index,
                                      cell_loop_pre_list,
                                      n_owned);
  }

  void
  local_apply(const MatrixFree<dim, Number> &               data,
              VectorType &                                  dst,
              const VectorType &                            src,
              const std::pair<unsigned int, unsigned int> & cell_range) const
  {
    const unsigned int degree = data.get_shape_info().data[0].fe_degree;
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      if(degree == 1)
        do_cell_operation<1, true>(cell, src, dst);
      else if(degree == 2)
        do_cell_operation<2, true>(cell, src, dst);
      else if(degree == 3)
        do_cell_operation<3, true>(cell, src, dst);
      else if(degree == 4)
        do_cell_operation<4, true>(cell, src, dst);
#ifndef DEBUG
      else if(degree == 5)
        do_cell_operation<5, true>(cell, src, dst);
      else if(degree == 6)
        do_cell_operation<6, true>(cell, src, dst);
      else if(degree == 7)
        do_cell_operation<7, true>(cell, src, dst);
      else if(degree == 8)
        do_cell_operation<8, true>(cell, src, dst);
#endif
      else
        AssertThrow(false, ExcMessage("Degree " + std::to_string(degree) + " not instantiated"));
    }
  }

  void
  local_compute_diagonal(const MatrixFree<dim, Number> & data,
                         VectorType &                    dst,
                         const unsigned int &,
                         const std::pair<unsigned int, unsigned int> & cell_range) const
  {
    const unsigned int degree = data.get_shape_info().data[0].fe_degree;
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      if(degree == 1)
        do_cell_operation<1, false>(cell, dst, dst);
      else if(degree == 2)
        do_cell_operation<2, false>(cell, dst, dst);
      else if(degree == 3)
        do_cell_operation<3, false>(cell, dst, dst);
      else if(degree == 4)
        do_cell_operation<4, false>(cell, dst, dst);
#ifndef DEBUG
      else if(degree == 5)
        do_cell_operation<5, false>(cell, dst, dst);
      else if(degree == 6)
        do_cell_operation<6, false>(cell, dst, dst);
      else if(degree == 7)
        do_cell_operation<7, false>(cell, dst, dst);
      else if(degree == 8)
        do_cell_operation<8, false>(cell, dst, dst);
#endif
      else
        AssertThrow(false, ExcMessage("Degree " + std::to_string(degree) + " not instantiated"));
    }
  }

  // Apply the operator on a cell batch and add the result into dst at the
  // positions given by dst_indices, in the format of compressed_dof_indices
  template<int degree, typename DstVectorType>
  void
  apply_cell_operation(const unsigned int                cell_index,
                       const VectorType &                src,
                       const std::vector<unsigned int> & dst_indices,
                       DstVectorType &                   dst) const
  {
    constexpr unsigned int  n_q_points_1d = degree + 1;
    constexpr unsigned int  n_q_points    = Utilities::pow(n_q_points_1d, dim);
    VectorizedArray<Number> quad_values[n_q_points];

    // Cell batches with hanging nodes: the hanging-node interpolation acts
    // on the nodal values, so read the values without interleaving the
    // first step of sum factorization as done in the fast path below
    if(!batch_has_hanging_nodes.empty() && batch_has_hanging_nodes[cell_index])
    {
      read_dof_values<degree>(cell_index, src, quad_values);
      apply_cell_operation_local<degree>(cell_index, quad_values);
      apply_hanging_node_constraints<degree>(cell_index, true, quad_values);
      scatter_dof_values<degree>(cell_index, dst_indices, quad_values, dst);
      return;
    }

    const internal::MatrixFreeFunctions::UnivariateShapeData<Number> & shape_data =
      matrix_free.get_shape_info().data[0];
    if(degree <= 2)
    {
      const unsigned int * dof_indices =
        compressed_dof_indices.data() + cell_index * Utilities::pow(degree + 1, dim) * n_lanes;
      const unsigned char * unconstrained =
        all_indices_unconstrained.data() + cell_index * Utilities::pow(degree + 1, dim);
      for(unsigned int i = 0; i < n_q_points; ++i)
        if(unconstrained[i])
          for(unsigned int v = 0; v < n_lanes; ++v)
            quad_values[i][v] = src.local_element(dof_indices[i * n_lanes + v]);
        else
        {
          quad_values[i] = VectorizedArray<Number>();
          for(unsigned int v = 0; v < n_lanes; ++v)
            if(dof_indices[i * n_lanes + v] != numbers::invalid_unsigned_int)
              quad_values[i][v] = src.local_element(dof_indices[i * n_lanes + v]);
        }

      dealii::internal::FEEvaluationImplBasisChange<dealii::internal::evaluate_evenodd,
                                                    dealii::internal::EvaluatorQuantity::value,
                                                    dim,
                                                    degree + 1,
                                                    degree + 1>::do_forward(1,
                                                                            shape_data
                                                                              .shape_values_eo,
                                                                            quad_values,
                                                                            quad_values);
    }
    else
      read_dof_values_compressed<degree, n_q_points_1d, 1>(src,
                                                           compressed_dof_indices,
                                                           all_indices_unconstrained,
                                                           cell_index,
                                                           shape_data.shape_values_eo,
                                                           false,
                                                           quad_values);

    if(factor_laplace != 0)
      compute_cell_lapl<n_q_points_1d>(shape_data,
                                       mapping_info,
                                       cell_index,
                                       factor_mass,
                                       factor_laplace,
                                       quad_values,
                                       quad_values);
    else
      AssertThrow(false, ExcNotImplemented());

    if(degree <= 2)
    {
      dealii::internal::FEEvaluationImplBasisChange<
        dealii::internal::evaluate_evenodd,
        dealii::internal::EvaluatorQuantity::value,
        dim,
        degree + 1,
        degree + 1>::do_backward(1, shape_data.shape_values_eo, false, quad_values, quad_values);
      const unsigned int * dof_indices =
        dst_indices.data() + cell_index * Utilities::pow(degree + 1, dim) * n_lanes;
      const unsigned char * unconstrained =
        all_indices_unconstrained.data() + cell_index * Utilities::pow(degree + 1, dim);
      for(unsigned int i = 0; i < n_q_points; ++i)
        if(unconstrained[i])
          for(unsigned int v = 0; v < n_lanes; ++v)
            dst.local_element(dof_indices[i * n_lanes + v]) += quad_values[i][v];
        else
        {
          for(unsigned int v = 0; v < n_lanes; ++v)
            if(dof_indices[i * n_lanes + v] != numbers::invalid_unsigned_int)
              dst.local_element(dof_indices[i * n_lanes + v]) += quad_values[i][v];
        }
    }
    else
      distribute_local_to_global_compressed<degree, n_q_points_1d, 1>(dst,
                                                                      dst_indices,
                                                                      all_indices_unconstrained,
                                                                      cell_index,
                                                                      shape_data.shape_values_eo,
                                                                      false,
                                                                      quad_values);
  }

  template<int degree, bool evaluate_operator>
  void
  do_cell_operation(const unsigned int cell_index, const VectorType & src, VectorType & dst) const
  {
    constexpr unsigned int  n_q_points_1d = degree + 1;
    constexpr unsigned int  n_q_points    = Utilities::pow(n_q_points_1d, dim);
    VectorizedArray<Number> quad_values[n_q_points];

    const internal::MatrixFreeFunctions::UnivariateShapeData<Number> & shape_data =
      matrix_free.get_shape_info().data[0];
    if(evaluate_operator)
      apply_cell_operation<degree>(cell_index, src, compressed_dof_indices, dst);
    else if(!batch_has_hanging_nodes.empty() && batch_has_hanging_nodes[cell_index])
    {
      // On cells with hanging nodes, compute the diagonal of C^T A C, with
      // C the hanging-node interpolation, column by column
      VectorizedArray<Number> diag_values[n_q_points];
      for(unsigned int i = 0; i < n_q_points; ++i)
      {
        for(unsigned int j = 0; j < n_q_points; ++j)
          quad_values[j] = VectorizedArray<Number>();
        quad_values[i] = Number(1.);
        apply_hanging_node_constraints<degree>(cell_index, false, quad_values);
        apply_cell_operation_local<degree>(cell_index, quad_values);
        apply_hanging_node_constraints<degree>(cell_index, true, quad_values);
        diag_values[i] = quad_values[i];
      }
      distribute_local_to_global<degree, false>(cell_index, diag_values, dst);
    }
    else
    {
      VectorizedArray<Number> diag_values[n_q_points];
      compute_cell_lapl_diagonal<n_q_points_1d>(shape_data,
                                                mapping_info,
                                                cell_index,
                                                factor_mass,
                                                factor_laplace,
                                                static_cast<const VectorizedArray<Number> *>(
                                                  nullptr),
                                                diag_values);
      distribute_local_to_global<degree, false>(cell_index, diag_values, dst);
    }
  }

public:
  template<int fe_degree, int n_q_points_1d, int n_components, typename SrcVectorType = VectorType>
  static void
  read_dof_values_compressed(const SrcVectorType &                      vec_in,
                             const std::vector<unsigned int> &          compressed_dof_indices,
                             const std::vector<unsigned char> &         all_indices_unconstrained,
                             const unsigned int                         cell_no,
                             const dealii::AlignedVector<Number> &      shape_values_eo,
                             const bool                                 is_collocation,
                             VectorizedArray<Number> * DEAL_II_RESTRICT dof_values)
  {
    SrcVectorType & vec = const_cast<SrcVectorType &>(vec_in);
    AssertIndexRange(cell_no * dealii::Utilities::pow(3, dim) * n_lanes,
                     compressed_dof_indices.size());
    const unsigned int * cell_indices =
      compressed_dof_indices.data() + cell_no * n_lanes * dealii::Utilities::pow(3, dim);
    const unsigned char * cell_unconstrained =
      all_indices_unconstrained.data() + cell_no * dealii::Utilities::pow(3, dim);
    constexpr unsigned int dofs_per_comp = dealii::Utilities::pow(n_q_points_1d, dim);
    dealii::internal::VectorReader<Number, VectorizedArray<Number>> reader;

    // Create a local copy to a stack variable of the shape values, such that
    // the compiler can decide on the most optimal placement of variables in
    // registers, rather than having to consider potential aliasing between
    // the 'shape_values_eo' variable and the other variables used during sum
    // factorization.
    dealii::VectorizedArray<Number> shape_val[(fe_degree + 1) * ((n_q_points_1d + 1) / 2)];
    if(is_collocation == false)
    {
      AssertDimension(shape_values_eo.size(), (fe_degree + 1) * ((n_q_points_1d + 1) / 2));
      for(unsigned int i = 0; i < (fe_degree + 1) * ((n_q_points_1d + 1) / 2); ++i)
        shape_val[i] = shape_values_eo[i];
    }

    for(unsigned int i2 = 0, compressed_i2 = 0, offset_i2 = 0;
        i2 < (dim == 3 ? (fe_degree + 1) : 1);
        ++i2)
    {
      bool all_unconstrained = true;
      for(unsigned int i = 0; i < 9; ++i)
        if(cell_unconstrained[9 * compressed_i2 + i] == 0)
        {
          all_unconstrained = false;
          break;
        }
      if(n_components == 1 && fe_degree < 8 && all_unconstrained)
      {
        const unsigned int *   indices       = cell_indices + 9 * n_lanes * compressed_i2;
        constexpr unsigned int dofs_per_line = (fe_degree - 1);
        // first line
        DEAL_II_OPENMP_SIMD_PRAGMA
        for(unsigned int v = 0; v < n_lanes; ++v)
          dof_values[0][v] = vec.local_element(offset_i2 + indices[v]);
        indices += n_lanes;
        dealii::vectorized_load_and_transpose(dofs_per_line,
                                              vec.begin() + offset_i2 * dofs_per_line,
                                              indices,
                                              dof_values + 1);
        indices += n_lanes;
        DEAL_II_OPENMP_SIMD_PRAGMA
        for(unsigned int v = 0; v < n_lanes; ++v)
          dof_values[fe_degree][v] = vec.local_element(offset_i2 + indices[v]);
        indices += n_lanes;

        // inner part
        VectorizedArray<Number> tmp[fe_degree > 1 ? (fe_degree - 1) * (fe_degree - 1) : 1];
        dealii::vectorized_load_and_transpose(
          dofs_per_line, vec.begin() + offset_i2 * (fe_degree - 1) * n_components, indices, tmp);
        for(unsigned int i0 = 0; i0 < dofs_per_line; ++i0)
          dof_values[(fe_degree + 1) * (i0 + 1)] = tmp[i0];
        indices += n_lanes;

        constexpr unsigned int dofs_per_quad = (fe_degree - 1) * (fe_degree - 1);
        dealii::vectorized_load_and_transpose(dofs_per_quad,
                                              vec.begin() + offset_i2 * dofs_per_quad,
                                              indices,
                                              tmp);
        for(unsigned int i1 = 0; i1 < dofs_per_line; ++i1)
          for(unsigned int i0 = 0; i0 < dofs_per_line; ++i0)
            dof_values[(i1 + 1) * (fe_degree + 1) + i0 + 1] = tmp[i1 * dofs_per_line + i0];
        indices += n_lanes;

        dealii::vectorized_load_and_transpose(dofs_per_line,
                                              vec.begin() + offset_i2 * dofs_per_line,
                                              indices,
                                              tmp);
        for(unsigned int i0 = 0; i0 < dofs_per_line; ++i0)
          dof_values[(fe_degree + 1) * (i0 + 1) + fe_degree] = tmp[i0];
        indices += n_lanes;

        // last line
        constexpr unsigned int i = fe_degree * (fe_degree + 1);
        DEAL_II_OPENMP_SIMD_PRAGMA
        for(unsigned int v = 0; v < n_lanes; ++v)
          dof_values[i][v] = vec.local_element(offset_i2 + indices[v]);
        indices += n_lanes;
        dealii::vectorized_load_and_transpose(dofs_per_line,
                                              vec.begin() + offset_i2 * dofs_per_line,
                                              indices,
                                              dof_values + i + 1);
        indices += n_lanes;
        DEAL_II_OPENMP_SIMD_PRAGMA
        for(unsigned int v = 0; v < n_lanes; ++v)
          dof_values[i + fe_degree][v] = vec.local_element(offset_i2 + indices[v]);
        indices += n_lanes;
      }
      else
        for(unsigned int i1 = 0, i = 0, compressed_i1 = 0, offset_i1 = 0;
            i1 < (dim > 1 ? (fe_degree + 1) : 1);
            ++i1)
        {
          const unsigned int offset =
            (compressed_i1 == 1 ? fe_degree - 1 : 1) * offset_i2 + offset_i1;
          const unsigned int * indices =
            cell_indices + 3 * n_lanes * (compressed_i2 * 3 + compressed_i1);
          const unsigned char * unconstrained =
            cell_unconstrained + 3 * (compressed_i2 * 3 + compressed_i1);

          // left end point
          if(unconstrained[0])
            for(unsigned int c = 0; c < n_components; ++c)
              reader.process_dof_gather(indices,
                                        vec,
                                        offset * n_components + c,
                                        vec.begin() + offset * n_components + c,
                                        dof_values[i + c * dofs_per_comp],
                                        std::integral_constant<bool, true>());
          else
            for(unsigned int v = 0; v < n_lanes; ++v)
              for(unsigned int c = 0; c < n_components; ++c)
                dof_values[i + c * dofs_per_comp][v] =
                  (indices[v] == dealii::numbers::invalid_unsigned_int) ?
                    0. :
                    vec.local_element(indices[v] + offset * n_components + c);
          ++i;
          indices += n_lanes;

          // interior points of line
          if(unconstrained[1])
          {
            constexpr unsigned int  dofs_per_line = (fe_degree - 1) * n_components;
            VectorizedArray<Number> tmp[fe_degree > 1 ? dofs_per_line : 1];
            dealii::vectorized_load_and_transpose(dofs_per_line,
                                                  vec.begin() + offset * dofs_per_line,
                                                  indices,
                                                  tmp);
            for(unsigned int i0 = 0; i0 < fe_degree - 1; ++i0, ++i)
              for(unsigned int c = 0; c < n_components; ++c)
                dof_values[i + c * dofs_per_comp] = tmp[i0 * n_components + c];
          }
          else
            for(unsigned int i0 = 0; i0 < fe_degree - 1; ++i0, ++i)
              for(unsigned int v = 0; v < n_lanes; ++v)
                if(indices[v] != dealii::numbers::invalid_unsigned_int)
                  for(unsigned int c = 0; c < n_components; ++c)
                    dof_values[i + c * dofs_per_comp][v] = vec.local_element(
                      indices[v] + (offset * (fe_degree - 1) + i0) * n_components + c);
                else
                  for(unsigned int c = 0; c < n_components; ++c)
                    dof_values[i + c * dofs_per_comp][v] = 0.;
          indices += n_lanes;

          // right end point
          if(unconstrained[2])
            for(unsigned int c = 0; c < n_components; ++c)
              reader.process_dof_gather(indices,
                                        vec,
                                        offset * n_components + c,
                                        vec.begin() + offset * n_components + c,
                                        dof_values[i + c * dofs_per_comp],
                                        std::integral_constant<bool, true>());
          else
            for(unsigned int v = 0; v < n_lanes; ++v)
              for(unsigned int c = 0; c < n_components; ++c)
                dof_values[i + c * dofs_per_comp][v] =
                  (indices[v] == dealii::numbers::invalid_unsigned_int) ?
                    0. :
                    vec.local_element(indices[v] + offset * n_components + c);
          ++i;

          if(i1 == 0 || i1 == fe_degree - 1)
          {
            ++compressed_i1;
            offset_i1 = 0;
          }
          else
            ++offset_i1;
        }
      if(!is_collocation)
        for(unsigned int c = 0; c < n_components; ++c)
        {
          for(int i = fe_degree; i >= 0; --i)
            internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                  internal::EvaluatorQuantity::value,
                                                  fe_degree + 1,
                                                  n_q_points_1d,
                                                  1,
                                                  1,
                                                  true,
                                                  false>(shape_val,
                                                         dof_values + i * (fe_degree + 1) +
                                                           c * dofs_per_comp,
                                                         dof_values + i * n_q_points_1d +
                                                           c * dofs_per_comp);
          for(unsigned int i = 0; i < n_q_points_1d; ++i)
            internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                  internal::EvaluatorQuantity::value,
                                                  fe_degree + 1,
                                                  n_q_points_1d,
                                                  n_q_points_1d,
                                                  n_q_points_1d,
                                                  true,
                                                  false>(shape_val,
                                                         dof_values + i + c * dofs_per_comp,
                                                         dof_values + i + c * dofs_per_comp);
        }

      if(i2 == 0 || i2 == fe_degree - 1)
      {
        ++compressed_i2;
        offset_i2 = 0;
      }
      else
        ++offset_i2;

      dof_values += n_q_points_1d * n_q_points_1d;
    }

    dof_values -= dofs_per_comp;
    if(dim == 3 && !is_collocation)
      for(unsigned int c = 0; c < n_components; ++c)
        for(unsigned int q1 = 0; q1 < n_q_points_1d * n_q_points_1d; ++q1)
          internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                internal::EvaluatorQuantity::value,
                                                fe_degree + 1,
                                                n_q_points_1d,
                                                n_q_points_1d * n_q_points_1d,
                                                n_q_points_1d * n_q_points_1d,
                                                true,
                                                false>(shape_val,
                                                       dof_values + q1 + c * dofs_per_comp,
                                                       dof_values + q1 + c * dofs_per_comp);
  }

  template<int fe_degree, int n_q_points_1d, int n_components, typename DstVectorType>
  static void
  distribute_local_to_global_compressed(
    DstVectorType &                            vec,
    const std::vector<unsigned int> &          compressed_dof_indices,
    const std::vector<unsigned char> &         all_indices_unconstrained,
    const unsigned int                         cell_no,
    const dealii::AlignedVector<Number> &      shape_values_eo,
    const bool                                 is_collocation,
    VectorizedArray<Number> * DEAL_II_RESTRICT dof_values)
  {
    AssertIndexRange(cell_no * dealii::Utilities::pow(3, dim) * n_lanes,
                     compressed_dof_indices.size());
    const unsigned int * cell_indices =
      compressed_dof_indices.data() + cell_no * n_lanes * dealii::Utilities::pow(3, dim);
    const unsigned char * cell_unconstrained =
      all_indices_unconstrained.data() + cell_no * dealii::Utilities::pow(3, dim);
    constexpr unsigned int dofs_per_comp = dealii::Utilities::pow(n_q_points_1d, dim);
    dealii::internal::VectorDistributorLocalToGlobal<Number, VectorizedArray<Number>> distributor;

    // Create a local copy to a stack variable of the shape values, such that
    // the compiler can decide on the most optimal placement of variables in
    // registers, rather than having to consider potential aliasing between
    // the 'shape_values_eo' variable and the other variables used during sum
    // factorization.
    dealii::VectorizedArray<Number> shape_val[(fe_degree + 1) * ((n_q_points_1d + 1) / 2)];
    if(is_collocation == false)
    {
      AssertDimension(shape_values_eo.size(), (fe_degree + 1) * ((n_q_points_1d + 1) / 2));
      for(unsigned int i = 0; i < (fe_degree + 1) * ((n_q_points_1d + 1) / 2); ++i)
        shape_val[i] = shape_values_eo[i];
    }

    if(dim == 3 && !is_collocation)
      for(unsigned int c = 0; c < n_components; ++c)
        for(unsigned int q1 = 0; q1 < n_q_points_1d * n_q_points_1d; ++q1)
          internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                internal::EvaluatorQuantity::value,
                                                fe_degree + 1,
                                                n_q_points_1d,
                                                n_q_points_1d * n_q_points_1d,
                                                n_q_points_1d * n_q_points_1d,
                                                false,
                                                false>(shape_val,
                                                       dof_values + q1 + c * dofs_per_comp,
                                                       dof_values + q1 + c * dofs_per_comp);

    for(unsigned int i2 = 0, compressed_i2 = 0, offset_i2 = 0;
        i2 < (dim == 3 ? (fe_degree + 1) : 1);
        ++i2)
    {
      if(!is_collocation)
        for(unsigned int c = 0; c < n_components; ++c)
        {
          for(unsigned int i = 0; i < n_q_points_1d; ++i)
            internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                  internal::EvaluatorQuantity::value,
                                                  fe_degree + 1,
                                                  n_q_points_1d,
                                                  n_q_points_1d,
                                                  n_q_points_1d,
                                                  false,
                                                  false>(shape_val,
                                                         dof_values + i + c * dofs_per_comp,
                                                         dof_values + i + c * dofs_per_comp);
          for(unsigned int i = 0; i < fe_degree + 1; ++i)
            internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                                  internal::EvaluatorQuantity::value,
                                                  fe_degree + 1,
                                                  n_q_points_1d,
                                                  1,
                                                  1,
                                                  false,
                                                  false>(shape_val,
                                                         dof_values + i * n_q_points_1d +
                                                           c * dofs_per_comp,
                                                         dof_values + i * (fe_degree + 1) +
                                                           c * dofs_per_comp);
        }
      bool all_unconstrained = true;
      for(unsigned int i = 0; i < 9; ++i)
        if(cell_unconstrained[9 * compressed_i2 + i] == 0)
          all_unconstrained = false;
      if(n_components == 1 && fe_degree < 8 && all_unconstrained)
      {
        const unsigned int *   indices       = cell_indices + 9 * n_lanes * compressed_i2;
        constexpr unsigned int dofs_per_line = fe_degree - 1;
        // first line
        for(unsigned int v = 0; v < n_lanes; ++v)
          vec.local_element(indices[v] + offset_i2) += dof_values[0][v];
        indices += n_lanes;
        dealii::vectorized_transpose_and_store(
          true, dofs_per_line, dof_values + 1, indices, vec.begin() + offset_i2 * dofs_per_line);
        indices += n_lanes;
        for(unsigned int v = 0; v < n_lanes; ++v)
          vec.local_element(indices[v] + offset_i2) += dof_values[fe_degree][v];
        indices += n_lanes;

        // inner part
        constexpr unsigned int  dofs_per_quad = (fe_degree - 1) * (fe_degree - 1);
        VectorizedArray<Number> tmp[fe_degree > 1 ? dofs_per_quad : 1];
        for(unsigned int i0 = 0; i0 < dofs_per_line; ++i0)
          tmp[i0] = dof_values[(fe_degree + 1) * (i0 + 1)];
        dealii::vectorized_transpose_and_store(
          true, dofs_per_line, tmp, indices, vec.begin() + offset_i2 * dofs_per_line);
        indices += n_lanes;

        for(unsigned int i1 = 0; i1 < dofs_per_line; ++i1)
          for(unsigned int i0 = 0; i0 < dofs_per_line; ++i0)
            tmp[i1 * dofs_per_line + i0] = dof_values[(i1 + 1) * (fe_degree + 1) + i0 + 1];
        dealii::vectorized_transpose_and_store(
          true, dofs_per_quad, tmp, indices, vec.begin() + offset_i2 * dofs_per_quad);
        indices += n_lanes;

        for(unsigned int i0 = 0; i0 < dofs_per_line; ++i0)
          tmp[i0] = dof_values[(fe_degree + 1) * (i0 + 1) + fe_degree];
        dealii::vectorized_transpose_and_store(true,
                                               dofs_per_line,
                                               tmp,
                                               indices,
                                               vec.begin() +
                                                 offset_i2 * (fe_degree - 1) * n_components);
        indices += n_lanes;

        // last line
        constexpr unsigned int i = fe_degree * (fe_degree + 1);
        for(unsigned int v = 0; v < n_lanes; ++v)
          vec.local_element(indices[v] + offset_i2) += dof_values[i][v];
        indices += n_lanes;
        dealii::vectorized_transpose_and_store(true,
                                               dofs_per_line,
                                               dof_values + i + 1,
                                               indices,
                                               vec.begin() + offset_i2 * dofs_per_line);
        indices += n_lanes;
        for(unsigned int v = 0; v < n_lanes; ++v)
          vec.local_element(indices[v] + offset_i2) += dof_values[i + fe_degree][v];
        indices += n_lanes;
      }
      else
        for(unsigned int i1 = 0, i = 0, compressed_i1 = 0, offset_i1 = 0;
            i1 < (dim > 1 ? (fe_degree + 1) : 1);
            ++i1)
        {
          const unsigned int offset =
            (compressed_i1 == 1 ? fe_degree - 1 : 1) * offset_i2 + offset_i1;
          const unsigned int * indices =
            cell_indices + 3 * n_lanes * (compressed_i2 * 3 + compressed_i1);
          const unsigned char * unconstrained =
            cell_unconstrained + 3 * (compressed_i2 * 3 + compressed_i1);

          // left end point
          if(unconstrained[0])
            for(unsigned int c = 0; c < n_components; ++c)
              distributor.process_dof_gather(indices,
                                             vec,
                                             offset * n_components + c,
                                             vec.begin() + offset * n_components + c,
                                             dof_values[i + c * dofs_per_comp],
                                             std::integral_constant<bool, true>());
          else
            for(unsigned int v = 0; v < n_lanes; ++v)
              for(unsigned int c = 0; c < n_components; ++c)
                if(indices[v] != dealii::numbers::invalid_unsigned_int)
                  vec.local_element(indices[v] + offset * n_components + c) +=
                    dof_values[i + c * dofs_per_comp][v];
          ++i;
          indices += n_lanes;

          // interior points of line
          if(unconstrained[1])
          {
            VectorizedArray<Number> tmp[fe_degree > 1 ? (fe_degree - 1) * n_components : 1];
            for(unsigned int i0 = 0; i0 < fe_degree - 1; ++i0, ++i)
              for(unsigned int c = 0; c < n_components; ++c)
                tmp[i0 * n_components + c] = dof_values[i + c * dofs_per_comp];

            constexpr unsigned int n_regular = (fe_degree - 1) * n_components / 4 * 4;
            dealii::vectorized_transpose_and_store(
              true, n_regular, tmp, indices, vec.begin() + offset * (fe_degree - 1) * n_components);
            for(unsigned int i0 = n_regular; i0 < (fe_degree - 1) * n_components; ++i0)
              distributor.process_dof_gather(indices,
                                             vec,
                                             offset * (fe_degree - 1) * n_components + i0,
                                             vec.begin() + offset * (fe_degree - 1) * n_components +
                                               i0,
                                             tmp[i0],
                                             std::integral_constant<bool, true>());
          }
          else
            for(unsigned int i0 = 0; i0 < fe_degree - 1; ++i0, ++i)
              for(unsigned int v = 0; v < n_lanes; ++v)
                if(indices[v] != dealii::numbers::invalid_unsigned_int)
                  for(unsigned int c = 0; c < n_components; ++c)
                    vec.local_element(indices[v] + (offset * (fe_degree - 1) + i0) * n_components +
                                      c) += dof_values[i + c * dofs_per_comp][v];
          indices += n_lanes;

          // right end point
          if(unconstrained[2])
            for(unsigned int c = 0; c < n_components; ++c)
              distributor.process_dof_gather(indices,
                                             vec,
                                             offset * n_components + c,
                                             vec.begin() + offset * n_components + c,
                                             dof_values[i + c * dofs_per_comp],
                                             std::integral_constant<bool, true>());
          else
            for(unsigned int v = 0; v < n_lanes; ++v)
              for(unsigned int c = 0; c < n_components; ++c)
                if(indices[v] != dealii::numbers::invalid_unsigned_int)
                  vec.local_element(indices[v] + offset * n_components + c) +=
                    dof_values[i + c * dofs_per_comp][v];
          ++i;

          if(i1 == 0 || i1 == fe_degree - 1)
          {
            ++compressed_i1;
            offset_i1 = 0;
          }
          else
            ++offset_i1;
        }
      if(i2 == 0 || i2 == fe_degree - 1)
      {
        ++compressed_i2;
        offset_i2 = 0;
      }
      else
        ++offset_i2;
      dof_values += n_q_points_1d * n_q_points_1d;
    }
  }

  // Read the cell-local values of a single lane of a cell batch, i.e., the
  // values including the hanging-node interpolation on cells with hanging
  // nodes
  template<int fe_degree_templ = -1>
  void
  read_dof_values_compressed(const VectorType & vec,
                             const unsigned int cell_no,
                             const unsigned int simd_lane,
                             Number *           dof_values) const
  {
    read_dof_values_compressed_raw<fe_degree_templ>(vec, cell_no, simd_lane, dof_values);
    apply_hanging_node_constraints<fe_degree_templ>(cell_no, simd_lane, false, dof_values);
  }

  // Add the cell-local values of a single lane of a cell batch into vec,
  // multiplied by the transpose of the hanging-node interpolation on cells
  // with hanging nodes
  template<int fe_degree_templ = -1>
  void
  distribute_local_to_global_compressed(VectorType &       vec,
                                        const unsigned int cell_no,
                                        const unsigned int simd_lane,
                                        const Number *     dof_values) const
  {
    if(!batch_has_hanging_nodes.empty() &&
       hanging_node_masks[cell_no][simd_lane] !=
         internal::MatrixFreeFunctions::unconstrained_compressed_constraint_kind)
    {
      std::vector<Number> values(dof_values,
                                 dof_values +
                                   matrix_free.get_shape_info().dofs_per_component_on_cell);
      apply_hanging_node_constraints<fe_degree_templ>(cell_no, simd_lane, true, values.data());
      distribute_local_to_global_compressed_raw<fe_degree_templ>(vec,
                                                                 cell_no,
                                                                 simd_lane,
                                                                 values.data());
    }
    else
      distribute_local_to_global_compressed_raw<fe_degree_templ>(vec,
                                                                 cell_no,
                                                                 simd_lane,
                                                                 dof_values);
  }

private:
  // Read the values at the positions of the compressed indices of a single
  // lane of a cell batch, without hanging-node interpolation
  template<int fe_degree_templ = -1>
  void
  read_dof_values_compressed_raw(const VectorType & vec,
                                 const unsigned int cell_no,
                                 const unsigned int simd_lane,
                                 Number *           dof_values) const
  {
    const unsigned int fe_degree =
      fe_degree_templ > 0 ? fe_degree_templ : matrix_free.get_dof_handler().get_fe().degree;
    AssertDimension(fe_degree, matrix_free.get_dof_handler().get_fe().degree);

    if(fe_degree == 1)
    {
      AssertIndexRange(cell_no * n_lanes * dealii::Utilities::pow(2, dim) + simd_lane,
                       compressed_dof_indices.size());
      const unsigned int * cell_indices = compressed_dof_indices.data() +
                                          cell_no * n_lanes * dealii::Utilities::pow(2, dim) +
                                          simd_lane;
      for(unsigned int i = 0; i < dealii::Utilities::pow(2, dim); ++i)
        if(cell_indices[i * n_lanes] != numbers::invalid_unsigned_int)
          dof_values[i] = vec.local_element(cell_indices[i * n_lanes]);
        else
          dof_values[i] = 0;
      return;
    }

    AssertIndexRange(cell_no * n_lanes * dealii::Utilities::pow(3, dim) + simd_lane,
                     compressed_dof_indices.size());
    const unsigned int * cell_indices = compressed_dof_indices.data() +
                                        cell_no * n_lanes * dealii::Utilities::pow(3, dim) +
                                        simd_lane;

    for(unsigned int i2 = 0, compressed_i2 = 0, offset_i2 = 0;
        i2 < (dim == 3 ? (fe_degree + 1) : 1);
        ++i2)
    {
      for(unsigned int i1 = 0, i = 0, compressed_i1 = 0, offset_i1 = 0;
          i1 < (dim > 1 ? (fe_degree + 1) : 1);
          ++i1)
      {
        const unsigned int offset =
          (compressed_i1 == 1 ? fe_degree - 1 : 1) * offset_i2 + offset_i1;
        const unsigned int * indices =
          cell_indices + 3 * n_lanes * (compressed_i2 * 3 + compressed_i1);

        // left end point
        dof_values[i] = (indices[0] == dealii::numbers::invalid_unsigned_int) ?
                          0. :
                          vec.local_element(indices[0] + offset);
        ++i;
        indices += n_lanes;

        // interior points of line
        if(indices[0] != dealii::numbers::invalid_unsigned_int)
          for(unsigned int i0 = 0; i0 < fe_degree - 1; ++i0, ++i)
            dof_values[i] = vec.local_element(indices[0] + offset * (fe_degree - 1) + i0);
        else
          for(unsigned int i0 = 0; i0 < fe_degree - 1; ++i0, ++i)
            dof_values[i] = 0.;
        indices += n_lanes;

        // right end point
        dof_values[i] = (indices[0] == dealii::numbers::invalid_unsigned_int) ?
                          0. :
                          vec.local_element(indices[0] + offset);
        ++i;

        if(i1 == 0 || i1 == fe_degree - 1)
        {
          ++compressed_i1;
          offset_i1 = 0;
        }
        else
          ++offset_i1;
      }

      if(i2 == 0 || i2 == fe_degree - 1)
      {
        ++compressed_i2;
        offset_i2 = 0;
      }
      else
        ++offset_i2;

      dof_values += (fe_degree + 1) * (fe_degree + 1);
    }
  }

  // Add values into the positions of the compressed indices of a single
  // lane of a cell batch, without hanging-node interpolation
  template<int fe_degree_templ = -1>
  void
  distribute_local_to_global_compressed_raw(VectorType &       vec,
                                            const unsigned int cell_no,
                                            const unsigned int simd_lane,
                                            const Number *     dof_values) const
  {
    const unsigned int fe_degree =
      fe_degree_templ > 0 ? fe_degree_templ : matrix_free.get_dof_handler().get_fe().degree;
    AssertDimension(fe_degree, matrix_free.get_dof_handler().get_fe().degree);

    if(fe_degree == 1)
    {
      AssertIndexRange(cell_no * n_lanes * dealii::Utilities::pow(2, dim) + simd_lane,
                       compressed_dof_indices.size());
      const unsigned int * cell_indices = compressed_dof_indices.data() +
                                          cell_no * n_lanes * dealii::Utilities::pow(2, dim) +
                                          simd_lane;
      for(unsigned int i = 0; i < dealii::Utilities::pow(2, dim); ++i)
        if(cell_indices[i * n_lanes] != numbers::invalid_unsigned_int)
          vec.local_element(cell_indices[i * n_lanes]) += dof_values[i];
      return;
    }

    AssertIndexRange(cell_no * n_lanes * dealii::Utilities::pow(3, dim) + simd_lane,
                     compressed_dof_indices.size());
    const unsigned int * cell_indices = compressed_dof_indices.data() +
                                        cell_no * n_lanes * dealii::Utilities::pow(3, dim) +
                                        simd_lane;

    for(unsigned int i2 = 0, compressed_i2 = 0, offset_i2 = 0;
        i2 < (dim == 3 ? (fe_degree + 1) : 1);
        ++i2)
    {
      for(unsigned int i1 = 0, i = 0, compressed_i1 = 0, offset_i1 = 0;
          i1 < (dim > 1 ? (fe_degree + 1) : 1);
          ++i1)
      {
        const unsigned int offset =
          (compressed_i1 == 1 ? fe_degree - 1 : 1) * offset_i2 + offset_i1;
        const unsigned int * indices =
          cell_indices + 3 * n_lanes * (compressed_i2 * 3 + compressed_i1);

        // left end point
        if(indices[0] != dealii::numbers::invalid_unsigned_int)
          vec.local_element(indices[0] + offset) += dof_values[i];
        ++i;
        indices += n_lanes;

        // interior points of line
        if(indices[0] != dealii::numbers::invalid_unsigned_int)
          for(unsigned int i0 = 0; i0 < fe_degree - 1; ++i0, ++i)
            vec.local_element(indices[0] + (offset * (fe_degree - 1) + i0)) += dof_values[i];
        indices += n_lanes;

        // right end point
        if(indices[0] != dealii::numbers::invalid_unsigned_int)
          vec.local_element(indices[0] + offset) += dof_values[i];
        ++i;

        if(i1 == 0 || i1 == fe_degree - 1)
        {
          ++compressed_i1;
          offset_i1 = 0;
        }
        else
          ++offset_i1;
      }
      if(i2 == 0 || i2 == fe_degree - 1)
      {
        ++compressed_i2;
        offset_i2 = 0;
      }
      else
        ++offset_i2;
      dof_values += (fe_degree + 1) * (fe_degree + 1);
    }
  }

private:
  // Multiply the data of a block with sizes 'size' (lexicographic, the
  // first direction running fastest) by the square matrix of direction
  // 'direction', stored with rows by node and columns by eigenvalue index,
  // or its transpose, in place, using 'tmp' of the same size as scratch
  static void
  apply_1d_matrix(const Number *                        matrix,
                  const bool                            transpose,
                  const unsigned int                    direction,
                  const std::array<unsigned int, dim> & size,
                  double *                              data,
                  double *                              tmp)
  {
    const unsigned int n      = size[direction];
    unsigned int       stride = 1;
    for(unsigned int d = 0; d < direction; ++d)
      stride *= size[d];
    unsigned int n_total = 1;
    for(unsigned int d = 0; d < dim; ++d)
      n_total *= size[d];
    const unsigned int n_outer = n_total / (stride * n);
    for(unsigned int o = 0; o < n_outer; ++o)
      for(unsigned int s = 0; s < stride; ++s)
      {
        const unsigned int base = o * stride * n + s;
        for(unsigned int i = 0; i < n; ++i)
        {
          double sum = 0;
          for(unsigned int j = 0; j < n; ++j)
            sum += (transpose ? matrix[j * n + i] : matrix[i * n + j]) * data[base + j * stride];
          tmp[base + i * stride] = sum;
        }
      }
    std::copy(tmp, tmp + n_total, data);
  }

  // data of the block-Jacobi preconditioner, see setup_block_inverse():
  // per block the start in block_indices, the sizes per direction, the
  // local indices of the unknowns, and the eigenvectors and eigenvalues per
  // direction (concatenated over the blocks)
  std::vector<unsigned int> block_start;
  std::vector<unsigned int> block_sizes;
  std::vector<unsigned int> block_indices;
  std::vector<Number>       block_eigenvectors;
  std::vector<Number>       block_eigenvalues;

  // data of the vectorized block-Jacobi preconditioner, see
  // setup_block_batches(): per block batch the start of the owned index
  // range per direction, the gathered compressed indices into the vector
  // and into the scratch array (n_stored entries with n_lanes indices each)
  // with flags whether all lanes are valid, the offset into
  // block_batch_data (eigenvectors per direction, then eigenvalues per
  // direction) and the lane mask; the block batches in the order of
  // execution with the range run after each cell batch (the last one after
  // the loop); the leftovers with their inverse diagonal; the chunks whose
  // residual is needed after the loop; and a vector with zero ghost entries
  std::vector<std::array<unsigned char, dim>> block_batch_start;
  std::vector<unsigned int>                   block_batch_fe_indices;
  std::vector<unsigned int>                   block_batch_scratch_indices;
  std::vector<unsigned char>                  block_batch_unconstrained;
  std::vector<unsigned int>                   block_batch_data_offset;
  AlignedVector<VectorizedArray<Number>>      block_batch_data;
  AlignedVector<VectorizedArray<Number>>      block_batch_mask;
  std::vector<unsigned int>                   block_batch_index;
  std::vector<unsigned int>                   block_batch_order;
  std::vector<unsigned int>                   leftover_indices;
  std::vector<Number>                         leftover_inverse_diagonal;
  std::vector<unsigned char>                  block_residual_chunks;
  mutable VectorType                          block_input;
  bool                                        block_from_scratch = true;

  template<bool from_scratch>
  void
  dispatch_block_batch(const unsigned int b,
                       const Number *     scratch_array,
                       const VectorType & src,
                       VectorType &       dst,
                       const Number       factor) const
  {
    const unsigned int degree = matrix_free.get_shape_info().data[0].fe_degree;
    if(degree == 2)
      do_block_batch<2, from_scratch>(b, scratch_array, src, dst, factor);
    else if(degree == 3)
      do_block_batch<3, from_scratch>(b, scratch_array, src, dst, factor);
    else if(degree == 4)
      do_block_batch<4, from_scratch>(b, scratch_array, src, dst, factor);
#ifndef DEBUG
    else if(degree == 5)
      do_block_batch<5, from_scratch>(b, scratch_array, src, dst, factor);
    else if(degree == 6)
      do_block_batch<6, from_scratch>(b, scratch_array, src, dst, factor);
    else if(degree == 7)
      do_block_batch<7, from_scratch>(b, scratch_array, src, dst, factor);
    else if(degree == 8)
      do_block_batch<8, from_scratch>(b, scratch_array, src, dst, factor);
#endif
    else
      AssertThrow(false, ExcMessage("Degree " + std::to_string(degree) + " not instantiated"));
  }

  // Read the cell values of a block batch with the gathered indices
  template<int degree, typename SrcVectorType>
  void
  read_block_batch(const SrcVectorType &             src,
                   const std::vector<unsigned int> & indices_all,
                   const unsigned int                b,
                   VectorizedArray<Number> *         values) const
  {
    constexpr unsigned int dofs_per_cell = Utilities::pow(degree + 1, dim);
    constexpr unsigned int n_stored      = Utilities::pow(3, dim);
    if constexpr(degree <= 2)
    {
      const unsigned int *  indices       = indices_all.data() + b * n_stored * n_lanes;
      const unsigned char * unconstrained = block_batch_unconstrained.data() + b * n_stored;
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        if(unconstrained[i])
          for(unsigned int v = 0; v < n_lanes; ++v)
            values[i][v] = src.local_element(indices[i * n_lanes + v]);
        else
        {
          values[i] = 0.;
          for(unsigned int v = 0; v < n_lanes; ++v)
            if(indices[i * n_lanes + v] != numbers::invalid_unsigned_int)
              values[i][v] = src.local_element(indices[i * n_lanes + v]);
        }
    }
    else
      read_dof_values_compressed<degree, degree + 1, 1>(
        src, indices_all, block_batch_unconstrained, b, {}, true, values);
  }

  // Run a block batch: with from_scratch, the residual is rhs (in 'src')
  // minus the result of the operator taken from the scratch array,
  // otherwise it is read from 'src'. Apply the inverse of the surrogate
  // blocks on the owned nodes and add factor times the result on the owned
  // nodes (zero elsewhere) into dst.
  template<int degree, bool from_scratch>
  void
  do_block_batch(const unsigned int b,
                 const Number *     scratch_array,
                 const VectorType & src,
                 VectorType &       dst,
                 const Number       factor) const
  {
    constexpr unsigned int  nn            = degree + 1;
    constexpr unsigned int  dofs_per_cell = Utilities::pow(nn, dim);
    constexpr unsigned int  n_stored      = Utilities::pow(3, dim);
    VectorizedArray<Number> values[dofs_per_cell];
    read_block_batch<degree>(src, block_batch_fe_indices, b, values);

    // owned sub-tensor of the residual; only the owned nodes are valid in
    // the scratch array
    const std::array<unsigned char, dim> & start = block_batch_start[b];
    std::array<unsigned int, dim>          size;
    for(unsigned int d = 0; d < dim; ++d)
      size[d] = nn - start[d];
    const auto owned_position =
      [&](const unsigned int i0, const unsigned int i1, const unsigned int i2) {
        return ((i2 + (dim == 3 ? start[dim - 1] : 0)) * nn + i1 + start[1]) * nn + i0 + start[0];
      };
    VectorizedArray<Number> owned[dofs_per_cell];
    if constexpr(from_scratch)
    {
      VectorizedArray<Number>                                  matvec[dofs_per_cell];
      typename ExaDG::ScratchMatrixVectorProduct<Number>::View view{
        const_cast<Number *>(scratch_array)};
      read_block_batch<degree>(view, block_batch_scratch_indices, b, matvec);
      for(unsigned int i2 = 0, c = 0; i2 < (dim == 3 ? size[dim - 1] : 1); ++i2)
        for(unsigned int i1 = 0; i1 < size[1]; ++i1)
          for(unsigned int i0 = 0; i0 < size[0]; ++i0, ++c)
          {
            const unsigned int p = owned_position(i0, i1, i2);
            owned[c]             = values[p] - matvec[p];
          }
    }
    else
      for(unsigned int i2 = 0, c = 0; i2 < (dim == 3 ? size[dim - 1] : 1); ++i2)
        for(unsigned int i1 = 0; i1 < size[1]; ++i1)
          for(unsigned int i0 = 0; i0 < size[0]; ++i0, ++c)
            owned[c] = values[owned_position(i0, i1, i2)];

    const VectorizedArray<Number> * data = block_batch_data.data() + block_batch_data_offset[b];
    std::array<const VectorizedArray<Number> *, dim> eigenvectors, eigenvalues;
    {
      const VectorizedArray<Number> * ptr = data;
      for(unsigned int d = 0; d < dim; ++d)
      {
        eigenvectors[d] = ptr;
        ptr += size[d] * size[d];
      }
      for(unsigned int d = 0; d < dim; ++d)
      {
        eigenvalues[d] = ptr;
        ptr += size[d];
      }
    }
    bool uniform = true;
    for(unsigned int d = 0; d < dim; ++d)
      uniform = uniform && size[d] == degree;
    if(uniform)
      dealii::internal::TensorProductMatrixSymmetricSum::apply_inverse<degree>(
        owned, owned, degree, eigenvectors, eigenvalues);
    else
      apply_inverse_anisotropic<nn>(size, eigenvectors, eigenvalues, owned);

    // scatter back with the lane mask and the factor
    const VectorizedArray<Number> scale = block_batch_mask[b] * factor;
    for(unsigned int i = 0; i < dofs_per_cell; ++i)
      values[i] = VectorizedArray<Number>();
    for(unsigned int i2 = 0, c = 0; i2 < (dim == 3 ? size[dim - 1] : 1); ++i2)
      for(unsigned int i1 = 0; i1 < size[1]; ++i1)
        for(unsigned int i0 = 0; i0 < size[0]; ++i0, ++c)
          values[owned_position(i0, i1, i2)] = owned[c] * scale;

    if constexpr(degree <= 2)
    {
      const unsigned int * indices = block_batch_fe_indices.data() + b * n_stored * n_lanes;
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        for(unsigned int v = 0; v < n_lanes; ++v)
          if(indices[i * n_lanes + v] != numbers::invalid_unsigned_int)
            dst.local_element(indices[i * n_lanes + v]) += values[i][v];
    }
    else
      distribute_local_to_global_compressed<degree, degree + 1, 1>(
        dst, block_batch_fe_indices, block_batch_unconstrained, b, {}, true, values);
  }

  // Fast diagonalization with different sizes per direction (up to
  // max_size), in place
  template<int max_size>
  static void
  apply_inverse_anisotropic(const std::array<unsigned int, dim> &                    size,
                            const std::array<const VectorizedArray<Number> *, dim> & eigenvectors,
                            const std::array<const VectorizedArray<Number> *, dim> & eigenvalues,
                            VectorizedArray<Number> *                                data)
  {
    constexpr unsigned int  max_total = Utilities::pow(max_size, dim);
    VectorizedArray<Number> tmp[max_total];
    unsigned int            n_total = 1;
    for(unsigned int d = 0; d < dim; ++d)
      n_total *= size[d];
    const auto apply_1d = [&](const unsigned int direction, const bool transpose) {
      const unsigned int              n      = size[direction];
      const VectorizedArray<Number> * matrix = eigenvectors[direction];
      unsigned int                    stride = 1;
      for(unsigned int d = 0; d < direction; ++d)
        stride *= size[d];
      const unsigned int n_outer = n_total / (stride * n);
      for(unsigned int o = 0; o < n_outer; ++o)
        for(unsigned int s = 0; s < stride; ++s)
        {
          const unsigned int base = o * stride * n + s;
          for(unsigned int i = 0; i < n; ++i)
          {
            VectorizedArray<Number> sum = VectorizedArray<Number>();
            for(unsigned int j = 0; j < n; ++j)
              sum += (transpose ? matrix[j * n + i] : matrix[i * n + j]) * data[base + j * stride];
            tmp[base + i * stride] = sum;
          }
        }
      for(unsigned int i = 0; i < n_total; ++i)
        data[i] = tmp[i];
    };
    for(unsigned int d = 0; d < dim; ++d)
      apply_1d(d, true);
    for(unsigned int i2 = 0, c = 0; i2 < (dim == 3 ? size[2] : 1); ++i2)
      for(unsigned int i1 = 0; i1 < size[1]; ++i1)
        for(unsigned int i0 = 0; i0 < size[0]; ++i0, ++c)
        {
          VectorizedArray<Number> sum = eigenvalues[0][i0] + eigenvalues[1][i1];
          if constexpr(dim == 3)
            sum += eigenvalues[2][i2];
          data[c] /= sum;
        }
    for(unsigned int d = dim; d-- > 0;)
      apply_1d(d, false);
  }

  ObserverPointer<const DoFHandler<dim>> dof_handler;
  MatrixFree<dim, Number>                matrix_free;
  // node of each entry of the 3^dim entities, starting at
  // scalar_entity_start[i] for entity i, see setup_scalar_read()
  // timings of vmult_scratch() and vmult(), see print_timings(); the three
  // entries from t_compress_send on are filled by
  // ScratchMatrixVectorProduct::finish_loop() and must stay consecutive
  enum TimingIndex : unsigned int
  {
    t_scratch_count,
    t_before_loop,
    t_ghost_start,
    t_ghost_wait,
    t_cell_loop,
    t_cells_boundary,
    t_cells_interior_2,
    t_compress_send,
    t_compress_wait,
    t_finish_ranges,
    t_scratch_total,
    t_vmult_count,
    t_vmult_total,
    t_block_interface,
    n_timings
  };
  static_assert(t_compress_wait == t_compress_send + 1 && t_finish_ranges == t_compress_send + 2);
  mutable std::array<double, n_timings> timings = {};

  // order of the cell batches in vmult_scratch() (position -> cell batch)
  // and the positions after which the ghost update is finished and the
  // compress started, see setup_loop_order(); interval of MPI_Iprobe calls
  std::vector<unsigned int> loop_order;
  unsigned int              n_before_ghosts       = 0;
  unsigned int              n_before_compress     = 0;
  unsigned int              mpi_progress_interval = 16;
  double                    overlap_split         = 0.5;

  // shared-memory exchange, see set_shared_memory(): the communicator of
  // the node for the allocation of the vectors, duplicates of the
  // communicators for the messages, and the exchange object with its
  // buffers; empty without shared memory. The compress uses channel 2, see
  // ScratchMatrixVectorProduct.
  static constexpr unsigned int ghost_channel           = 0;
  bool                          shared_memory           = false;
  unsigned int                  emulated_ranks_per_node = 0;
  unsigned int                  n_sm_processes          = 1;
  std::shared_ptr<MPI_Comm>     comm_sm;
  std::shared_ptr<MPI_Comm>     comm_exchange;
  std::shared_ptr<MPI_Comm>     comm_exchange_sm;
  std::shared_ptr<const internal::MatrixFreeFunctions::VectorDataExchange::Full> sm_exchange;
  mutable AlignedVector<Number>                                                  ghost_temporary;
  mutable std::vector<MPI_Request>                                               ghost_requests;

  mutable std::vector<unsigned int> scalar_entity_start;
  mutable std::vector<unsigned int> scalar_entity_node;

  std::vector<unsigned int>  compressed_dof_indices;
  std::vector<unsigned char> all_indices_unconstrained;

  // hanging-node data: the constraint kind of all lanes of a cell batch and
  // a flag whether a batch contains any hanging node, both empty if the mesh
  // does not have hanging nodes, as well as the positions in
  // compressed_dof_indices that have been replaced by the indices of the
  // coarse side
  std::vector<std::array<internal::MatrixFreeFunctions::compressed_constraint_kind, n_lanes>>
                             hanging_node_masks;
  std::vector<unsigned char> batch_has_hanging_nodes;
  std::vector<unsigned int>  positions_replaced_by_hanging_nodes;

  // scratch array of vmult_scratch() and the indices into it, in the
  // format of compressed_dof_indices
  ExaDG::ScratchMatrixVectorProduct<Number> scratch;
  std::vector<unsigned int>                 scratch_dof_indices;

  // ranges of entries for 'before_loop' in vmult_scratch() by cell batch,
  // with the entries before the ghost exchange at index n_cell_batches,
  // empty if the process has fewer than 'min_size_for_pre_list' locally
  // owned entries, see setup_pre_list()
  static constexpr unsigned int                      pre_list_chunk_size   = 64;
  unsigned int                                       min_size_for_pre_list = 100000;
  std::vector<unsigned int>                          cell_loop_pre_list_index;
  std::vector<std::pair<unsigned int, unsigned int>> cell_loop_pre_list;

  std::vector<dealii::ndarray<unsigned int, n_lanes, 2>> cell_level_index;

  Extruded::MappingInfo<dim, Number> mapping_info;

  Number factor_laplace;
  Number factor_mass;
};



template<int dim, typename Number = double>
class LaplaceOperatorDG : public EnableObserverPointer
{
public:
  static constexpr unsigned int dimension = dim;
  static constexpr unsigned int n_lanes   = VectorizedArray<Number>::size();
  using VectorType                        = LinearAlgebra::distributed::Vector<Number>;

  LaplaceOperatorDG() = default;

  template<typename OtherNumber>
  void
  reinit(const Mapping<dim> &                   mapping,
         const DoFHandler<dim> &                dof_handler,
         const AffineConstraints<OtherNumber> & constraints,
         const std::vector<unsigned int> &      cell_vectorization_category,
         const Quadrature<1> &                  quadrature,
         const bool                             is_test,
         const MPI_Comm                         communicator_shared = MPI_COMM_SELF)
  {
    (void)communicator_shared; // TODO: not yet implemented
    this->is_test                                       = is_test;
    this->dof_handler                                   = &dof_handler;
    const FiniteElement<dim> &                       fe = dof_handler.get_fe();
    typename MatrixFree<dim, Number>::AdditionalData mf_data;
    mf_data.cell_vectorization_category          = cell_vectorization_category;
    mf_data.cell_vectorization_categories_strict = false;
    mf_data.overlap_communication_computation    = false;
    mf_data.initialize_mapping                   = false;
    matrix_free.reinit(MappingQ1<dim>(), dof_handler, constraints, quadrature, mf_data);
    dof_indices.resize(matrix_free.n_cell_batches());
    for(unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
      for(unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(cell); ++v)
        dof_indices[cell][v] =
          matrix_free.get_dof_info()
            .dof_indices_contiguous[internal::MatrixFreeFunctions::DoFInfo::dof_access_cell]
                                   [cell * n_lanes + v];
    partitioner = matrix_free.get_dof_info().vector_partitioner;

    {
      ndarray<unsigned int, 2 * dim, n_lanes> default_argument;
      for(unsigned int i = 0; i < 2 * dim; ++i)
        for(unsigned int j = 0; j < n_lanes; ++j)
          default_argument[i][j] = numbers::invalid_unsigned_int;
      neighbor_cells.resize(matrix_free.n_cell_batches(), default_argument);
      mpi_exchange_data_on_faces.resize(matrix_free.n_cell_batches(), default_argument);
    }
    {
      IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();

      std::vector<unsigned int> cell_indices(dof_handler.get_triangulation().n_active_cells(),
                                             numbers::invalid_unsigned_int);
      for(unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
        for(unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(cell); ++v)
          cell_indices[matrix_free.get_cell_iterator(cell, v)->active_cell_index()] =
            cell * VectorizedArray<Number>::size() + v;

      std::map<unsigned int, std::vector<std::array<types::global_dof_index, 5>>> proc_neighbors;
      for(unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
        for(unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(cell); ++v)
        {
          const auto dcell = matrix_free.get_cell_iterator(cell, v);
          for(unsigned int f = 0; f < 2 * dim; ++f)
          {
            const bool at_boundary           = dcell->at_boundary(f);
            const bool has_periodic_neighbor = at_boundary && dcell->has_periodic_neighbor(f);
            if(at_boundary == false || has_periodic_neighbor)
            {
              const auto neighbor =
                has_periodic_neighbor ? dcell->periodic_neighbor(f) : dcell->neighbor(f);
              if(neighbor->is_locally_owned())
              {
                AssertIndexRange(neighbor->active_cell_index(), cell_indices.size());
                neighbor_cells[cell][f][v] = cell_indices[neighbor->active_cell_index()];
              }
              else
              {
                std::array<types::global_dof_index, 5> neighbor_data;
                neighbor_data[0] = cell * n_lanes + v;
                neighbor_data[1] = has_periodic_neighbor ? dcell->periodic_neighbor_face_no(f) :
                                                           dcell->neighbor_face_no(f);
                neighbor_data[2] = dcell->global_active_cell_index();
                neighbor_data[3] = neighbor->global_active_cell_index();
                neighbor_data[4] = f;
                proc_neighbors[neighbor->subdomain_id()].push_back(neighbor_data);
                // set dummy
                neighbor_cells[cell][f][v] = cell_indices[dcell->active_cell_index()];
              }
            }
          }
        }

      const unsigned int nn = fe.degree + 1;
      // data: projected to face, 2 because of values and derivatives
      const unsigned int data_per_face = 2 * Utilities::pow(nn, dim - 1);

      // find out how the neighbor wants us to send the data -> we sort by the
      // global cell index. We use a reverse order of processes as the data
      // access might favor nearby access to cell data.
      send_data_process.clear();
      send_data_cell_index.clear();
      send_data_face_index.clear();
      {
        std::size_t sizes = 0;
        for(const auto & it : proc_neighbors)
          sizes += it.second.size();
        send_data_cell_index.reserve(sizes);
        send_data_face_index.reserve(sizes);
      }
      unsigned int offset = 0;
      for(auto it = proc_neighbors.rbegin(); it != proc_neighbors.rend(); ++it)
      {
        std::sort(it->second.begin(),
                  it->second.end(),
                  [](const std::array<types::global_dof_index, 5> & a,
                     const std::array<types::global_dof_index, 5> & b) {
                    if(a[4] < b[4])
                      return true;
                    else if(a[4] == b[4] && a[3] < b[3])
                      return true;
                    else
                      return false;
                  });
        send_data_process.emplace_back(it->first, it->second.size());
        for(unsigned int i = 0; i < it->second.size(); ++i)
        {
          send_data_cell_index.emplace_back(it->second[i][0]);
          send_data_face_index.emplace_back(it->second[i][4]);
        }

        // finally figure out where to read the imported face data from,
        // which is done by sorting in the way the neighboring process would
        // have run the sorting we just did a few lines up
        std::sort(it->second.begin(),
                  it->second.end(),
                  [](const std::array<types::global_dof_index, 5> & a,
                     const std::array<types::global_dof_index, 5> & b) {
                    if(a[1] < b[1])
                      return true;
                    else if(a[1] == b[1] && a[2] < b[2])
                      return true;
                    else
                      return false;
                  });

        for(unsigned int i = 0; i < it->second.size(); ++i)
        {
          const unsigned int cell                               = it->second[i][0] / n_lanes;
          const unsigned int v                                  = it->second[i][0] % n_lanes;
          mpi_exchange_data_on_faces[cell][it->second[i][4]][v] = offset;

          offset += data_per_face;
        }
      }
      import_values.clear();
      import_values.resize_fast(send_data_cell_index.size() * data_per_face);
      export_values.clear();
      export_values.resize_fast(send_data_cell_index.size() * data_per_face);
      mpi_requests.resize(2 * send_data_process.size());
    }

    cell_level_index.resize(matrix_free.n_cell_batches());
    AssertDimension(cell_level_index.size(), dof_indices.size());
    for(unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
      for(unsigned int lane = 0; lane < n_lanes; ++lane)
        if(lane >= matrix_free.n_active_entries_per_cell_batch(cell))
          cell_level_index[cell][lane] = {
            {numbers::invalid_unsigned_int, numbers::invalid_unsigned_int}};
        else
        {
          const auto iter                 = matrix_free.get_cell_iterator(cell, lane);
          cell_level_index[cell][lane][0] = iter->level();
          cell_level_index[cell][lane][1] = iter->index();
        }

    mapping_info.reinit(mapping, quadrature, dof_handler.get_triangulation(), cell_level_index);
    AssertThrow(mapping_info.has_uniform_h_z,
                ExcNotImplemented("The DG Laplace operator requires all cells to have the "
                                  "same extent in z direction"));
    AssertThrow(!dof_handler.get_triangulation().has_hanging_nodes(),
                ExcNotImplemented("The DG Laplace operator does not support hanging nodes"));

    compute_vector_access_pattern();

    {
      std::vector<Polynomials::Polynomial<double>> basis =
        Polynomials::generate_complete_Lagrange_basis(quadrature.get_points());
      interpolate_quad_to_boundary[0].resize(quadrature.size());
      interpolate_quad_to_boundary[1].resize(quadrature.size());
      std::vector<double> val_and_der(2);
      for(unsigned int i = 0; i < basis.size(); ++i)
      {
        basis[i].value(0., val_and_der);
        interpolate_quad_to_boundary[0][i][0] = val_and_der[0];
        interpolate_quad_to_boundary[0][i][1] = val_and_der[1];
        basis[i].value(1., val_and_der);
        interpolate_quad_to_boundary[1][i][0] = val_and_der[0];
        interpolate_quad_to_boundary[1][i][1] = val_and_der[1];
      }
    }

    shape_info = matrix_free.get_shape_info();

    detect_dependencies_of_face_integrals();

    setup_scratch();

    set_parameters(0., 1.0);

    for(double & t : timings)
      t = 0.0;
  }

  ~LaplaceOperatorDG()
  {
    if(not this->is_test and timings[0] > 0)
    {
      const MPI_Comm comm = dof_handler->get_mpi_communicator();
      std::cout << std::defaultfloat << std::setprecision(3);
      const double total_time =
        Utilities::MPI::sum(timings[9], comm) / timings[0] / Utilities::MPI::n_mpi_processes(comm);
      if(Utilities::MPI::this_mpi_process(comm) == 0)
        std::cout << "Collected timings for DG Laplace operator <"
                  << (std::is_same_v<Number, double> ? "double" : "float") << "> in "
                  << static_cast<unsigned long>(timings[0])
                  << " evaluations [t_total=" << total_time * timings[0] << "s]" << std::endl;
      if(Utilities::MPI::n_mpi_processes(comm) > 1)
      {
        Helper::print_time(timings[3] / timings[0], "Pack/send data dg ghosts", comm, total_time);
        Helper::print_time(timings[5] / timings[0], "MPI_Waitall dg ghosts", comm, total_time);
        Helper::print_time(timings[4] / timings[0], "Pre-loop before ghosts", comm, total_time);
      }

      Helper::print_time(timings[7] / timings[0], "Matrix-free loop", comm, total_time);
      Helper::print_time_per_node(timings[7] / timings[0],
                                  "Matrix-free loop, node avg",
                                  comm,
                                  total_time);
      if(Utilities::MPI::this_mpi_process(comm) == 0)
        std::cout << std::endl;
    }
    if(not this->is_test and timings[10] > 0)
    {
      const MPI_Comm comm       = MPI_COMM_WORLD;
      const double   total_time = Utilities::MPI::sum(timings[14], comm) / timings[10] /
                                Utilities::MPI::n_mpi_processes(comm);
      if(Utilities::MPI::this_mpi_process(comm) == 0)
        std::cout << "Collected timings for DG mass operator <"
                  << (std::is_same_v<Number, double> ? "double" : "float") << "> in "
                  << static_cast<unsigned long>(timings[10])
                  << " evaluations [t_total=" << total_time * timings[10] << "s]" << std::endl;

      Helper::print_time(timings[12] / timings[10], "Matrix-free loop", comm, total_time);
      if(Utilities::MPI::this_mpi_process(comm) == 0)
        std::cout << std::endl;
    }
  }

  const DoFHandler<dim> &
  get_dof_handler() const
  {
    return *dof_handler;
  }

  const MatrixFree<dim, Number> &
  get_matrix_free() const
  {
    return matrix_free;
  }

  void
  initialize_dof_vector(VectorType & vec) const
  {
    vec.reinit(partitioner);
  }

  void
  set_parameters(const double factor_mass, const double factor_laplace)
  {
    this->factor_mass    = factor_mass;
    this->factor_laplace = factor_laplace;
  }

  void
  vmult(VectorType & dst, const VectorType & src) const
  {
    vmult(
      dst,
      src,
      [&dst](const unsigned int start, const unsigned int end) {
        std::memset(dst.begin() + start,
                    0,
                    (end - start) * sizeof(typename VectorType::value_type));
      },
      [](const unsigned int, const unsigned int) {});
  }

  void
  compute_inverse_diagonal(VectorType & dst) const
  {
    initialize_dof_vector(dst);
    const unsigned int n_cell_batches = dof_indices.size();
    for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
      diagonal_operation(cell, dst);
    for(Number & a : dst)
    {
      AssertThrow(a > 0,
                  ExcInternalError("Error DG degree " +
                                   std::to_string(dof_handler->get_fe().degree) + " size " +
                                   std::to_string(dof_handler->n_dofs())));
      a = Number(1.0) / a;
    }
  }

  // --- Block-Jacobi preconditioner with fast diagonalization ---
  //
  // The block of the operator on a cell, i.e., the cell integral and the
  // face integrals with zero exterior values on interior faces, is replaced
  // by the block of a Cartesian surrogate cell, whose inverse is applied by
  // the fast diagonalization method (Lynch, Rice, Thomas 1964), see also
  // deal.II's step-59 and TensorProductMatrixSymmetricSum. The surrogate has
  // the extents h_0, h_1 given by the average length of the columns of the
  // Jacobian of the deformed xy geometry over the quadrature points, and h_z
  // in z direction. Per direction d, the 1D matrices are the DG mass matrix
  // M_d = h_d M^ and the interior penalty matrix L_d with the stiffness
  // factor_laplace/h_d K^ plus, on the two faces of the direction, the
  // penalty and consistency terms with the penalty parameters of the actual
  // operator and the exterior value zero. Boundary faces, where the operator
  // applies the mirror principle, have no face terms. The block is
  //   A_K = factor_mass M_2 x M_1 x M_0 + sum_d (prod_{e != d} M_e) x L_d,
  // whose inverse follows from the generalized eigenvalue problems
  // L_d S_d = M_d S_d Lambda_d as
  //   A_K^{-1} = (S_2 x S_1 x S_0) (factor_mass + Lambda_0 + Lambda_1 +
  //              Lambda_2)^{-1} (S_2 x S_1 x S_0)^T.
  // The eigenvectors and eigenvalues are stored per cell batch.
  //
  // Must be called after set_penalty_parameters() and set_parameters().
  void
  setup_block_inverse()
  {
    const unsigned int degree = shape_info.data[0].fe_degree;
    if(degree == 1)
      do_setup_block_inverse<1>();
    else if(degree == 2)
      do_setup_block_inverse<2>();
    else if(degree == 3)
      do_setup_block_inverse<3>();
    else if(degree == 4)
      do_setup_block_inverse<4>();
#ifndef DEBUG
    else if(degree == 5)
      do_setup_block_inverse<5>();
    else if(degree == 6)
      do_setup_block_inverse<6>();
    else if(degree == 7)
      do_setup_block_inverse<7>();
    else if(degree == 8)
      do_setup_block_inverse<8>();
#endif
    else
      AssertThrow(false, ExcMessage("Degree " + std::to_string(degree) + " not instantiated"));
  }

  bool
  has_block_inverse() const
  {
    return !block_eigenvalues.empty();
  }

  unsigned int
  n_cell_batches() const
  {
    return dof_indices.size();
  }

  unsigned int
  n_dofs_per_cell() const
  {
    return Utilities::pow(shape_info.data[0].fe_degree + 1, dim);
  }

  unsigned int
  n_active_lanes(const unsigned int cell) const
  {
    return n_active_entries_per_cell_batch(cell);
  }

  // Read the entries of the cells of a batch from 'vector' into 'values'
  // in the layout of VectorizedArray, with the lexicographic numbering
  // within the cells. Lanes without a cell read the entries of the first
  // lane.
  void
  read_cell_dofs(const unsigned int        cell,
                 const Number *            vector,
                 VectorizedArray<Number> * values) const
  {
    const unsigned int n_active = n_active_entries_per_cell_batch(cell);
    if(n_active == n_lanes)
      vectorized_load_and_transpose(n_dofs_per_cell(), vector, dof_indices[cell].data(), values);
    else
    {
      std::array<unsigned int, n_lanes> indices;
      for(unsigned int v = 0; v < n_lanes; ++v)
        indices[v] = dof_indices[cell][v < n_active ? v : 0];
      vectorized_load_and_transpose(n_dofs_per_cell(), vector, indices.data(), values);
    }
  }

  // Write the entries of the active lanes of 'values' to the cells of a
  // batch in 'vector', overwriting the previous content
  void
  write_cell_dofs(const unsigned int              cell,
                  const VectorizedArray<Number> * values,
                  Number *                        vector) const
  {
    const unsigned int n_active      = n_active_entries_per_cell_batch(cell);
    const unsigned int dofs_per_cell = n_dofs_per_cell();
    if(n_active == n_lanes)
      vectorized_transpose_and_store(
        false, dofs_per_cell, values, dof_indices[cell].data(), vector);
    else
      for(unsigned int v = 0; v < n_active; ++v)
      {
        Number * dst = vector + dof_indices[cell][v];
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          dst[i] = values[i][v];
      }
  }

  // Number of Richardson steps with the actual cell block that correct the
  // inverse of the surrogate block in apply_block_inverse_cell(), see
  // do_apply_block_inverse_cell()
  void
  set_block_inverse_iterations(const unsigned int n_iterations)
  {
    block_inverse_iterations = n_iterations;
  }

  // Apply the inverse of the surrogate block on the cell batch 'cell' to
  // the cell values in 'values' (layout as in read_cell_dofs()), in place,
  // corrected by the number of Richardson steps given by
  // set_block_inverse_iterations() or the argument, if given
  void
  apply_block_inverse_cell(const unsigned int        cell,
                           VectorizedArray<Number> * values,
                           const unsigned int n_iterations_in = numbers::invalid_unsigned_int) const
  {
    Assert(has_block_inverse(), ExcMessage("Call setup_block_inverse() first"));
    const unsigned int n_iterations =
      n_iterations_in == numbers::invalid_unsigned_int ? block_inverse_iterations : n_iterations_in;
    const unsigned int degree = shape_info.data[0].fe_degree;
    if(degree == 1)
      do_apply_block_inverse_cell<1>(cell, values, n_iterations);
    else if(degree == 2)
      do_apply_block_inverse_cell<2>(cell, values, n_iterations);
    else if(degree == 3)
      do_apply_block_inverse_cell<3>(cell, values, n_iterations);
    else if(degree == 4)
      do_apply_block_inverse_cell<4>(cell, values, n_iterations);
#ifndef DEBUG
    else if(degree == 5)
      do_apply_block_inverse_cell<5>(cell, values, n_iterations);
    else if(degree == 6)
      do_apply_block_inverse_cell<6>(cell, values, n_iterations);
    else if(degree == 7)
      do_apply_block_inverse_cell<7>(cell, values, n_iterations);
    else if(degree == 8)
      do_apply_block_inverse_cell<8>(cell, values, n_iterations);
#endif
    else
      AssertThrow(false, ExcMessage("Degree " + std::to_string(degree) + " not instantiated"));
  }

  // Apply the block diagonal of the operator, dst = A_K src on all cells
  void
  apply_block_diagonal(VectorType & dst, const VectorType & src) const
  {
    const unsigned int                     n_dofs = n_dofs_per_cell();
    AlignedVector<VectorizedArray<Number>> in(n_dofs), out(n_dofs);
    const unsigned int                     degree = shape_info.data[0].fe_degree;
    for(unsigned int cell = 0; cell < n_cell_batches(); ++cell)
    {
      read_cell_dofs(cell, src.begin(), in.data());
      if(degree == 1)
        do_apply_cell_block<1>(cell, in.data(), out.data());
      else if(degree == 2)
        do_apply_cell_block<2>(cell, in.data(), out.data());
      else if(degree == 3)
        do_apply_cell_block<3>(cell, in.data(), out.data());
      else if(degree == 4)
        do_apply_cell_block<4>(cell, in.data(), out.data());
#ifndef DEBUG
      else if(degree == 5)
        do_apply_cell_block<5>(cell, in.data(), out.data());
      else if(degree == 6)
        do_apply_cell_block<6>(cell, in.data(), out.data());
      else if(degree == 7)
        do_apply_cell_block<7>(cell, in.data(), out.data());
      else if(degree == 8)
        do_apply_cell_block<8>(cell, in.data(), out.data());
#endif
      else
        AssertThrow(false, ExcMessage("Degree " + std::to_string(degree) + " not instantiated"));
      write_cell_dofs(cell, out.data(), dst.begin());
    }
  }

  // Quality of the approximate block inverse B with the given number of
  // Richardson steps: |r - A_K B r| / |r| for a random vector r, i.e., how
  // far B is from the exact block-Jacobi preconditioner
  double
  compute_block_inverse_defect(const unsigned int n_iterations) const
  {
    VectorType r, z, az;
    initialize_dof_vector(r);
    initialize_dof_vector(z);
    initialize_dof_vector(az);
    std::mt19937                           gen(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));
    std::uniform_real_distribution<Number> dist(-1., 1.);
    for(Number & a : r)
      a = dist(gen);
    AlignedVector<VectorizedArray<Number>> values(n_dofs_per_cell());
    for(unsigned int cell = 0; cell < n_cell_batches(); ++cell)
    {
      read_cell_dofs(cell, r.begin(), values.data());
      apply_block_inverse_cell(cell, values.data(), n_iterations);
      write_cell_dofs(cell, values.data(), z.begin());
    }
    apply_block_diagonal(az, z);
    az -= r;
    return az.l2_norm() / r.l2_norm();
  }

  // Apply the block-Jacobi preconditioner dst = P^{-1} src on all cells
  void
  apply_block_inverse(VectorType & dst, const VectorType & src) const
  {
    AlignedVector<VectorizedArray<Number>> values(n_dofs_per_cell());
    for(unsigned int cell = 0; cell < n_cell_batches(); ++cell)
    {
      read_cell_dofs(cell, src.begin(), values.data());
      apply_block_inverse_cell(cell, values.data());
      write_cell_dofs(cell, values.data(), dst.begin());
    }
  }

  // Matrix-vector product that hands the result of each cell batch, which
  // is complete after the batch as each cell only writes into its own
  // entries, to 'cell_function(cell, values)' in the layout of
  // read_cell_dofs() instead of writing it into a vector. The array
  // 'values' can be modified by the function.
  template<typename CellFunction>
  void
  vmult_cellwise(const VectorType & src, const CellFunction & cell_function) const
  {
    AssertThrow(factor_laplace != 0., ExcNotImplemented());
    Timer total_timer;
    timings[0] += 1;
    exchange_face_data(src);

    Timer time;
    for(unsigned int cell = 0; cell < n_cell_batches(); ++cell)
      dispatch_cell_operation_cellwise(cell, src, cell_function);
    timings[7] += time.wall_time();
    timings[8] += time.wall_time();
    timings[9] += total_timer.wall_time();
  }

  // Settings of the prolongation within the DG smoother of
  // vmult_scratch_with_prolongation(), to be set before its first call: per
  // cell batch and lane instead of batches of arbitrary cells ordered by the
  // consumers, and the look-ahead in cell batches when filling those batches
  void
  set_prolongation_settings(const bool lane_wise, const unsigned int lookahead)
  {
    prolongation_lane_wise = lane_wise;
    prolongation_lookahead = lookahead;
  }

  void
  set_penalty_parameters(const Number additional_factor)
  {
    // Same as in ExaDG::IP::get_penalty_factor<dim> but split off to test
    // this function independently of ExaDG
    const Number factor =
      additional_factor * Utilities::fixed_power<2>(dof_handler->get_fe().degree + 1.0);
    this->penalty_parameters.resize(dof_indices.size());
    const unsigned int n_q_points_1d = shape_info.data[0].n_q_points_1d;
    for(unsigned int cell = 0; cell < dof_indices.size(); ++cell)
      for(unsigned int v = 0; v < n_lanes; ++v)
      {
        for(unsigned int face = 0; face < 4; ++face)
        {
          const unsigned int idx =
            mapping_info.face_mapping_data_index[cell][face][0][v] / n_q_points_1d / 4;
          AssertThrow(idx < mapping_info.ip_penalty_factors.size(),
                      ExcIndexRange(idx, 0, mapping_info.ip_penalty_factors.size()));
          const unsigned int neigh_idx =
            mapping_info.face_mapping_data_index[cell][face][1][v] / n_q_points_1d / 4;
          AssertThrow(neigh_idx < mapping_info.ip_penalty_factors.size(),
                      ExcIndexRange(neigh_idx, 0, mapping_info.ip_penalty_factors.size()));
          this->penalty_parameters[cell][face][v] =
            std::max(mapping_info.ip_penalty_factors[idx],
                     mapping_info.ip_penalty_factors[neigh_idx]) *
            factor;
        }
        for(unsigned int face = 4; face < 2 * dim; ++face)
          this->penalty_parameters[cell][face][v] =
            mapping_info.ip_penalty_factors[mapping_info.mapping_data_index[cell][v] /
                                            Utilities::pow(n_q_points_1d, 2)] *
            factor;
      }
  }

  types::global_dof_index
  m() const
  {
    return partitioner->size();
  }

  Number
  el(const types::global_dof_index, const types::global_dof_index) const
  {
    AssertThrow(false, ExcNotImplemented());
    return 0;
  }

  void
  vmult(VectorType &                                                        dst,
        const VectorType &                                                  src,
        const std::function<void(const unsigned int, const unsigned int)> & before_loop,
        const std::function<void(const unsigned int, const unsigned int)> & after_loop) const
  {
    Timer total_timer;
    if(factor_laplace != 0.)
      timings[0] += 1;
    else
      timings[10] += 1;
    Timer time;

    const unsigned int n_cell_batches = dof_indices.size();

    if(factor_laplace != 0.)
    {
      for(unsigned int range = cell_loop_pre_list_index[n_cell_batches];
          range < cell_loop_pre_list_index[n_cell_batches + 1];
          ++range)
        before_loop(cell_loop_pre_list[range].first, cell_loop_pre_list[range].second);

      timings[4] += time.wall_time();
    }

    exchange_face_data(src);

    time.restart();
    for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
    {
      if(factor_laplace != 0.)
        for(unsigned int range = cell_loop_pre_list_index[cell];
            range < cell_loop_pre_list_index[cell + 1];
            ++range)
          before_loop(cell_loop_pre_list[range].first, cell_loop_pre_list[range].second);
      else
        for(unsigned int range = cell_loop_mass_pre_list_index[cell];
            range < cell_loop_mass_pre_list_index[cell + 1];
            ++range)
          before_loop(cell_loop_mass_pre_list[range].first, cell_loop_mass_pre_list[range].second);

      const unsigned int degree = shape_info.data[0].fe_degree;
      if(degree == 1)
        do_cell_operation<1, true>(cell, src, dst);
      else if(degree == 2)
        do_cell_operation<2, true>(cell, src, dst);
      else if(degree == 3)
        do_cell_operation<3, true>(cell, src, dst);
      else if(degree == 4)
        do_cell_operation<4, true>(cell, src, dst);
#ifndef DEBUG
      else if(degree == 5)
        do_cell_operation<5, true>(cell, src, dst);
      else if(degree == 6)
        do_cell_operation<6, true>(cell, src, dst);
      else if(degree == 7)
        do_cell_operation<7, true>(cell, src, dst);
      else if(degree == 8)
        do_cell_operation<8, true>(cell, src, dst);
#endif
      else
        AssertThrow(false, ExcMessage("Degree " + std::to_string(degree) + " not instantiated"));
      for(unsigned int range = cell_loop_post_list_index[cell];
          range < cell_loop_post_list_index[cell + 1];
          ++range)
        after_loop(cell_loop_post_list[range].first, cell_loop_post_list[range].second);
    }

    if(factor_laplace != 0)
      timings[7] += time.wall_time();
    else
      timings[12] += time.wall_time();

    if(factor_laplace != 0)
    {
      timings[8] += time.wall_time();
      timings[9] += total_timer.wall_time();
    }
    else
    {
      timings[13] += time.wall_time();
      timings[14] += total_timer.wall_time();
    }
  }

  // Variant of the matrix-vector product that accumulates the result into
  // a small scratch array instead of a global vector, see
  // ExaDG::ScratchMatrixVectorProduct for the format of the ranges passed
  // to 'after_loop'. As each cell only writes into its own entries, the
  // result of a cell batch is passed on right after the cell batch.
  template<typename AfterLoop>
  void
  vmult_scratch(const VectorType & src, const AfterLoop & after_loop) const
  {
    vmult_scratch(src, nullptr, after_loop);
  }

  // Variant of vmult_scratch() that also runs 'before_loop(begin, end)' on
  // the locally owned entries of src, like vmult(), before their first read
  // in the cell integrals or the data sent to other processes. Each entry
  // passes through 'before_loop' exactly once.
  template<typename BeforeLoop, typename AfterLoop>
  void
  vmult_scratch(const VectorType & src,
                const BeforeLoop & before_loop,
                const AfterLoop &  after_loop) const
  {
    Timer total_timer;
    if(factor_laplace != 0.)
      timings[0] += 1;
    else
      timings[10] += 1;

    constexpr bool     has_before_loop = !std::is_same_v<BeforeLoop, std::nullptr_t>;
    const unsigned int n_cell_batches  = dof_indices.size();
    const auto &       pre_list_index =
      factor_laplace != 0. ? cell_loop_pre_list_index : cell_loop_mass_pre_list_index;
    const auto & pre_list     = factor_laplace != 0. ? cell_loop_pre_list : cell_loop_mass_pre_list;
    const auto   run_pre_list = [&](const unsigned int index) {
      if constexpr(has_before_loop)
        for(unsigned int range = pre_list_index[index]; range < pre_list_index[index + 1]; ++range)
          before_loop(pre_list[range].first, pre_list[range].second);
    };

    // entries of the cells whose face data is sent to other processes
    if constexpr(has_before_loop)
      if(factor_laplace != 0.)
      {
        Timer time;
        run_pre_list(n_cell_batches);
        timings[4] += time.wall_time();
      }

    scratch.prepare_loop();
    exchange_face_data(src);

    Timer time;
    scratch.cell_loop(
      [&](const unsigned int cell, Number * scratch_array) {
        run_pre_list(cell);
        dispatch_cell_operation(cell, src, scratch_indices[cell], scratch_array);
      },
      after_loop);
    scratch.finish_loop(after_loop, {});

    if(factor_laplace != 0)
    {
      timings[7] += time.wall_time();
      timings[8] += time.wall_time();
      timings[9] += total_timer.wall_time();
    }
    else
    {
      timings[12] += time.wall_time();
      timings[13] += time.wall_time();
      timings[14] += total_timer.wall_time();
    }
  }

  // Variant of vmult_scratch() that first adds the prolongation of the FE
  // vector fe_correction to src, like prolongate_and_add(), within the
  // loop: the cell batches whose face data is sent to other processes are
  // prolongated before the exchange, all other cell batches right before
  // the first cell batch reading them, i.e., the batch itself or one with
  // a face neighbor in it. The result is the same as prolongate_and_add()
  // followed by vmult_scratch().
  template<typename AfterLoop>
  void
  vmult_scratch_with_prolongation(VectorType &                           src,
                                  const VectorType &                     fe_correction,
                                  const LaplaceOperatorFE<dim, Number> & fe_operator,
                                  const AfterLoop &                      after_loop) const
  {
    do_vmult_with_prolongation<false>(src, fe_correction, fe_operator, after_loop);
  }

  // Variant of vmult_scratch_with_prolongation() that hands the result of
  // each cell batch to 'cell_function' like vmult_cellwise(): the result is
  // the same as prolongate_and_add() followed by vmult_cellwise(). As each
  // cell is prolongated before the first cell batch reading it, which is at
  // the latest the cell batch itself, 'cell_function' sees the updated
  // entries of src on the cells of the batch.
  template<typename CellFunction>
  void
  vmult_cellwise_with_prolongation(VectorType &                           src,
                                   const VectorType &                     fe_correction,
                                   const LaplaceOperatorFE<dim, Number> & fe_operator,
                                   const CellFunction &                   cell_function) const
  {
    do_vmult_with_prolongation<true>(src, fe_correction, fe_operator, cell_function);
  }

  // Implementation of vmult_scratch_with_prolongation() (cellwise = false,
  // 'output' is 'after_loop') and vmult_cellwise_with_prolongation()
  // (cellwise = true, 'output' is 'cell_function')
  template<bool cellwise, typename Output>
  void
  do_vmult_with_prolongation(VectorType &                           src,
                             const VectorType &                     fe_correction,
                             const LaplaceOperatorFE<dim, Number> & fe_operator,
                             const Output &                         output) const
  {
    AssertThrow(factor_laplace != 0., ExcNotImplemented());
    const unsigned int n_cell_batches = dof_indices.size();
    if(!prolongation_setup_done)
    {
      prolongation_setup_done = true;
      // batches of arbitrary cells ordered by the consumers, unless the FE
      // level has hanging nodes, where the lane-wise prolongation of the
      // cell batches is used
      use_prolongation_batches =
        Utilities::MPI::max(fe_operator.has_hanging_nodes() ? 1 : 0, MPI_COMM_WORLD) == 0 &&
        !prolongation_lane_wise;
      if(use_prolongation_batches)
        setup_prolongation_batches(fe_operator);
      else
        setup_prolongation_lists();
    }
    if(use_prolongation_batches)
    {
      vmult_with_prolongation_batches<cellwise>(src, fe_correction, output);
      return;
    }

    Timer total_timer;
    timings[0] += 1;
    const unsigned int degree                = shape_info.data[0].fe_degree;
    const auto         run_prolongation_list = [&](const unsigned int index) {
      for(unsigned int i = prolongation_list_index[index]; i < prolongation_list_index[index + 1];
          ++i)
        dispatch_prolongate(degree,
                            prolongation_list[i].first,
                            prolongation_list[i].second,
                            fe_correction,
                            fe_operator,
                            src);
    };

    fe_correction.update_ghost_values();
    {
      Timer time;
      run_prolongation_list(n_cell_batches);
      timings[4] += time.wall_time();
    }

    if constexpr(!cellwise)
      scratch.prepare_loop();
    exchange_face_data(src);

    Timer time;
    if constexpr(cellwise)
      for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
      {
        run_prolongation_list(cell);
        dispatch_cell_operation_cellwise(cell, src, output);
      }
    else
    {
      scratch.cell_loop(
        [&](const unsigned int cell, Number * scratch_array) {
          run_prolongation_list(cell);
          dispatch_cell_operation(cell, src, scratch_indices[cell], scratch_array);
        },
        output);
      scratch.finish_loop(output, {});
    }
    fe_correction.zero_out_ghost_values();

    timings[7] += time.wall_time();
    timings[8] += time.wall_time();
    timings[9] += total_timer.wall_time();
  }

  // Part of do_vmult_with_prolongation() with the batches of
  // setup_prolongation_batches()
  template<bool cellwise, typename Output>
  void
  vmult_with_prolongation_batches(VectorType &       src,
                                  const VectorType & fe_correction,
                                  const Output &     output) const
  {
    const unsigned int degree = shape_info.data[0].fe_degree;
    if(degree == 1)
      vmult_with_prolongation_batches<1, cellwise>(src, fe_correction, output);
    else if(degree == 2)
      vmult_with_prolongation_batches<2, cellwise>(src, fe_correction, output);
    else if(degree == 3)
      vmult_with_prolongation_batches<3, cellwise>(src, fe_correction, output);
    else if(degree == 4)
      vmult_with_prolongation_batches<4, cellwise>(src, fe_correction, output);
#ifndef DEBUG
    else if(degree == 5)
      vmult_with_prolongation_batches<5, cellwise>(src, fe_correction, output);
    else if(degree == 6)
      vmult_with_prolongation_batches<6, cellwise>(src, fe_correction, output);
    else if(degree == 7)
      vmult_with_prolongation_batches<7, cellwise>(src, fe_correction, output);
    else if(degree == 8)
      vmult_with_prolongation_batches<8, cellwise>(src, fe_correction, output);
#endif
    else
      AssertThrow(false, ExcMessage("Degree " + std::to_string(degree) + " not instantiated"));
  }

  template<int degree, bool cellwise, typename Output>
  void
  vmult_with_prolongation_batches(VectorType &       src,
                                  const VectorType & fe_correction,
                                  const Output &     output) const
  {
    Timer              total_timer;
    const unsigned int n_cell_batches = dof_indices.size();
    timings[0] += 1;

    fe_correction.update_ghost_values();
    if constexpr(!cellwise)
      scratch.prepare_loop();
    const auto pack_hook = [&](const unsigned int group) {
      do_prolongate_on_batch<degree>(group, fe_correction, src);
    };
    exchange_face_data(src, &pack_hook);

    Timer      time;
    const auto run_prolongation = [&](const unsigned int cell) {
      for(unsigned int p = prolongation_batch_index[cell]; p < prolongation_batch_index[cell + 1];
          ++p)
        do_prolongate_on_batch<degree>(p, fe_correction, src);
    };
    if constexpr(cellwise)
      for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
      {
        run_prolongation(cell);
        do_cell_operation_cellwise<degree>(cell, src, output);
      }
    else
    {
      scratch.cell_loop(
        [&](const unsigned int cell, Number * scratch_array) {
          run_prolongation(cell);
          dispatch_cell_operation(cell, src, scratch_indices[cell], scratch_array);
        },
        output);
      scratch.finish_loop(output, {});
    }
    fe_correction.zero_out_ghost_values();

    timings[7] += time.wall_time();
    timings[8] += time.wall_time();
    timings[9] += total_timer.wall_time();
  }

  // For benchmarking the kernels of the prolongation from the FE vector
  // fe_correction (with ghost values) into dg, run over all cells in the
  // order of the cell batches: variant 0 per cell batch as in
  // prolongate_and_add(), variant 1 with the FE indices gathered into a
  // separate array as needed for batches of arbitrary cells, variant 2 with
  // a scalar read per cell and a contiguous update of the DG values
  void
  benchmark_prolongation_kernel(const unsigned int                     variant,
                                const VectorType &                     fe_correction,
                                const LaplaceOperatorFE<dim, Number> & fe_operator,
                                VectorType &                           dg) const
  {
    const unsigned int degree = shape_info.data[0].fe_degree;
    if(degree == 2)
      benchmark_prolongation_kernel<2>(variant, fe_correction, fe_operator, dg);
    else if(degree == 3)
      benchmark_prolongation_kernel<3>(variant, fe_correction, fe_operator, dg);
    else if(degree == 4)
      benchmark_prolongation_kernel<4>(variant, fe_correction, fe_operator, dg);
#ifndef DEBUG
    else if(degree == 5)
      benchmark_prolongation_kernel<5>(variant, fe_correction, fe_operator, dg);
    else if(degree == 6)
      benchmark_prolongation_kernel<6>(variant, fe_correction, fe_operator, dg);
    else if(degree == 7)
      benchmark_prolongation_kernel<7>(variant, fe_correction, fe_operator, dg);
    else if(degree == 8)
      benchmark_prolongation_kernel<8>(variant, fe_correction, fe_operator, dg);
#endif
    else
      AssertThrow(false, ExcMessage("Degree " + std::to_string(degree) + " not instantiated"));
  }

  template<int degree>
  void
  benchmark_prolongation_kernel(const unsigned int                     variant,
                                const VectorType &                     fe_correction,
                                const LaplaceOperatorFE<dim, Number> & fe_operator,
                                VectorType &                           dg) const
  {
    constexpr unsigned int dofs_per_cell  = Utilities::pow(degree + 1, dim);
    constexpr unsigned int n_stored       = Utilities::pow(3, dim);
    const unsigned int     n_cell_batches = dof_indices.size();
    if(variant == 0)
      for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
        do_prolongate<degree>(cell, fe_correction, fe_operator, dg);
    else if(variant == 1)
    {
      if constexpr(degree > 2)
      {
        if(benchmark_fe_indices.empty())
        {
          benchmark_fe_indices = fe_operator.get_compressed_dof_indices();
          benchmark_fe_unconstrained.assign(n_cell_batches * n_stored, 1);
          for(unsigned int c = 0; c < n_cell_batches; ++c)
            for(unsigned int i = 0; i < n_stored; ++i)
              for(unsigned int v = 0; v < n_lanes; ++v)
                if(benchmark_fe_indices[(c * n_stored + i) * n_lanes + v] ==
                   numbers::invalid_unsigned_int)
                  benchmark_fe_unconstrained[c * n_stored + i] = 0;
          benchmark_dg_starts = dof_indices;
        }
        for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
        {
          VectorizedArray<Number> values[dofs_per_cell];
          LaplaceOperatorFE<dim, Number>::
            template read_dof_values_compressed<degree, degree + 1, 1>(fe_correction,
                                                                       benchmark_fe_indices,
                                                                       benchmark_fe_unconstrained,
                                                                       cell,
                                                                       {},
                                                                       true,
                                                                       values);
          vectorized_transpose_and_store(
            true, dofs_per_cell, values, benchmark_dg_starts[cell].data(), dg.begin());
        }
      }
      else
        AssertThrow(false, ExcNotImplemented());
    }
    else
      for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
        for(unsigned int v = 0; v < n_active_entries_per_cell_batch(cell); ++v)
        {
          alignas(64) Number values[dofs_per_cell];
          fe_operator.template read_cell_values_scalar<degree>(cell, v, fe_correction, values);
          Number * DEAL_II_RESTRICT dst = dg.begin() + dof_indices[cell][v];
          DEAL_II_OPENMP_SIMD_PRAGMA
          for(unsigned int i = 0; i < dofs_per_cell; ++i)
            dst[i] += values[i];
        }
  }

  void
  vmult_residual_and_restrict_to_fe(const VectorType &                     rhs,
                                    const VectorType &                     solution,
                                    const LaplaceOperatorFE<dim, Number> & fe_operator,
                                    VectorType &                           dst) const
  {
    Timer total_timer;
    if(factor_laplace != 0.)
      timings[0] += 1;
    else
      timings[10] += 1;
    Timer time;

    dst = 0;

    const unsigned int n_cell_batches = dof_indices.size();

    time.restart();

    // only do the data exchange for face integral if we have Laplacian contribution
    if(factor_laplace != 0. && Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 1)
    {
      const int nn = shape_info.data[0].fe_degree + 1;

      // data: everything projected to face, 2 because of values and derivatives
      const int data_per_face = 2 * Utilities::pow(nn, dim - 1);
      AssertDimension(mpi_requests.size(), 2 * send_data_process.size());

      unsigned int offset = 0;
      for(unsigned int p = 0; p < send_data_process.size(); ++p)
      {
        MPI_Irecv(&import_values[offset * data_per_face],
                  send_data_process[p].second * data_per_face * sizeof(Number),
                  MPI_BYTE,
                  send_data_process[p].first,
                  send_data_process[p].first + 47,
                  solution.get_mpi_communicator(),
                  &mpi_requests[p]);
        offset += send_data_process[p].second;
      }
      AssertDimension(offset * data_per_face, import_values.size());

      if(nn == 2)
        vmult_pack_and_send_data<2>(solution);
      else if(nn == 3)
        vmult_pack_and_send_data<3>(solution);
      else if(nn == 4)
        vmult_pack_and_send_data<4>(solution);
#ifndef DEBUG
      else if(nn == 5)
        vmult_pack_and_send_data<5>(solution);
      else if(nn == 6)
        vmult_pack_and_send_data<6>(solution);
      else if(nn == 7)
        vmult_pack_and_send_data<7>(solution);
      else if(nn == 8)
        vmult_pack_and_send_data<8>(solution);
      else if(nn == 9)
        vmult_pack_and_send_data<9>(solution);
#endif
      else
        AssertThrow(false, ExcNotImplemented());

      timings[3] += time.wall_time();

      time.restart();
      if(!mpi_requests.empty())
        MPI_Waitall(mpi_requests.size(), &mpi_requests[0], MPI_STATUSES_IGNORE);
      timings[5] += time.wall_time();
    }

    time.restart();
    for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
    {
      const unsigned int degree = shape_info.data[0].fe_degree;
      if(degree == 1)
        do_residual_operation<1>(cell, rhs, solution, fe_operator, dst);
      else if(degree == 2)
        do_residual_operation<2>(cell, rhs, solution, fe_operator, dst);
      else if(degree == 3)
        do_residual_operation<3>(cell, rhs, solution, fe_operator, dst);
      else if(degree == 4)
        do_residual_operation<4>(cell, rhs, solution, fe_operator, dst);
#ifndef DEBUG
      else if(degree == 5)
        do_residual_operation<5>(cell, rhs, solution, fe_operator, dst);
      else if(degree == 6)
        do_residual_operation<6>(cell, rhs, solution, fe_operator, dst);
      else if(degree == 7)
        do_residual_operation<7>(cell, rhs, solution, fe_operator, dst);
      else if(degree == 8)
        do_residual_operation<8>(cell, rhs, solution, fe_operator, dst);
#endif
      else
        AssertThrow(false, ExcMessage("Degree " + std::to_string(degree) + " not instantiated"));
    }

    if(factor_laplace != 0)
      timings[7] += time.wall_time();
    else
      timings[12] += time.wall_time();

    time.restart();
    dst.compress(VectorOperation::add);

    if(factor_laplace != 0)
    {
      timings[8] += time.wall_time();
      timings[9] += total_timer.wall_time();
    }
    else
    {
      timings[13] += time.wall_time();
      timings[14] += total_timer.wall_time();
    }
  }

  void
  prolongate_and_add(VectorType &                           dg_solution,
                     const VectorType &                     fe_correction,
                     const LaplaceOperatorFE<dim, Number> & fe_operator) const
  {
    fe_correction.update_ghost_values();
    const unsigned int n_cell_batches = dof_indices.size();

    // Run loop in reverse direction to increase the cache hits with data
    // remaining from the previous cell loop on the fe level and the next cell
    // loop on the dg level (that both run in forward order through the cells)
    for(int cell = n_cell_batches - 1; cell >= 0; --cell)
    {
      const unsigned int degree = shape_info.data[0].fe_degree;
      if(degree == 1)
        do_prolongate<1>(cell, fe_correction, fe_operator, dg_solution);
      else if(degree == 2)
        do_prolongate<2>(cell, fe_correction, fe_operator, dg_solution);
      else if(degree == 3)
        do_prolongate<3>(cell, fe_correction, fe_operator, dg_solution);
      else if(degree == 4)
        do_prolongate<4>(cell, fe_correction, fe_operator, dg_solution);
#ifndef DEBUG
      else if(degree == 5)
        do_prolongate<5>(cell, fe_correction, fe_operator, dg_solution);
      else if(degree == 6)
        do_prolongate<6>(cell, fe_correction, fe_operator, dg_solution);
      else if(degree == 7)
        do_prolongate<7>(cell, fe_correction, fe_operator, dg_solution);
      else if(degree == 8)
        do_prolongate<8>(cell, fe_correction, fe_operator, dg_solution);
#endif
      else
        AssertThrow(false, ExcMessage("Degree " + std::to_string(degree) + " not instantiated"));
    }

    fe_correction.zero_out_ghost_values();
  }

  void
  evaluate_add_divergence_body_force(
    const double                                                        time,
    const dealii::Function<dim> &                                       rhs_function,
    const std::function<void(const unsigned int, const unsigned int)> & before_loop,
    VectorType &                                                        dst) const
  {
    const unsigned int n_cell_batches = dof_indices.size();
    const_cast<dealii::Function<dim> &>(rhs_function).set_time(time);

    for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
    {
      for(unsigned int range = cell_loop_mass_pre_list_index[cell];
          range < cell_loop_mass_pre_list_index[cell + 1];
          ++range)
        before_loop(cell_loop_mass_pre_list[range].first, cell_loop_mass_pre_list[range].second);

      const unsigned int degree = shape_info.data[0].fe_degree;
      if(degree == 1)
        do_body_force_on_cell<1>(cell, rhs_function, dst);
      else if(degree == 2)
        do_body_force_on_cell<2>(cell, rhs_function, dst);
      else if(degree == 3)
        do_body_force_on_cell<3>(cell, rhs_function, dst);
#ifndef DEBUG
      else if(degree == 4)
        do_body_force_on_cell<4>(cell, rhs_function, dst);
      else if(degree == 5)
        do_body_force_on_cell<5>(cell, rhs_function, dst);
      else if(degree == 6)
        do_body_force_on_cell<6>(cell, rhs_function, dst);
      else if(degree == 7)
        do_body_force_on_cell<7>(cell, rhs_function, dst);
      else if(degree == 8)
        do_body_force_on_cell<8>(cell, rhs_function, dst);
#endif
      else
        AssertThrow(false, ExcMessage("Degree " + std::to_string(degree) + " not instantiated"));
    }
  }

  std::size_t
  memory_consumption() const
  {
    return MemoryConsumption::memory_consumption(dof_indices) +
           MemoryConsumption::memory_consumption(neighbor_cells) +
           MemoryConsumption::memory_consumption(mpi_exchange_data_on_faces) +
           MemoryConsumption::memory_consumption(import_values) +
           MemoryConsumption::memory_consumption(export_values) +
           MemoryConsumption::memory_consumption(all_owned_faces) +
           mapping_info.memory_consumption() +
           MemoryConsumption::memory_consumption(interpolate_quad_to_boundary) +
           MemoryConsumption::memory_consumption(send_data_process) +
           MemoryConsumption::memory_consumption(send_data_cell_index) +
           MemoryConsumption::memory_consumption(send_data_face_index) +
           MemoryConsumption::memory_consumption(cell_loop_pre_list_index) +
           MemoryConsumption::memory_consumption(cell_loop_pre_list) +
           MemoryConsumption::memory_consumption(cell_loop_mass_pre_list_index) +
           MemoryConsumption::memory_consumption(cell_loop_mass_pre_list) +
           MemoryConsumption::memory_consumption(cell_loop_post_list_index) +
           MemoryConsumption::memory_consumption(cell_loop_post_list) +
           MemoryConsumption::memory_consumption(face_flux_buffer) +
           MemoryConsumption::memory_consumption(face_flux_buffer_index) +
           MemoryConsumption::memory_consumption(all_left_face_fluxes_from_buffer);
  }

  const std::vector<dealii::ndarray<unsigned int, n_lanes, 2>> &
  get_cell_level_index() const
  {
    return cell_level_index;
  }

  const Extruded::MappingInfo<dim, Number> &
  get_mapping_info() const
  {
    return mapping_info;
  }

  const std::vector<std::array<unsigned int, n_lanes>> &
  get_dof_indices() const
  {
    return dof_indices;
  }

  // Print statistics of the access to the vectors in the cell loop as
  // minimum, average and maximum over the processes: the number of ranges
  // passed to 'before_loop' and 'after_loop', the share of the entries
  // passing through 'before_loop' ahead of the loop (data sent to other
  // processes) and the number of cell batches until their first read in the
  // loop, and the number of positions in the loop where the next cell batch
  // does not start in memory where the previous one ended
  void
  print_data_locality_statistics() const
  {
    const MPI_Comm     comm           = dof_handler->get_mpi_communicator();
    const unsigned int n_cell_batches = dof_indices.size();
    const unsigned int n_dofs         = partitioner->locally_owned_size();
    const unsigned int dofs_per_cell  = dof_handler->get_fe().dofs_per_cell;

    std::vector<unsigned int> first_read(n_dofs, numbers::invalid_unsigned_int);
    unsigned int              n_jumps = 0, last_end = numbers::invalid_unsigned_int;
    for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
      for(unsigned int v = 0; v < n_active_entries_per_cell_batch(cell); ++v)
      {
        if(dof_indices[cell][v] != last_end)
          ++n_jumps;
        last_end = dof_indices[cell][v] + dofs_per_cell;
        for(unsigned int f = 0; f <= 2 * dim; ++f)
        {
          const unsigned int c = f < 2 * dim ? neighbor_cells[cell][f][v] : cell * n_lanes + v;
          if(c == numbers::invalid_unsigned_int)
            continue;
          const unsigned int idx = dof_indices[c / n_lanes][c % n_lanes];
          if(idx < n_dofs)
            for(unsigned int i = idx; i < idx + dofs_per_cell; ++i)
              first_read[i] = std::min(first_read[i], cell);
        }
      }
    unsigned int n_ahead      = 0;
    double       sum_distance = 0;
    for(unsigned int r = cell_loop_pre_list_index[n_cell_batches];
        r < cell_loop_pre_list_index[n_cell_batches + 1];
        ++r)
      for(unsigned int i = cell_loop_pre_list[r].first; i < cell_loop_pre_list[r].second; ++i)
      {
        ++n_ahead;
        sum_distance += first_read[i];
      }

    if(Utilities::MPI::this_mpi_process(comm) == 0)
      std::cout << "Data locality of DG Laplace operator <"
                << (std::is_same_v<Number, double> ? "double" : "float")
                << ">, min [proc] avg max [proc]:" << std::endl;
    Helper::print_time(n_cell_batches, "Cell batches", comm);
    Helper::print_time(n_dofs * sizeof(Number) / 1024., "Vector size [KiB]", comm);
    Helper::print_time(double(n_ahead) / std::max(1U, n_dofs), "Pre/post: share pre ahead", comm);
    Helper::print_time(n_ahead > 0 ? sum_distance / n_ahead : 0.,
                       "Pre/post: ahead to read [cb]",
                       comm);
    Helper::print_time(cell_loop_pre_list.size(), "Ranges before_loop", comm);
    Helper::print_time(cell_loop_post_list.size(), "Ranges after_loop", comm);
    Helper::print_time(n_jumps, "Jumps between cells", comm);
    if(Utilities::MPI::this_mpi_process(comm) == 0)
      std::cout << std::endl;
  }

private:
  bool                                   is_test;
  ObserverPointer<const DoFHandler<dim>> dof_handler;
  MatrixFree<dim, Number>                matrix_free;

  internal::MatrixFreeFunctions::ShapeInfo<Number>  shape_info;
  std::array<std::vector<std::array<Number, 2>>, 2> interpolate_quad_to_boundary;

  std::shared_ptr<const Utilities::MPI::Partitioner>           partitioner;
  std::vector<std::array<unsigned int, n_lanes>>               dof_indices;
  std::vector<dealii::ndarray<unsigned int, 2 * dim, n_lanes>> neighbor_cells;
  std::vector<dealii::ndarray<unsigned int, 2 * dim, n_lanes>> mpi_exchange_data_on_faces;
  mutable AlignedVector<Number>                                import_values;
  mutable AlignedVector<Number>                                export_values;
  mutable std::vector<MPI_Request>                             mpi_requests;
  Table<2, unsigned char>                                      all_owned_faces;
  std::vector<std::pair<unsigned int, unsigned int>>           send_data_process;
  std::vector<unsigned int>                                    send_data_cell_index;
  std::vector<unsigned char>                                   send_data_face_index;

  // scratch array of vmult_scratch() and the offsets of the cells into it
  ExaDG::ScratchMatrixVectorProduct<Number>      scratch;
  std::vector<std::array<unsigned int, n_lanes>> scratch_indices;

  Extruded::MappingInfo<dim, Number> mapping_info;

  mutable std::array<double, 15> timings;

  Number factor_mass;
  Number factor_laplace;

  AlignedVector<std::array<VectorizedArray<Number>, 2 * dim>> penalty_parameters;

  // eigenvectors (dim x (degree+1)^2 per cell batch, the rows of each matrix
  // corresponding to the shape functions) and eigenvalues (dim x (degree+1)
  // per cell batch) of the block-Jacobi preconditioner, see
  // setup_block_inverse()
  AlignedVector<VectorizedArray<Number>> block_eigenvectors;
  AlignedVector<VectorizedArray<Number>> block_eigenvalues;
  unsigned int                           block_inverse_iterations = 0;

  // empty vector for compute_face() without exterior values, which does
  // not read the source vector
  VectorType empty_vector;

  // prolongation batches of arbitrary cells, see
  // setup_prolongation_batches(): FE indices in the layout of the FE
  // operator, DG start indices, index into the masks (or invalid), and
  // the range of batches run before each cell batch (the batches before
  // prolongation_batch_index[0] are run with the packing of face data)
  mutable bool                                           prolongation_setup_done  = false;
  bool                                                   prolongation_lane_wise   = false;
  unsigned int                                           prolongation_lookahead   = 2;
  mutable bool                                           use_prolongation_batches = false;
  mutable std::vector<unsigned int>                      prolongation_fe_indices;
  mutable std::vector<unsigned char>                     prolongation_fe_unconstrained;
  mutable std::vector<std::array<unsigned int, n_lanes>> prolongation_dg_starts;
  mutable std::vector<unsigned int>                      prolongation_mask_index;
  mutable AlignedVector<VectorizedArray<Number>>         prolongation_masks;
  mutable std::vector<unsigned int>                      prolongation_batch_index;

  // index arrays of benchmark_prolongation_kernel(), variant 1
  mutable std::vector<unsigned int>                      benchmark_fe_indices;
  mutable std::vector<unsigned char>                     benchmark_fe_unconstrained;
  mutable std::vector<std::array<unsigned int, n_lanes>> benchmark_dg_starts;

  // cell batches with the bit mask of their lanes to prolongate before the
  // cell batch with the given index or, at index n_cell_batches, before the
  // exchange of face data, see setup_prolongation_lists()
  mutable std::vector<unsigned int>                          prolongation_list_index;
  mutable std::vector<std::pair<unsigned int, unsigned int>> prolongation_list;

  std::vector<unsigned int>                          cell_loop_pre_list_index;
  std::vector<std::pair<unsigned int, unsigned int>> cell_loop_pre_list;
  std::vector<unsigned int>                          cell_loop_mass_pre_list_index;
  std::vector<std::pair<unsigned int, unsigned int>> cell_loop_mass_pre_list;
  std::vector<unsigned int>                          cell_loop_post_list_index;
  std::vector<std::pair<unsigned int, unsigned int>> cell_loop_post_list;

  std::vector<dealii::ndarray<unsigned int, n_lanes, 2>> cell_level_index;

  mutable AlignedVector<Number>                        face_flux_buffer;
  std::vector<ndarray<unsigned int, 2 * dim, n_lanes>> face_flux_buffer_index;
  std::vector<std::array<bool, dim>>                   all_left_face_fluxes_from_buffer;

  unsigned int
  n_active_entries_per_cell_batch(const unsigned int cell) const
  {
    AssertIndexRange(cell, dof_indices.size());
    unsigned int index = n_lanes - 1;
    while(cell_level_index[cell][index][0] == numbers::invalid_unsigned_int)
    {
      Assert(index > 0, ExcInternalError());
      --index;
    }
    return index + 1;
  }

  typename DoFHandler<dim>::cell_iterator
  get_cell_iterator(const unsigned int cell, const unsigned int lane) const
  {
    return typename DoFHandler<dim>::cell_iterator(&dof_handler->get_triangulation(),
                                                   cell_level_index[cell][lane][0],
                                                   cell_level_index[cell][lane][1],
                                                   dof_handler.get());
  }

  void
  compute_vector_access_pattern()
  {
    const unsigned int        n_dofs     = partitioner->locally_owned_size();
    constexpr unsigned int    chunk_size = 128;
    std::vector<unsigned int> touched_first_by((n_dofs + chunk_size - 1) / chunk_size,
                                               numbers::invalid_unsigned_int);
    std::vector<unsigned int> touched_mass_first_by((n_dofs + chunk_size - 1) / chunk_size,
                                                    numbers::invalid_unsigned_int);
    std::vector<unsigned int> touched_last_by((n_dofs + chunk_size - 1) / chunk_size,
                                              numbers::invalid_unsigned_int);
    const unsigned int        n_cell_batches = dof_indices.size();
    const unsigned int        dofs_on_cell   = dof_handler->get_fe().dofs_per_cell;
    for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
    {
      for(unsigned int face = 0; face < 2 * dim; ++face)
        for(unsigned int v = 0; v < n_lanes; ++v)
        {
          const unsigned int c = neighbor_cells[cell][face][v];
          if(c != numbers::invalid_unsigned_int)
          {
            const unsigned int idx = dof_indices[c / n_lanes][c % n_lanes];
            if(idx < n_dofs)
            {
              const unsigned int first = idx / chunk_size;
              const unsigned int last  = (idx + dofs_on_cell - 1) / chunk_size;
              for(unsigned int j = first; j <= last; ++j)
                if(touched_first_by[j] == numbers::invalid_unsigned_int)
                  touched_first_by[j] = cell;
            }
          }
        }
      for(unsigned int v = 0; v < n_lanes; ++v)
      {
        const unsigned int idx = dof_indices[cell][v];
        if(idx < n_dofs)
        {
          const unsigned int first = idx / chunk_size;
          const unsigned int last  = (idx + dofs_on_cell - 1) / chunk_size;
          for(unsigned int j = first; j <= last; ++j)
          {
            if(touched_first_by[j] == numbers::invalid_unsigned_int)
              touched_first_by[j] = cell;
            if(touched_mass_first_by[j] == numbers::invalid_unsigned_int)
              touched_mass_first_by[j] = cell;
            touched_last_by[j] = cell;
          }
        }
      }
    }

    // ensure that all indices are touched at least during the last round
    for(const auto & index : touched_first_by)
      AssertThrow(index != numbers::invalid_unsigned_int, ExcInternalError());
    for(const auto & index : touched_mass_first_by)
      AssertThrow(index != numbers::invalid_unsigned_int, ExcInternalError());
    for(const auto & index : touched_last_by)
      AssertThrow(index != numbers::invalid_unsigned_int, ExcInternalError());

    // debug output to assess sanity of implementation
    if(false)
    {
      unsigned int n_batches_half = 0, n_batches_10 = 0;
      for(unsigned int i = 0; i < touched_first_by.size(); ++i)
        if(touched_last_by[i] - touched_first_by[i] > n_cell_batches / 2)
          ++n_batches_half;
        else if(touched_last_by[i] - touched_first_by[i] > 10)
          ++n_batches_10;

      Helper::print_time(static_cast<double>(n_batches_half) / touched_first_by.size(),
                         "Pre-/post distance > 1/2 size",
                         MPI_COMM_WORLD);
      Helper::print_time(static_cast<double>(n_batches_half + n_batches_10) /
                           touched_first_by.size(),
                         "Pre-/post distance > 10",
                         MPI_COMM_WORLD);
    }

    for(const unsigned int cell_lane : send_data_cell_index)
    {
      const unsigned int cell = cell_lane / n_lanes;
      const unsigned int lane = cell_lane % n_lanes;
      const unsigned int idx  = dof_indices[cell][lane];
      if(idx < n_dofs)
      {
        const unsigned int first = idx / chunk_size;
        const unsigned int last  = (idx + dofs_on_cell - 1) / chunk_size;
        for(unsigned int j = first; j <= last; ++j)
          if(touched_first_by[j] != n_cell_batches)
            touched_first_by[j] = n_cell_batches;
      }
    }

#if 0
    std::cout << "data locality quality: ";
    for (unsigned int i = 0; i < touched_first_by.size(); ++i)
      std::cout << touched_last_by[i] - touched_first_by[i] << "  ";
    std::cout << std::endl;
#endif

    // debug output to assess sanity of implementation
    if(false)
    {
      unsigned int n_batches_half = 0, n_batches_10 = 0;
      for(unsigned int i = 0; i < touched_first_by.size(); ++i)
        if(touched_last_by[i] - touched_first_by[i] > n_cell_batches / 2)
          ++n_batches_half;
        else if(touched_last_by[i] - touched_first_by[i] > 10)
          ++n_batches_10;

      Helper::print_time(static_cast<double>(n_batches_half) / touched_first_by.size(),
                         "Pre-/post distance > 1/2 size",
                         MPI_COMM_WORLD);
      Helper::print_time(static_cast<double>(n_batches_half + n_batches_10) /
                           touched_first_by.size(),
                         "Pre-/post distance > 10",
                         MPI_COMM_WORLD);
    }

    std::map<unsigned int, std::vector<unsigned int>> chunk_must_do_pre;
    for(unsigned int i = 0; i < touched_first_by.size(); ++i)
      chunk_must_do_pre[touched_first_by[i]].push_back(i);
    Helper::convert_map_to_range_list(n_cell_batches + 1,
                                      chunk_size,
                                      chunk_must_do_pre,
                                      cell_loop_pre_list_index,
                                      cell_loop_pre_list,
                                      n_dofs);

    std::map<unsigned int, std::vector<unsigned int>> chunk_must_do_mass_pre;
    for(unsigned int i = 0; i < touched_mass_first_by.size(); ++i)
      chunk_must_do_mass_pre[touched_mass_first_by[i]].push_back(i);
    Helper::convert_map_to_range_list(n_cell_batches,
                                      chunk_size,
                                      chunk_must_do_mass_pre,
                                      cell_loop_mass_pre_list_index,
                                      cell_loop_mass_pre_list,
                                      n_dofs);

    std::map<unsigned int, std::vector<unsigned int>> chunk_must_do_post;
    for(unsigned int i = 0; i < touched_last_by.size(); ++i)
      chunk_must_do_post[touched_last_by[i]].push_back(i);
    Helper::convert_map_to_range_list(n_cell_batches,
                                      chunk_size,
                                      chunk_must_do_post,
                                      cell_loop_post_list_index,
                                      cell_loop_post_list,
                                      n_dofs);

#if 0
    for (unsigned int i = 0; i < n_cell_batches + 2; ++i)
      {
        std::cout << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) << " pre ";
        for (unsigned int j = cell_loop_pre_list_index[i];
             j < cell_loop_pre_list_index[i + 1];
             ++j)
          std::cout << "[" << cell_loop_pre_list[j].first << ","
                    << cell_loop_pre_list[j].second << ") ";
        if (i < n_cell_batches + 1)
          {
        std::cout << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) << " pre mass ";
        for (unsigned int j = cell_loop_mass_pre_list_index[i];
             j < cell_loop_mass_pre_list_index[i + 1];
             ++j)
          std::cout << "[" << cell_loop_mass_pre_list[j].first << ","
                    << cell_loop_mass_pre_list[j].second << ") ";
        std::cout << "post ";
        for (unsigned int j = cell_loop_post_list_index[i];
             j < cell_loop_post_list_index[i + 1];
             ++j)
          std::cout << "[" << cell_loop_post_list[j].first << ","
                    << cell_loop_post_list[j].second << ")";
        std::cout << std::endl;
      }
      }
#endif
  }

  template<int nn, int n_points_1d>
  void
  read_cell_values(const Number *                            src_vector,
                   const std::array<unsigned int, n_lanes> & dof_indices,
                   VectorizedArray<Number> *                 out) const
  {
    static_assert(nn > 1, "Degree 0 not supported");
    constexpr unsigned int          dofs_per_plane = nn * nn;
    constexpr unsigned int          n_points_2d    = n_points_1d * n_points_1d;
    VectorizedArray<Number>         cell_data[dofs_per_plane];
    const Number * DEAL_II_RESTRICT shape_data = shape_info.data[0].shape_values_eo.data();

    for(unsigned int i_z = 0; i_z < (dim > 2 ? nn : 1); ++i_z)
    {
      vectorized_load_and_transpose(dofs_per_plane,
                                    src_vector + i_z * dofs_per_plane,
                                    dof_indices.data(),
                                    cell_data);
      VectorizedArray<Number> * out_z = out + i_z * n_points_2d;

      // perform interpolation in x direction
      for(unsigned int i_y = 0; i_y < nn; ++i_y)
      {
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::value,
                                              nn,
                                              n_points_1d,
                                              1,
                                              1,
                                              true,
                                              false>(shape_data,
                                                     cell_data + i_y * nn,
                                                     out_z + i_y * n_points_1d);
      }

      // perform interpolation in y direction
      for(unsigned int i_x = 0; i_x < n_points_1d; ++i_x)
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::value,
                                              nn,
                                              n_points_1d,
                                              n_points_1d,
                                              n_points_1d,
                                              true,
                                              false>(shape_data, out_z + i_x, out_z + i_x);
    }
    if constexpr(dim == 3)
      for(unsigned int i = 0; i < n_points_2d; ++i)
      {
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::value,
                                              nn,
                                              n_points_1d,
                                              n_points_2d,
                                              n_points_2d,
                                              true,
                                              false>(shape_data, out + i, out + i);
      }
  }

  template<int nn, int n_points_1d>
  void
  integrate_cell_scatter(const dealii::ndarray<unsigned int, n_lanes> & dof_indices,
                         const unsigned int                             n_filled_lanes,
                         VectorizedArray<Number> *                      quad_values,
                         Number *                                       dst_vector) const
  {
    constexpr unsigned int  dofs_per_plane = nn * nn;
    constexpr unsigned int  n_points_2d    = n_points_1d * n_points_1d;
    VectorizedArray<Number> cell_data[n_points_2d];

    const Number * DEAL_II_RESTRICT shape_data = shape_info.data[0].shape_values_eo.data();
    if constexpr(dim == 3)
      for(unsigned int i = 0; i < n_points_1d * n_points_1d; ++i)
      {
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::value,
                                              nn,
                                              n_points_1d,
                                              n_points_2d,
                                              n_points_2d,
                                              false,
                                              false>(shape_data, quad_values + i, quad_values + i);
      }
    for(unsigned int i_z = 0; i_z < (dim > 2 ? nn : 1); ++i_z)
    {
      VectorizedArray<Number> * out_z = quad_values + i_z * n_points_2d;
      // perform integration in y direction
      for(unsigned int i_x = 0; i_x < n_points_1d; ++i_x)
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::value,
                                              nn,
                                              n_points_1d,
                                              n_points_1d,
                                              n_points_1d,
                                              false,
                                              false>(shape_data, out_z + i_x, out_z + i_x);

      // perform integration in x direction
      for(unsigned int i_y = 0; i_y < nn; ++i_y)
      {
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::value,
                                              nn,
                                              n_points_1d,
                                              1,
                                              1,
                                              false,
                                              false>(shape_data,
                                                     out_z + i_y * n_points_1d,
                                                     cell_data + i_y * nn);
      }
      if(n_filled_lanes == n_lanes)
        vectorized_transpose_and_store(
          true, dofs_per_plane, cell_data, dof_indices.data(), dst_vector + i_z * dofs_per_plane);
      else
        for(unsigned int v = 0; v < n_filled_lanes; ++v)
        {
          Number * dst_ptr = dst_vector + dof_indices[v] + i_z * dofs_per_plane;
          for(unsigned int i = 0; i < dofs_per_plane; ++i)
            dst_ptr[i] += cell_data[i][v];
        }
    }
  }

  template<int nn>
  void
  read_face_values(const Number *                            src_vector,
                   const unsigned int                        face,
                   const std::array<unsigned int, n_lanes> & dof_indices,
                   VectorizedArray<Number> *                 out) const
  {
    const unsigned int face_direction = face / 2;
    const unsigned int side           = face % 2;

    constexpr unsigned int  dofs_per_face  = Utilities::pow(nn, dim - 1);
    constexpr unsigned int  dofs_per_plane = nn * nn;
    VectorizedArray<Number> data_cell[dofs_per_face];

    const Number * DEAL_II_RESTRICT shape_data = shape_info.data[0].shape_data_on_face[side].data();

    for(unsigned int i2 = 0; i2 < (dim > 2 ? nn : 1); ++i2)
    {
      vectorized_load_and_transpose(dofs_per_plane,
                                    src_vector + i2 * dofs_per_plane,
                                    dof_indices.data(),
                                    data_cell);

      if(face_direction < 2)
      {
        VectorizedArray<Number> * out_val  = out + i2 * nn;
        VectorizedArray<Number> * out_grad = out + dofs_per_face + i2 * nn;
        if(face_direction == 0)
        {
          for(unsigned int i1 = 0; i1 < nn; ++i1)
          {
            VectorizedArray<Number> sum_v = data_cell[i1 * nn] * shape_data[0];
            VectorizedArray<Number> sum_g = data_cell[i1 * nn] * shape_data[nn];
            for(unsigned int i = 1; i < nn; ++i)
            {
              sum_v += data_cell[i1 * nn + i] * shape_data[i];
              sum_g += data_cell[i1 * nn + i] * shape_data[nn + i];
            }

            out_val[i1]  = sum_v;
            out_grad[i1] = sum_g;
          }
        }
        else
        {
          // face_direction == 1
          for(unsigned int i1 = 0; i1 < nn; ++i1)
          {
            VectorizedArray<Number> sum_v = data_cell[i1] * shape_data[0];
            VectorizedArray<Number> sum_g = data_cell[i1] * shape_data[nn];
            for(unsigned int i = 1; i < nn; ++i)
            {
              sum_v += data_cell[i * nn + i1] * shape_data[i];
              sum_g += data_cell[i * nn + i1] * shape_data[nn + i];
            }

            out_val[i1]  = sum_v;
            out_grad[i1] = sum_g;
          }
        }
      }
      else
      {
        Assert(dim == 3, ExcInternalError());
        Assert(face_direction == 2, ExcInternalError());
        VectorizedArray<Number> *     out_val  = out;
        VectorizedArray<Number> *     out_grad = out + dofs_per_face;
        const VectorizedArray<Number> val_i2   = shape_data[i2];
        const VectorizedArray<Number> grad_i2  = shape_data[i2 + nn];
        if(i2 == 0)
          for(unsigned int i = 0; i < dofs_per_face; ++i)
          {
            out_val[i]  = val_i2 * data_cell[i];
            out_grad[i] = grad_i2 * data_cell[i];
          }
        else
          for(unsigned int i = 0; i < dofs_per_face; ++i)
          {
            out_val[i] += val_i2 * data_cell[i];
            out_grad[i] += grad_i2 * data_cell[i];
          }
      }
    }
  }



  // Prolongate the lanes given by the bit mask 'lanes' of a cell batch
  void
  dispatch_prolongate(const unsigned int                     degree,
                      const unsigned int                     cell,
                      const unsigned int                     lanes,
                      const VectorType &                     fe_correction,
                      const LaplaceOperatorFE<dim, Number> & fe_operator,
                      VectorType &                           dg_solution) const
  {
    if(degree == 1)
      do_prolongate_lanes<1>(cell, lanes, fe_correction, fe_operator, dg_solution);
    else if(degree == 2)
      do_prolongate_lanes<2>(cell, lanes, fe_correction, fe_operator, dg_solution);
    else if(degree == 3)
      do_prolongate_lanes<3>(cell, lanes, fe_correction, fe_operator, dg_solution);
    else if(degree == 4)
      do_prolongate_lanes<4>(cell, lanes, fe_correction, fe_operator, dg_solution);
#ifndef DEBUG
    else if(degree == 5)
      do_prolongate_lanes<5>(cell, lanes, fe_correction, fe_operator, dg_solution);
    else if(degree == 6)
      do_prolongate_lanes<6>(cell, lanes, fe_correction, fe_operator, dg_solution);
    else if(degree == 7)
      do_prolongate_lanes<7>(cell, lanes, fe_correction, fe_operator, dg_solution);
    else if(degree == 8)
      do_prolongate_lanes<8>(cell, lanes, fe_correction, fe_operator, dg_solution);
#endif
    else
      AssertThrow(false, ExcMessage("Degree " + std::to_string(degree) + " not instantiated"));
  }

  // Variant of do_prolongate() for the lanes given by a bit mask, using the
  // vectorized store if all active lanes are selected
  template<int degree>
  void
  do_prolongate_lanes(const unsigned int                     cell,
                      const unsigned int                     lanes,
                      const VectorType &                     fe_correction,
                      const LaplaceOperatorFE<dim, Number> & fe_operator,
                      VectorType &                           dg_solution) const
  {
    const unsigned int n_active = n_active_entries_per_cell_batch(cell);
    if(lanes == (1u << n_active) - 1u || (n_active == 32 && lanes == ~0u))
    {
      do_prolongate<degree>(cell, fe_correction, fe_operator, dg_solution);
      return;
    }
    constexpr unsigned int  dofs_per_cell = Utilities::pow(degree + 1, dim);
    VectorizedArray<Number> values_dofs[dofs_per_cell];
    fe_operator.template read_dof_values<degree>(cell, fe_correction, values_dofs);
    for(unsigned int v = 0; v < n_active; ++v)
      if(lanes & (1u << v))
      {
        Number * dst = dg_solution.begin() + dof_indices[cell][v];
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          dst[i] += values_dofs[i][v];
      }
  }

  // Set up the lists of cell batches to prolongate in
  // vmult_scratch_with_prolongation(): the batches with face data sent to
  // other processes at index n_cell_batches (before the exchange), all
  // others at the first cell batch reading them via the cell or face
  // integrals
  void
  setup_prolongation_lists() const
  {
    static_assert(n_lanes <= 32, "Lane mask must fit into 32 bits");
    const unsigned int        n_cell_batches = dof_indices.size();
    std::vector<unsigned int> first_reader(n_cell_batches, numbers::invalid_unsigned_int);
    std::vector<unsigned int> sent_lanes(n_cell_batches, 0);
    for(const unsigned int cell_lane : send_data_cell_index)
      sent_lanes[cell_lane / n_lanes] |= 1u << (cell_lane % n_lanes);
    for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
    {
      if(first_reader[cell] == numbers::invalid_unsigned_int)
        first_reader[cell] = cell;
      for(unsigned int face = 0; face < 2 * dim; ++face)
        for(unsigned int v = 0; v < n_lanes; ++v)
        {
          const unsigned int neighbor = neighbor_cells[cell][face][v];
          if(neighbor != numbers::invalid_unsigned_int &&
             first_reader[neighbor / n_lanes] == numbers::invalid_unsigned_int)
            first_reader[neighbor / n_lanes] = cell;
        }
    }

    // the lanes with face data sent to other processes before the
    // exchange, the remaining lanes before the first reader
    std::vector<std::vector<std::pair<unsigned int, unsigned int>>> lists(n_cell_batches + 1);
    for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
    {
      const unsigned int n_active = n_active_entries_per_cell_batch(cell);
      const unsigned int active   = n_active == 32 ? ~0u : (1u << n_active) - 1u;
      if(sent_lanes[cell] != 0)
        lists[n_cell_batches].emplace_back(cell, sent_lanes[cell]);
      if((active & ~sent_lanes[cell]) != 0)
        lists[first_reader[cell]].emplace_back(cell, active & ~sent_lanes[cell]);
    }
    prolongation_list_index.assign(n_cell_batches + 2, 0);
    prolongation_list.clear();
    for(unsigned int i = 0; i <= n_cell_batches; ++i)
    {
      prolongation_list.insert(prolongation_list.end(), lists[i].begin(), lists[i].end());
      prolongation_list_index[i + 1] = prolongation_list.size();
    }

    // share of cells prolongated before the exchange, for information
    unsigned int n_cells = 0, n_sent = 0, n_batches_sent = 0;
    for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
    {
      n_cells += n_active_entries_per_cell_batch(cell);
      n_sent += __builtin_popcount(sent_lanes[cell]);
      n_batches_sent += sent_lanes[cell] != 0;
    }
    const auto cells        = Utilities::MPI::sum(n_cells, MPI_COMM_WORLD);
    const auto sent         = Utilities::MPI::sum(n_sent, MPI_COMM_WORLD);
    const auto batches      = Utilities::MPI::sum(n_cell_batches, MPI_COMM_WORLD);
    const auto batches_sent = Utilities::MPI::sum(n_batches_sent, MPI_COMM_WORLD);
    if(!is_test && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "DG prolongation within the smoother: " << 100. * sent / cells
                << "% of cells (in " << 100. * batches_sent / batches
                << "% of cell batches) prolongated before the face exchange" << std::endl;
  }

  // Prolongate the cells of the prolongation batch p from the FE vector
  // fe_correction (with ghost values) into dg, see
  // setup_prolongation_batches()
  template<int degree>
  void
  do_prolongate_on_batch(const unsigned int p,
                         const VectorType & fe_correction,
                         VectorType &       dg) const
  {
    constexpr unsigned int  dofs_per_cell = Utilities::pow(degree + 1, dim);
    VectorizedArray<Number> values[dofs_per_cell];
    if constexpr(degree <= 2)
    {
      const unsigned int *  indices = prolongation_fe_indices.data() + p * dofs_per_cell * n_lanes;
      const unsigned char * unconstrained =
        prolongation_fe_unconstrained.data() + p * dofs_per_cell;
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        if(unconstrained[i])
          for(unsigned int v = 0; v < n_lanes; ++v)
            values[i][v] = fe_correction.local_element(indices[i * n_lanes + v]);
        else
        {
          values[i] = 0.;
          for(unsigned int v = 0; v < n_lanes; ++v)
            if(indices[i * n_lanes + v] != numbers::invalid_unsigned_int)
              values[i][v] = fe_correction.local_element(indices[i * n_lanes + v]);
        }
    }
    else
      LaplaceOperatorFE<dim, Number>::template read_dof_values_compressed<degree, degree + 1, 1>(
        fe_correction, prolongation_fe_indices, prolongation_fe_unconstrained, p, {}, true, values);
    const unsigned int mask = prolongation_mask_index[p];
    if(mask != numbers::invalid_unsigned_int)
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        values[i] *= prolongation_masks[mask];
    vectorized_transpose_and_store(
      true, dofs_per_cell, values, prolongation_dg_starts[p].data(), dg.begin());
  }

  // Set up the prolongation batches of vmult_scratch_with_prolongation():
  // batches of arbitrary cells, given by cell batch and lane, with their FE
  // indices gathered from the FE operator. First, one batch per group of
  // faces packed for other processes, in the order of
  // vmult_pack_and_send_data(), run right before the group is read; a cell
  // already prolongated in an earlier group is masked out with a factor 0.
  // Then the remaining cells in the order of the first cell batch reading
  // them via the cell or face integrals, in batches run before that cell
  // batch and filled with cells read up to 'look-ahead' cell batches later.
  // Unused lanes repeat the first lane with a factor 0.
  void
  setup_prolongation_batches(const LaplaceOperatorFE<dim, Number> & fe_operator) const
  {
    const unsigned int                n_cell_batches = dof_indices.size();
    const unsigned int                degree         = shape_info.data[0].fe_degree;
    const unsigned int                n_stored = Utilities::pow(degree <= 2 ? degree + 1 : 3, dim);
    const std::vector<unsigned int> & fe_indices = fe_operator.get_compressed_dof_indices();
    const unsigned int                look_ahead = prolongation_lookahead;

    prolongation_fe_indices.clear();
    prolongation_dg_starts.clear();
    prolongation_mask_index.clear();
    prolongation_masks.clear();
    const auto add_batch = [&](const std::vector<unsigned int> & cells,
                               const std::vector<bool> &         active) {
      AssertThrow(!cells.empty() && cells.size() <= n_lanes, ExcInternalError());
      std::array<unsigned int, n_lanes> starts;
      VectorizedArray<Number>           mask      = 1.;
      bool                              need_mask = false;
      for(unsigned int v = 0; v < n_lanes; ++v)
      {
        const unsigned int cell_lane = cells[v < cells.size() ? v : 0];
        const bool         is_active = v < cells.size() && active[v];
        starts[v]                    = dof_indices[cell_lane / n_lanes][cell_lane % n_lanes];
        if(!is_active)
        {
          mask[v]   = 0.;
          need_mask = true;
        }
      }
      const unsigned int p = prolongation_dg_starts.size();
      prolongation_dg_starts.push_back(starts);
      prolongation_fe_indices.resize((p + 1) * n_stored * n_lanes);
      for(unsigned int i = 0; i < n_stored; ++i)
        for(unsigned int v = 0; v < n_lanes; ++v)
        {
          const unsigned int cell_lane = cells[v < cells.size() ? v : 0];
          prolongation_fe_indices[(p * n_stored + i) * n_lanes + v] =
            fe_indices[((cell_lane / n_lanes) * n_stored + i) * n_lanes + cell_lane % n_lanes];
        }
      if(need_mask)
      {
        prolongation_mask_index.push_back(prolongation_masks.size());
        prolongation_masks.push_back(mask);
      }
      else
        prolongation_mask_index.push_back(numbers::invalid_unsigned_int);
    };

    // groups of the packing loop
    std::vector<char> done(n_cell_batches * n_lanes, 0);
    unsigned int      n_pack_lanes = 0, n_pack_repeated = 0;
    if(factor_laplace != 0. && Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 1)
    {
      unsigned int offset = 0;
      for(unsigned int p = 0; p < send_data_process.size(); ++p)
      {
        const unsigned int my_faces = send_data_process[p].second;
        for(unsigned int count = offset; count < offset + my_faces;)
        {
          // same grouping as in vmult_pack_and_send_data()
          const unsigned int face    = send_data_face_index[count];
          unsigned int       n_faces = 1;
          if(count + n_lanes <= offset + my_faces &&
             send_data_face_index[count + n_lanes - 1] == face)
            n_faces = n_lanes;
          else
            while(count + n_faces < offset + my_faces &&
                  send_data_face_index[count + n_faces] == face)
              ++n_faces;
          std::vector<unsigned int> cells;
          std::vector<bool>         active;
          for(unsigned int v = 0; v < n_faces; ++v, ++count)
          {
            const unsigned int cell_lane = send_data_cell_index[count];
            cells.push_back(cell_lane);
            active.push_back(!done[cell_lane]);
            n_pack_repeated += done[cell_lane];
            done[cell_lane] = 1;
          }
          n_pack_lanes += n_faces;
          add_batch(cells, active);
        }
        offset += my_faces;
      }
    }
    const unsigned int n_pack_batches = prolongation_dg_starts.size();

    // first reader of each remaining cell
    std::vector<unsigned int> first_reader(n_cell_batches * n_lanes, numbers::invalid_unsigned_int);
    for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
    {
      for(unsigned int v = 0; v < n_active_entries_per_cell_batch(cell); ++v)
        first_reader[cell * n_lanes + v] = std::min(first_reader[cell * n_lanes + v], cell);
      for(unsigned int face = 0; face < 2 * dim; ++face)
        for(unsigned int v = 0; v < n_lanes; ++v)
        {
          const unsigned int neighbor = neighbor_cells[cell][face][v];
          if(neighbor != numbers::invalid_unsigned_int)
            first_reader[neighbor] = std::min(first_reader[neighbor], cell);
        }
    }
    std::vector<std::pair<unsigned int, unsigned int>> order;
    for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
      for(unsigned int v = 0; v < n_active_entries_per_cell_batch(cell); ++v)
        if(!done[cell * n_lanes + v])
          order.emplace_back(first_reader[cell * n_lanes + v], cell * n_lanes + v);
    std::sort(order.begin(), order.end());

    // batches in the loop, run before the first reader of their first cell
    std::vector<unsigned int> batch_slot;
    unsigned int              n_partial = 0;
    for(unsigned int i = 0; i < order.size();)
    {
      const unsigned int        reader = order[i].first;
      std::vector<unsigned int> cells;
      while(i < order.size() && cells.size() < n_lanes && order[i].first <= reader + look_ahead)
        cells.push_back(order[i++].second);
      n_partial += cells.size() < n_lanes;
      add_batch(cells, std::vector<bool>(cells.size(), true));
      batch_slot.push_back(reader);
    }
    prolongation_batch_index.assign(n_cell_batches + 1, n_pack_batches);
    {
      std::vector<unsigned int> count(n_cell_batches + 1, 0);
      for(const unsigned int slot : batch_slot)
        ++count[slot + 1];
      for(unsigned int c = 0; c < n_cell_batches; ++c)
        prolongation_batch_index[c + 1] = prolongation_batch_index[c] + count[c + 1];
      // the batches are created in the order of their slot
      for(unsigned int b = 1; b < batch_slot.size(); ++b)
        AssertThrow(batch_slot[b] >= batch_slot[b - 1], ExcInternalError());
    }

    prolongation_fe_unconstrained.assign(prolongation_dg_starts.size() * n_stored, 1);
    for(unsigned int p = 0; p < prolongation_dg_starts.size(); ++p)
      for(unsigned int i = 0; i < n_stored; ++i)
        for(unsigned int v = 0; v < n_lanes; ++v)
          if(prolongation_fe_indices[(p * n_stored + i) * n_lanes + v] ==
             numbers::invalid_unsigned_int)
            prolongation_fe_unconstrained[p * n_stored + i] = 0;

    // statistics
    unsigned int n_cells = 0;
    for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
      n_cells += n_active_entries_per_cell_batch(cell);
    const auto sum = [](const unsigned int value) {
      return Utilities::MPI::sum(value, MPI_COMM_WORLD);
    };
    const unsigned int cells_total = sum(n_cells), pack_lanes = sum(n_pack_lanes),
                       pack_repeated = sum(n_pack_repeated), pack_batches = sum(n_pack_batches),
                       loop_batches = sum(batch_slot.size()), partial = sum(n_partial),
                       batches_total = sum(n_cell_batches);
    if(!is_test && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "DG prolongation batches within the smoother: " << pack_batches
                << " with packed faces (" << pack_lanes << " lanes, " << pack_repeated
                << " repeated cells), " << loop_batches << " in the loop (" << partial
                << " partial, look-ahead " << look_ahead << "), for " << cells_total << " cells in "
                << batches_total << " cell batches" << std::endl;
  }

  // Pack the face data sent to other processes and send it. If given,
  // 'pack_hook(group)' is called before the cells of each group of faces
  // read together are read, with the groups numbered over all processes in
  // the order of this loop (see setup_prolongation_batches()).
  template<int nn, typename PackHook = std::nullptr_t>
  void
  vmult_pack_and_send_data(const VectorType & src, const PackHook * pack_hook = nullptr) const
  {
    constexpr int data_per_face = 2 * Utilities::pow(nn, dim - 1);
    unsigned int  group         = 0;

    std::array<VectorizedArray<Number>, 2 * nn * nn> tmp_vec;

    AssertDimension(mpi_requests.size(), 2 * send_data_process.size());

    unsigned int offset = 0;
    for(unsigned int p = 0; p < send_data_process.size(); ++p)
    {
      const unsigned int my_faces = send_data_process[p].second;
      for(unsigned int count = offset; count < offset + my_faces;)
      {
        const unsigned int face = send_data_face_index[count];

        // to use vectorized code path, must check that faces are
        // available
        unsigned int n_faces = 1;
        if(count + n_lanes <= offset + my_faces &&
           send_data_face_index[count + n_lanes - 1] == face)
          n_faces = n_lanes;
        else
          while(count + n_faces < offset + my_faces &&
                send_data_face_index[count + n_faces] == face)
            ++n_faces;
        AssertIndexRange(n_faces, n_lanes + 1);

        std::array<unsigned int, n_lanes> dof_indices_vec;
        dof_indices_vec[0] = 0;
        std::array<unsigned int, n_lanes> indices;
        for(unsigned int v = 0; v < n_faces; ++v, ++count)
        {
          indices[v]              = count * data_per_face;
          const unsigned int cell = send_data_cell_index[count] / n_lanes;
          const unsigned int lane = send_data_cell_index[count] % n_lanes;
          AssertDimension(face, send_data_face_index[count]);
          dof_indices_vec[v] = dof_indices[cell][lane];
        }
        for(unsigned int v = n_faces; v < n_lanes; ++v)
        {
          dof_indices_vec[v] = dof_indices_vec[0];
        }
        if constexpr(!std::is_same_v<PackHook, std::nullptr_t>)
          (*pack_hook)(group);
        ++group;
        read_face_values<nn>(src.begin(), face, dof_indices_vec, tmp_vec.data());

        // copy to dedicated data field
        if(n_faces == n_lanes)
          vectorized_transpose_and_store(
            false, data_per_face, tmp_vec.data(), indices.data(), export_values.data());
        else
          for(unsigned int v = 0; v < n_faces; ++v)
            for(unsigned int i = 0; i < data_per_face; ++i)
              export_values[indices[v] + i] = tmp_vec[i][v];
      }

      MPI_Isend(&export_values[offset * data_per_face],
                my_faces * data_per_face * sizeof(Number),
                MPI_BYTE,
                send_data_process[p].first,
                src.get_partitioner()->this_mpi_process() + 47,
                src.get_mpi_communicator(),
                &mpi_requests[send_data_process.size() + p]);
      offset += my_faces;
    }
    AssertDimension(offset * data_per_face, export_values.size());
  }

  // Send and receive the data of src on the faces to other processes
  // needed for the face integrals
  template<typename PackHook = std::nullptr_t>
  void
  exchange_face_data(const VectorType & src, const PackHook * pack_hook = nullptr) const
  {
    Timer time;
    // only do the data exchange for face integral if we have Laplacian contribution
    if(factor_laplace != 0. && Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 1)
    {
      const int nn = shape_info.data[0].fe_degree + 1;

      // data: everything projected to face, 2 because of values and derivatives
      const int data_per_face = 2 * Utilities::pow(nn, dim - 1);

      AssertDimension(mpi_requests.size(), 2 * send_data_process.size());
      unsigned int offset = 0;
      for(unsigned int p = 0; p < send_data_process.size(); ++p)
      {
        MPI_Irecv(&import_values[offset * data_per_face],
                  send_data_process[p].second * data_per_face * sizeof(Number),
                  MPI_BYTE,
                  send_data_process[p].first,
                  send_data_process[p].first + 47,
                  src.get_mpi_communicator(),
                  &mpi_requests[p]);
        offset += send_data_process[p].second;
      }
      AssertDimension(offset * data_per_face, import_values.size());

      if(nn == 2)
        vmult_pack_and_send_data<2>(src, pack_hook);
      else if(nn == 3)
        vmult_pack_and_send_data<3>(src, pack_hook);
      else if(nn == 4)
        vmult_pack_and_send_data<4>(src, pack_hook);
#ifndef DEBUG
      else if(nn == 5)
        vmult_pack_and_send_data<5>(src, pack_hook);
      else if(nn == 6)
        vmult_pack_and_send_data<6>(src, pack_hook);
      else if(nn == 7)
        vmult_pack_and_send_data<7>(src, pack_hook);
      else if(nn == 8)
        vmult_pack_and_send_data<8>(src, pack_hook);
      else if(nn == 9)
        vmult_pack_and_send_data<9>(src, pack_hook);
#endif
      else
        AssertThrow(false, ExcNotImplemented());

      timings[3] += time.wall_time();

      time.restart();
      if(!mpi_requests.empty())
        MPI_Waitall(mpi_requests.size(), &mpi_requests[0], MPI_STATUSES_IGNORE);
      timings[5] += time.wall_time();
    }
  }

  void
  diagonal_operation(const unsigned int cell, VectorType & dst) const
  {
    const unsigned int degree = shape_info.data[0].fe_degree;
    if(degree == 1)
      do_cell_operation<1, false>(cell, dst, dst);
    else if(degree == 2)
      do_cell_operation<2, false>(cell, dst, dst);
    else if(degree == 3)
      do_cell_operation<3, false>(cell, dst, dst);
    else if(degree == 4)
      do_cell_operation<4, false>(cell, dst, dst);
#ifndef DEBUG
    else if(degree == 5)
      do_cell_operation<5, false>(cell, dst, dst);
    else if(degree == 6)
      do_cell_operation<6, false>(cell, dst, dst);
    else if(degree == 7)
      do_cell_operation<7, false>(cell, dst, dst);
    else if(degree == 8)
      do_cell_operation<8, false>(cell, dst, dst);
#endif
    else
      AssertThrow(false, ExcMessage("Degree " + std::to_string(degree) + " not instantiated"));
  }

  template<int degree>
  void
  do_residual_operation(const unsigned int                     cell,
                        const VectorType &                     rhs,
                        const VectorType &                     solution,
                        const LaplaceOperatorFE<dim, Number> & fe_operator,
                        VectorType &                           dst) const
  {
    constexpr unsigned int dofs_per_cell = Utilities::pow(degree + 1, dim);
    constexpr unsigned int n_q_points_1d = degree + 1;
    constexpr unsigned int n_points      = Utilities::pow(n_q_points_1d, dim);
    const auto &           shape_data    = shape_info.data[0];

    VectorizedArray<Number> quad_values[n_points], out_values[n_points];
    read_cell_values<degree + 1, n_q_points_1d>(solution.begin(), dof_indices[cell], quad_values);

    if(factor_laplace != 0)
      compute_cell_lapl<n_q_points_1d>(
        shape_data, mapping_info, cell, factor_mass, factor_laplace, quad_values, out_values);
    else
      AssertThrow(false, ExcNotImplemented("Pure mass case not implemented yet"));

    // Face integrals if Laplace factor is positive
    if(factor_laplace != 0.)
      for(unsigned int f = 0; f < 2 * dim; ++f)
        compute_face<degree, true>(shape_data, solution, cell, f, quad_values, out_values);

    dealii::internal::FEEvaluationImplBasisChange<
      dealii::internal::evaluate_evenodd,
      dealii::internal::EvaluatorQuantity::value,
      dim,
      degree + 1,
      n_q_points_1d>::do_backward(1, shape_data.shape_values_eo, false, out_values, out_values);
    vectorized_load_and_transpose(dofs_per_cell,
                                  rhs.begin(),
                                  dof_indices[cell].data(),
                                  quad_values);
    for(unsigned int i = 0; i < dofs_per_cell; ++i)
      quad_values[i] -= out_values[i];
    fe_operator.template distribute_local_to_global<degree>(cell, quad_values, dst);
  }

  // Apply the operator on a cell batch and add the result into the array
  // dst at the positions given by dst_indices
  template<int degree>
  void
  do_cell_operation(const unsigned int                        cell,
                    const VectorType &                        src,
                    const std::array<unsigned int, n_lanes> & dst_indices,
                    Number *                                  dst) const
  {
    constexpr unsigned int n_q_points_1d = degree + 1;
    constexpr unsigned int n_points      = Utilities::pow(n_q_points_1d, dim);
    const auto &           shape_data    = shape_info.data[0];

    VectorizedArray<Number> quad_values[n_points], out_values[n_points];
    read_cell_values<degree + 1, n_q_points_1d>(src.begin(), dof_indices[cell], quad_values);

    if(factor_laplace != 0)
      compute_cell_lapl<n_q_points_1d>(
        shape_data, mapping_info, cell, factor_mass, factor_laplace, quad_values, out_values);
    else
      AssertThrow(false, ExcNotImplemented("Pure mass case not implemented yet"));

    // Face integrals if Laplace factor is positive
    if(factor_laplace != 0.)
      for(unsigned int f = 0; f < 2 * dim; ++f)
        compute_face<degree, true>(shape_data, src, cell, f, quad_values, out_values);

    integrate_cell_scatter<degree + 1, n_q_points_1d>(dst_indices,
                                                      n_active_entries_per_cell_batch(cell),
                                                      out_values,
                                                      dst);
  }

  void
  dispatch_cell_operation(const unsigned int                        cell,
                          const VectorType &                        src,
                          const std::array<unsigned int, n_lanes> & dst_indices,
                          Number *                                  dst) const
  {
    const unsigned int degree = shape_info.data[0].fe_degree;
    if(degree == 1)
      do_cell_operation<1>(cell, src, dst_indices, dst);
    else if(degree == 2)
      do_cell_operation<2>(cell, src, dst_indices, dst);
    else if(degree == 3)
      do_cell_operation<3>(cell, src, dst_indices, dst);
    else if(degree == 4)
      do_cell_operation<4>(cell, src, dst_indices, dst);
#ifndef DEBUG
    else if(degree == 5)
      do_cell_operation<5>(cell, src, dst_indices, dst);
    else if(degree == 6)
      do_cell_operation<6>(cell, src, dst_indices, dst);
    else if(degree == 7)
      do_cell_operation<7>(cell, src, dst_indices, dst);
    else if(degree == 8)
      do_cell_operation<8>(cell, src, dst_indices, dst);
#endif
    else
      AssertThrow(false, ExcMessage("Degree " + std::to_string(degree) + " not instantiated"));
  }

  // Integrate the quadrature point values of a cell batch against the
  // shape functions, like integrate_cell_scatter(), but in place in the
  // layout of read_cell_dofs() instead of adding into a vector
  template<int nn>
  void
  integrate_cell_in_place(VectorizedArray<Number> * values) const
  {
    constexpr unsigned int          n_points_1d = nn;
    constexpr unsigned int          n_points_2d = n_points_1d * n_points_1d;
    VectorizedArray<Number>         plane[n_points_2d];
    const Number * DEAL_II_RESTRICT shape_data = shape_info.data[0].shape_values_eo.data();
    if constexpr(dim == 3)
      for(unsigned int i = 0; i < n_points_2d; ++i)
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::value,
                                              nn,
                                              n_points_1d,
                                              n_points_2d,
                                              n_points_2d,
                                              false,
                                              false>(shape_data, values + i, values + i);
    for(unsigned int i_z = 0; i_z < (dim > 2 ? nn : 1); ++i_z)
    {
      VectorizedArray<Number> * out_z = values + i_z * n_points_2d;
      for(unsigned int i_x = 0; i_x < n_points_1d; ++i_x)
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::value,
                                              nn,
                                              n_points_1d,
                                              n_points_1d,
                                              n_points_1d,
                                              false,
                                              false>(shape_data, out_z + i_x, out_z + i_x);
      for(unsigned int i_y = 0; i_y < nn; ++i_y)
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::value,
                                              nn,
                                              n_points_1d,
                                              1,
                                              1,
                                              false,
                                              false>(shape_data,
                                                     out_z + i_y * n_points_1d,
                                                     plane + i_y * nn);
      for(unsigned int i = 0; i < n_points_2d; ++i)
        out_z[i] = plane[i];
    }
  }

  template<typename CellFunction>
  void
  dispatch_cell_operation_cellwise(const unsigned int   cell,
                                   const VectorType &   src,
                                   const CellFunction & cell_function) const
  {
    const unsigned int degree = shape_info.data[0].fe_degree;
    if(degree == 1)
      do_cell_operation_cellwise<1>(cell, src, cell_function);
    else if(degree == 2)
      do_cell_operation_cellwise<2>(cell, src, cell_function);
    else if(degree == 3)
      do_cell_operation_cellwise<3>(cell, src, cell_function);
    else if(degree == 4)
      do_cell_operation_cellwise<4>(cell, src, cell_function);
#ifndef DEBUG
    else if(degree == 5)
      do_cell_operation_cellwise<5>(cell, src, cell_function);
    else if(degree == 6)
      do_cell_operation_cellwise<6>(cell, src, cell_function);
    else if(degree == 7)
      do_cell_operation_cellwise<7>(cell, src, cell_function);
    else if(degree == 8)
      do_cell_operation_cellwise<8>(cell, src, cell_function);
#endif
    else
      AssertThrow(false, ExcMessage("Degree " + std::to_string(degree) + " not instantiated"));
  }

  // Apply the operator on a cell batch like do_cell_operation() and pass
  // the result in the layout of read_cell_dofs() to cell_function
  template<int degree, typename CellFunction>
  void
  do_cell_operation_cellwise(const unsigned int   cell,
                             const VectorType &   src,
                             const CellFunction & cell_function) const
  {
    constexpr unsigned int n_q_points_1d = degree + 1;
    constexpr unsigned int n_points      = Utilities::pow(n_q_points_1d, dim);
    const auto &           shape_data    = shape_info.data[0];

    VectorizedArray<Number> quad_values[n_points], out_values[n_points];
    read_cell_values<degree + 1, n_q_points_1d>(src.begin(), dof_indices[cell], quad_values);
    compute_cell_lapl<n_q_points_1d>(
      shape_data, mapping_info, cell, factor_mass, factor_laplace, quad_values, out_values);
    for(unsigned int f = 0; f < 2 * dim; ++f)
      compute_face<degree, true>(shape_data, src, cell, f, quad_values, out_values);
    integrate_cell_in_place<degree + 1>(out_values);
    cell_function(cell, static_cast<VectorizedArray<Number> *>(out_values));
  }

  // Compute the eigenvectors and eigenvalues of the 1D matrices of the
  // Cartesian surrogate of each cell, see setup_block_inverse()
  template<int degree>
  void
  do_setup_block_inverse()
  {
    constexpr unsigned int nn            = degree + 1;
    constexpr unsigned int n_q_points_1d = degree + 1;
    constexpr unsigned int n_q_points_2d = n_q_points_1d * n_q_points_1d;
    const auto &           shape_data    = shape_info.data[0];

    // traces of the shape functions and their derivatives on the left and
    // right end of the unit interval as in compute_diagonal_cell()
    double trace_value[2][nn], trace_derivative[2][nn];
    for(unsigned int side = 0; side < 2; ++side)
      for(unsigned int i = 0; i < nn; ++i)
      {
        double value = 0, derivative = 0;
        for(unsigned int q = 0; q < n_q_points_1d; ++q)
        {
          value += interpolate_quad_to_boundary[side][q][0] * shape_data.shape_values[i * nn + q];
          derivative +=
            interpolate_quad_to_boundary[side][q][1] * shape_data.shape_values[i * nn + q];
        }
        trace_value[side][i]      = value;
        trace_derivative[side][i] = derivative;
      }

    // 1D mass and stiffness matrices on the unit interval
    std::array<double, nn * nn> mass_ref, stiffness_ref;
    for(unsigned int i = 0; i < nn; ++i)
      for(unsigned int j = 0; j < nn; ++j)
      {
        double m = 0, k = 0;
        for(unsigned int q = 0; q < n_q_points_1d; ++q)
        {
          const double w = mapping_info.quad_weights_z[q];
          m += w * shape_data.shape_values[i * nn + q] * shape_data.shape_values[j * nn + q];
          k += w * shape_data.shape_gradients[i * nn + q] * shape_data.shape_gradients[j * nn + q];
        }
        mass_ref[i * nn + j]      = m;
        stiffness_ref[i * nn + j] = k;
      }

    const unsigned int n_batches = n_cell_batches();
    block_eigenvectors.resize_fast(n_batches * dim * nn * nn);
    block_eigenvalues.resize_fast(n_batches * dim * nn);
    std::array<double, nn * nn> mass, laplace, eigenvectors;
    std::array<double, nn>      eigenvalues;
    for(unsigned int cell = 0; cell < n_batches; ++cell)
    {
      const unsigned int n_active = n_active_entries_per_cell_batch(cell);
      for(unsigned int v = 0; v < n_lanes; ++v)
      {
        const unsigned int lane = v < n_active ? v : 0;

        // extents of the Cartesian surrogate
        std::array<double, dim> h;
        {
          const unsigned int base = mapping_info.mapping_data_index[cell][lane];
          double             h0 = 0, h1 = 0, weight_sum = 0;
          for(unsigned int q = 0; q < n_q_points_2d; ++q)
          {
            const double w   = mapping_info.quad_weights_xy[q];
            const auto & jac = mapping_info.jacobians_xy[base + q];
            h0 += w * std::sqrt(jac[0][0] * jac[0][0] + jac[1][0] * jac[1][0]);
            h1 += w * std::sqrt(jac[0][1] * jac[0][1] + jac[1][1] * jac[1][1]);
            weight_sum += w;
          }
          h[0] = h0 / weight_sum;
          h[1] = h1 / weight_sum;
          if constexpr(dim == 3)
            h[2] = mapping_info.h_z;
        }

        for(unsigned int d = 0; d < dim; ++d)
        {
          for(unsigned int i = 0; i < nn * nn; ++i)
          {
            mass[i]    = h[d] * mass_ref[i];
            laplace[i] = factor_laplace / h[d] * stiffness_ref[i];
          }
          for(unsigned int side = 0; side < 2; ++side)
          {
            const unsigned int f = 2 * d + side;
            // faces in z direction are always interior faces, see
            // compute_face()
            const bool interior =
              d == 2 || neighbor_cells[cell][f][lane] != numbers::invalid_unsigned_int;
            if(!interior)
              continue;
            const double sigma             = penalty_parameters[cell][f][lane] * factor_laplace;
            const double factor_derivative = 0.5 * factor_laplace * (side == 1 ? 1. : -1.) / h[d];
            for(unsigned int i = 0; i < nn; ++i)
              for(unsigned int j = 0; j < nn; ++j)
                laplace[i * nn + j] +=
                  sigma * trace_value[side][i] * trace_value[side][j] -
                  factor_derivative * (trace_derivative[side][i] * trace_value[side][j] +
                                       trace_value[side][i] * trace_derivative[side][j]);
          }
          dealii::internal::TensorProductMatrixSymmetricSum::spectral_assembly<double>(
            mass.data(), laplace.data(), nn, nn, eigenvalues.data(), eigenvectors.data());

          VectorizedArray<Number> * stored_vectors =
            block_eigenvectors.data() + (cell * dim + d) * nn * nn;
          VectorizedArray<Number> * stored_values =
            block_eigenvalues.data() + (cell * dim + d) * nn;
          for(unsigned int i = 0; i < nn * nn; ++i)
            stored_vectors[i][v] = eigenvectors[i];
          // the mass term is the identity in the eigenbasis of all
          // directions, so add it in equal parts to each direction
          for(unsigned int i = 0; i < nn; ++i)
            stored_values[i][v] = eigenvalues[i] + factor_mass / dim;
        }
      }
    }
  }

  // Apply the inverse of the surrogate block by fast diagonalization in
  // place
  template<int degree>
  void
  do_apply_fdm_cell(const unsigned int cell, VectorizedArray<Number> * values) const
  {
    constexpr unsigned int                           nn = degree + 1;
    std::array<const VectorizedArray<Number> *, dim> eigenvectors, eigenvalues;
    for(unsigned int d = 0; d < dim; ++d)
    {
      eigenvectors[d] = block_eigenvectors.data() + (cell * dim + d) * nn * nn;
      eigenvalues[d]  = block_eigenvalues.data() + (cell * dim + d) * nn;
    }
    dealii::internal::TensorProductMatrixSymmetricSum::apply_inverse<nn>(
      values, values, nn, eigenvectors, eigenvalues);
  }

  // Apply the approximate inverse of the cell block A_K: the fast
  // diagonalization z = P^{-1} r, followed by 'n_iterations' steps of the
  // preconditioned Richardson iteration z += P^{-1} (r - A_K z) with the
  // actual cell block. With a fixed number of steps, the result is a
  // linear and symmetric operator in r, as needed by the Chebyshev
  // iteration.
  template<int degree>
  void
  do_apply_block_inverse_cell(const unsigned int        cell,
                              VectorizedArray<Number> * values,
                              const unsigned int        n_iterations) const
  {
    constexpr unsigned int dofs_per_cell = Utilities::pow(degree + 1, dim);
    if(n_iterations == 0)
    {
      do_apply_fdm_cell<degree>(cell, values);
      return;
    }
    VectorizedArray<Number> rhs[dofs_per_cell], correction[dofs_per_cell];
    for(unsigned int i = 0; i < dofs_per_cell; ++i)
      rhs[i] = values[i];
    do_apply_fdm_cell<degree>(cell, values);
    for(unsigned int it = 0; it < n_iterations; ++it)
    {
      do_apply_cell_block<degree>(cell, values, correction);
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        correction[i] = rhs[i] - correction[i];
      do_apply_fdm_cell<degree>(cell, correction);
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        values[i] += correction[i];
    }
  }

  // Apply the block of the operator on the cells of a batch to the cell
  // values 'in' (layout of read_cell_dofs()) and write the result to 'out':
  // the cell integral and the face integrals with zero exterior values on
  // interior faces and the mirror principle on boundary faces
  template<int degree>
  void
  do_apply_cell_block(const unsigned int              cell,
                      const VectorizedArray<Number> * in,
                      VectorizedArray<Number> *       out) const
  {
    constexpr unsigned int          nn            = degree + 1;
    constexpr unsigned int          n_q_points_1d = degree + 1;
    constexpr unsigned int          n_points      = Utilities::pow(n_q_points_1d, dim);
    constexpr unsigned int          n_points_2d   = n_q_points_1d * n_q_points_1d;
    const auto &                    shape_data    = shape_info.data[0];
    const Number * DEAL_II_RESTRICT shape         = shape_data.shape_values_eo.data();

    // interpolate to the quadrature points as in read_cell_values()
    VectorizedArray<Number> quad_values[n_points], out_values[n_points];
    for(unsigned int i_z = 0; i_z < (dim > 2 ? nn : 1); ++i_z)
    {
      const VectorizedArray<Number> * in_z  = in + i_z * nn * nn;
      VectorizedArray<Number> *       out_z = quad_values + i_z * n_points_2d;
      for(unsigned int i_y = 0; i_y < nn; ++i_y)
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::value,
                                              nn,
                                              n_q_points_1d,
                                              1,
                                              1,
                                              true,
                                              false>(shape,
                                                     in_z + i_y * nn,
                                                     out_z + i_y * n_q_points_1d);
      for(unsigned int i_x = 0; i_x < n_q_points_1d; ++i_x)
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::value,
                                              nn,
                                              n_q_points_1d,
                                              n_q_points_1d,
                                              n_q_points_1d,
                                              true,
                                              false>(shape, out_z + i_x, out_z + i_x);
    }
    if constexpr(dim == 3)
      for(unsigned int i = 0; i < n_points_2d; ++i)
        internal::apply_matrix_vector_product<internal::evaluate_evenodd,
                                              internal::EvaluatorQuantity::value,
                                              nn,
                                              n_q_points_1d,
                                              n_points_2d,
                                              n_points_2d,
                                              true,
                                              false>(shape, quad_values + i, quad_values + i);

    compute_cell_lapl<n_q_points_1d>(
      shape_data, mapping_info, cell, factor_mass, factor_laplace, quad_values, out_values);
    if(factor_laplace != 0.)
      for(unsigned int f = 0; f < 2 * dim; ++f)
        compute_face<degree, false>(shape_data, empty_vector, cell, f, quad_values, out_values);
    integrate_cell_in_place<nn>(out_values);
    for(unsigned int i = 0; i < n_points; ++i)
      out[i] = out_values[i];
  }

  // Set up the scratch array of vmult_scratch() with the cells as entities
  void
  setup_scratch()
  {
    const unsigned int n_cell_batches = dof_indices.size();
    const unsigned int dofs_per_cell  = dof_handler->get_fe().dofs_per_cell;
    scratch.reinit(
      partitioner,
      n_cell_batches,
      [&](const unsigned int cell, const auto & add) {
        for(unsigned int v = 0; v < n_active_entries_per_cell_batch(cell); ++v)
          add(dof_indices[cell][v], dofs_per_cell);
      },
      false);

    scratch_indices.resize(n_cell_batches);
    for(unsigned int cell = 0; cell < n_cell_batches; ++cell)
    {
      const unsigned int n_active = n_active_entries_per_cell_batch(cell);
      for(unsigned int v = 0; v < n_lanes; ++v)
        scratch_indices[cell][v] =
          scratch.scratch_index(dof_indices[cell][std::min(v, n_active - 1)]);
    }
    scratch.release_setup_data();
  }

  template<int degree, bool compute_exterior>
  void
  do_cell_operation(const unsigned int cell, const VectorType & src, VectorType & dst) const
  {
    if(compute_exterior)
      do_cell_operation<degree>(cell, src, dof_indices[cell], dst.begin());
    else
      compute_diagonal_cell<degree>(cell, dst);
  }

  // Compute the diagonal of the operator on a cell batch and write it into
  // dst. As the DG basis functions live on a single cell, the diagonal
  // consists of the cell integral and the face integrals with the interior
  // values only, i.e., the exterior value of zero on interior faces and the
  // mirror values on boundary faces as in compute_face(). We use the tensor
  // product structure of the basis functions in compute_cell_lapl_diagonal()
  // for the cell term and the faces in z direction, whose integrals have
  // the same xy structure. On the faces with normal in x or y direction,
  // the face integral of basis function i is a function of the 1D traces of
  // the shape function and its derivative in normal direction times 1D
  // integrals in the tangential direction and in z direction.
  template<int degree>
  void
  compute_diagonal_cell(const unsigned int cell, VectorType & dst) const
  {
    static_assert(dim == 3, "Only implemented for dim = 3");
    constexpr unsigned int nn            = degree + 1;
    constexpr unsigned int n_q_points_1d = degree + 1;
    constexpr unsigned int dofs_per_cell = Utilities::pow(nn, dim);
    const auto &           shape_data    = shape_info.data[0];

    // traces of the shape functions and their derivatives on the left
    // (side 0) and right (side 1) end of the unit interval, with the same
    // interpolation from quadrature points as in compute_face()
    Number trace_value[2][nn], trace_derivative[2][nn];
    for(unsigned int side = 0; side < 2; ++side)
      for(unsigned int i = 0; i < nn; ++i)
      {
        Number value = 0, derivative = 0;
        for(unsigned int q = 0; q < n_q_points_1d; ++q)
        {
          value += interpolate_quad_to_boundary[side][q][0] * shape_data.shape_values[i * nn + q];
          derivative +=
            interpolate_quad_to_boundary[side][q][1] * shape_data.shape_values[i * nn + q];
        }
        trace_value[side][i]      = value;
        trace_derivative[side][i] = derivative;
      }

    // faces in z direction, assumed to be interior faces as in
    // compute_face()
    VectorizedArray<Number> z_face_terms[nn];
    for(unsigned int i2 = 0; i2 < nn; ++i2)
      z_face_terms[i2] = VectorizedArray<Number>();
    if(factor_laplace != 0.)
      for(unsigned int side = 0; side < 2; ++side)
      {
        const Number                  normal_sign = side ? 1 : -1;
        const VectorizedArray<Number> sigmaF =
          penalty_parameters[cell][4 + side] * this->factor_laplace;
        const Number factor_derivative =
          this->factor_laplace * normal_sign * mapping_info.h_z_inverse;
        for(unsigned int i2 = 0; i2 < nn; ++i2)
          z_face_terms[i2] +=
            sigmaF * (trace_value[side][i2] * trace_value[side][i2]) -
            factor_derivative * (trace_value[side][i2] * trace_derivative[side][i2]);
      }

    VectorizedArray<Number> diagonal[dofs_per_cell];
    compute_cell_lapl_diagonal<n_q_points_1d>(shape_data,
                                              mapping_info,
                                              cell,
                                              factor_mass,
                                              factor_laplace,
                                              factor_laplace != 0. ? z_face_terms : nullptr,
                                              diagonal);

    // faces in x and y direction
    if(factor_laplace != 0.)
    {
      Number z_value[nn];
      for(unsigned int i2 = 0; i2 < nn; ++i2)
      {
        z_value[i2] = 0;
        for(unsigned int qz = 0; qz < n_q_points_1d; ++qz)
          z_value[i2] += mapping_info.quad_weights_z[qz] * shape_data.shape_values[i2 * nn + qz] *
                         shape_data.shape_values[i2 * nn + qz];
      }

      for(unsigned int f = 0; f < 4; ++f)
      {
        const unsigned int face_direction = f / 2;
        const unsigned int side           = f % 2;

        // the mirror principle of compute_face() on boundary faces, with
        // boundary_mask = 1, cancels the face integrals
        VectorizedArray<Number> interior_face;
        for(unsigned int v = 0; v < n_lanes; ++v)
          interior_face[v] = neighbor_cells[cell][f][v] == numbers::invalid_unsigned_int ? 0. : 1.;
        const VectorizedArray<Number> sigmaF = penalty_parameters[cell][f] * this->factor_laplace;

        std::array<unsigned int, n_lanes> shifted_data_indices;
        for(unsigned int v = 0; v < n_lanes; ++v)
          shifted_data_indices[v] = mapping_info.face_mapping_data_index[cell][f][0][v] * 2;

        // 1D integrals in tangential direction of the factors multiplying
        // the squared trace value and the product of trace value and
        // normal derivative
        VectorizedArray<Number> factor_value[nn], factor_derivative[nn];
        for(unsigned int it = 0; it < nn; ++it)
        {
          factor_value[it]      = VectorizedArray<Number>();
          factor_derivative[it] = VectorizedArray<Number>();
        }
        for(unsigned int q1 = 0; q1 < n_q_points_1d; ++q1)
        {
          Tensor<1, 2, VectorizedArray<Number>> jac_x_normal;
          vectorized_load_and_transpose(2,
                                        &mapping_info.face_jxn_xy[q1][0],
                                        shifted_data_indices.data(),
                                        &jac_x_normal[0]);
          VectorizedArray<Number> JxW_xy;
          JxW_xy.gather(mapping_info.face_JxW_xy.data() + q1,
                        mapping_info.face_mapping_data_index[cell][f][0].data());
          const VectorizedArray<Number> factor = (JxW_xy * mapping_info.h_z) * interior_face;
          const VectorizedArray<Number> factor_normal =
            factor * (this->factor_laplace * jac_x_normal[face_direction]);
          const VectorizedArray<Number> factor_tangential =
            factor * (this->factor_laplace * jac_x_normal[1 - face_direction]);
          const VectorizedArray<Number> factor_penalty = factor * sigmaF;
          for(unsigned int it = 0; it < nn; ++it)
          {
            const Number value      = shape_data.shape_values[it * nn + q1];
            const Number derivative = shape_data.shape_gradients[it * nn + q1];
            factor_value[it] +=
              factor_penalty * (value * value) - factor_tangential * (value * derivative);
            factor_derivative[it] -= factor_normal * (value * value);
          }
        }

        for(unsigned int i2 = 0; i2 < nn; ++i2)
          for(unsigned int i1 = 0; i1 < nn; ++i1)
            for(unsigned int i0 = 0; i0 < nn; ++i0)
            {
              const unsigned int in = face_direction == 0 ? i0 : i1;
              const unsigned int it = face_direction == 0 ? i1 : i0;
              diagonal[(i2 * nn + i1) * nn + i0] +=
                z_value[i2] *
                (factor_value[it] * (trace_value[side][in] * trace_value[side][in]) +
                 factor_derivative[it] * (trace_value[side][in] * trace_derivative[side][in]));
            }
      }
    }

    const unsigned int n_entries_per_batch = n_active_entries_per_cell_batch(cell);
    for(unsigned int i = 0; i < dofs_per_cell; ++i)
      for(unsigned int v = 0; v < n_entries_per_batch; ++v)
        dst.local_element(dof_indices[cell][v] + i) = diagonal[i][v];
  }

  template<int degree, bool compute_exterior>
  void
  compute_face(const internal::MatrixFreeFunctions::UnivariateShapeData<Number> & shape_data,
               const VectorType &                                                 src,
               const unsigned int                                                 cell,
               const unsigned int                                                 f,
               const VectorizedArray<Number> *                                    quad_values,
               VectorizedArray<Number> *                                          out_values) const
  {
    constexpr bool use_face_buffer = true && compute_exterior;

    if(use_face_buffer && f % 2 == 0 && all_left_face_fluxes_from_buffer[cell][f / 2])
      return;

    constexpr unsigned int n_q_points_1d = degree + 1;
    constexpr unsigned int n_q_points_2d = n_q_points_1d * n_q_points_1d;
    constexpr unsigned int n_lanes       = VectorizedArray<Number>::size();

    std::array<unsigned int, n_lanes> dof_indices_neighbor;
    unsigned int                      lane_with_neighbor = 0;
    for(unsigned int v = 0; v < n_lanes; ++v)
      if(neighbor_cells[cell][f][v] != numbers::invalid_unsigned_int)
      {
        lane_with_neighbor = v;
        break;
      }

    // TODO: 0: interior face, 1: Neumann, -1: Dirichlet, apply from actual b.c. in operator
    VectorizedArray<Number> boundary_mask;
    for(unsigned int v = 0; v < n_lanes; ++v)
      if(neighbor_cells[cell][f][v] == numbers::invalid_unsigned_int)
        boundary_mask[v] = 1.;
      else
        boundary_mask[v] = 0.;

    if(compute_exterior)
    {
      for(unsigned int v = 0; v < n_lanes; ++v)
      {
        // use a dummy neighbor with nearby data for cells at boundary to make
        // sure vectorized functions continue working
        const unsigned int neighbor_idx =
          neighbor_cells[cell][f][v] != numbers::invalid_unsigned_int ?
            neighbor_cells[cell][f][v] :
            neighbor_cells[cell][f][lane_with_neighbor];
        if(neighbor_idx != numbers::invalid_unsigned_int)
          dof_indices_neighbor[v] = dof_indices[neighbor_idx / n_lanes][neighbor_idx % n_lanes];
        else
          dof_indices_neighbor[v] = dof_indices[0][0];
      }
    }

    VectorizedArray<Number> face_array[2 * Utilities::pow(n_q_points_1d, dim - 1)];
    constexpr int           nn            = degree + 1;
    constexpr int           dofs_per_face = Utilities::pow(nn, dim - 1);
    const Number            factor_lapl   = this->factor_laplace;

    internal::EvaluatorTensorProduct<internal::evaluate_evenodd,
                                     dim - 1,
                                     n_q_points_1d,
                                     n_q_points_1d,
                                     VectorizedArray<Number>,
                                     Number>
      eval_g({}, shape_data.shape_gradients_collocation_eo.data(), {});
    internal::EvaluatorTensorProduct<internal::evaluate_evenodd,
                                     dim - 1,
                                     nn,
                                     n_q_points_1d,
                                     VectorizedArray<Number>,
                                     Number>
      eval(shape_data.shape_values_eo.data(), {}, {});

    const unsigned int      face_direction  = f / 2;
    constexpr unsigned int  n_q_points_face = Utilities::pow(n_q_points_1d, dim - 1);
    VectorizedArray<Number> scratch_data[n_q_points_face + 1];
    VectorizedArray<Number> values_face[2][n_q_points_face + 1];
    VectorizedArray<Number> grads_face[2][n_q_points_face * dim + 1];

    if constexpr(compute_exterior)
    {
      read_face_values<degree + 1>(src.begin(),
                                   f + (f % 2 ? -1 : 1),
                                   dof_indices_neighbor,
                                   face_array);
      // For faces located on MPI-remote processes, we replace the dummy
      // data read out above by the data sent to us
      for(unsigned int v = 0; v < n_lanes; ++v)
        if(mpi_exchange_data_on_faces[cell][f][v] != numbers::invalid_unsigned_int)
          for(unsigned int i = 0; i < 2 * dofs_per_face; ++i)
          {
            face_array[i][v] = import_values[mpi_exchange_data_on_faces[cell][f][v] + i];
          }

      eval.template values<0, true, false>(face_array, scratch_data);
      eval.template values<1, true, false>(scratch_data, values_face[1]);
      eval.template values<0, true, false>(face_array + nn * nn, scratch_data);
      eval.template values<1, true, false, dim>(scratch_data, grads_face[1] + face_direction);
    }

    // interpolate from interior to face
    const std::array<Number, 2> * shape = interpolate_quad_to_boundary[f % 2].data();

    for(unsigned int i1 = 0; i1 < n_q_points_1d; ++i1)
      for(unsigned int i0 = 0; i0 < n_q_points_1d; ++i0)
      {
        unsigned int idx = 0;
        if(face_direction == 0)
          idx = n_q_points_1d * (n_q_points_1d * i1 + i0);
        else if(face_direction == 1)
          idx = i0 + i1 * n_q_points_1d * n_q_points_1d;
        else
          idx = i0 + i1 * n_q_points_1d;

        const VectorizedArray<Number> * my_vals = quad_values + idx;
        VectorizedArray<Number>         v0      = shape[0][0] * my_vals[0];
        VectorizedArray<Number>         d0      = shape[0][1] * my_vals[0];
        const unsigned int              stride  = Utilities::pow(n_q_points_1d, face_direction);
        for(unsigned int i = 1; i < n_q_points_1d; ++i)
        {
          v0 += shape[i][0] * my_vals[i * stride + 0];
          d0 += shape[i][1] * my_vals[i * stride + 0];
        }
        const unsigned int val_idx   = i1 * n_q_points_1d + i0;
        values_face[0][val_idx]      = v0;
        const unsigned int deriv_idx = val_idx * dim + face_direction;
        grads_face[0][deriv_idx]     = d0;
      }

    // Only need some tangential derivative in xy plane, as the domain is
    // extruded in z direction
    if(face_direction < 2)
      for(unsigned int side = 0; side < (compute_exterior ? 2 : 1); ++side)
      {
        const unsigned int first  = face_direction == 0 ? 1 : 0;
        const unsigned int second = face_direction == 1 ? 2 : first + 1;
        eval_g.template gradients<0, true, false, dim>(values_face[side], grads_face[side] + first);
        // This is a logical contradiction to the above if statement, so
        // this is actually never executed, but we keep it for
        // completeness
        if(face_direction == 2)
          eval_g.template gradients<1, true, false, dim>(values_face[side],
                                                         grads_face[side] + second);
      }

    const Number                  h_z_inverse = mapping_info.h_z_inverse;
    const Number                  h_z         = mapping_info.h_z;
    const VectorizedArray<Number> sigmaF      = penalty_parameters[cell][f] * factor_lapl;

    if(face_direction < 2)
    {
      std::array<unsigned int, n_lanes> shifted_data_indices;
      std::array<unsigned int, n_lanes> shifted_data_indices_neigh;
      for(unsigned int v = 0; v < n_lanes; ++v)
      {
        const unsigned int idx          = mapping_info.face_mapping_data_index[cell][f][0][v];
        shifted_data_indices[v]         = idx * 2;
        const unsigned int neighbor_idx = mapping_info.face_mapping_data_index[cell][f][1][v];
        shifted_data_indices_neigh[v]   = neighbor_idx * 2;
      }

      for(unsigned int q1 = 0; q1 < n_q_points_1d; ++q1)
      {
        Tensor<1, 2, VectorizedArray<Number>> jac_x_normal[2];
        vectorized_load_and_transpose(2,
                                      &mapping_info.face_jxn_xy[q1][0],
                                      shifted_data_indices.data(),
                                      &jac_x_normal[0][0]);
        vectorized_load_and_transpose(2,
                                      &mapping_info.face_jxn_xy[q1][0],
                                      shifted_data_indices_neigh.data(),
                                      &jac_x_normal[1][0]);

        VectorizedArray<Number> JxW_xy;
        JxW_xy.gather(mapping_info.face_JxW_xy.data() + q1,
                      mapping_info.face_mapping_data_index[cell][f][0].data());

        for(unsigned int qz = 0, q = q1; qz < n_q_points_1d; ++qz, q += n_q_points_1d)
        {
          VectorizedArray<Number>       val[2]     = {values_face[0][q], values_face[1][q]};
          const VectorizedArray<Number> grad[2][2] = {
            {grads_face[0][q * dim], grads_face[0][q * dim + 1]},
            {grads_face[1][q * dim], grads_face[1][q * dim + 1]}};

          VectorizedArray<Number> normal_derivatives[2];
          for(unsigned int s = 0; s < (compute_exterior ? 2 : 1); ++s)
          {
            normal_derivatives[s] =
              jac_x_normal[s][0] * grad[s][0] + jac_x_normal[s][1] * grad[s][1];
          }

          if(!compute_exterior)
          {
            val[1]                = 0;
            normal_derivatives[1] = 0;
          }
          {
            val[1] = boundary_mask * val[0] + (Number(1.0) - std::abs(boundary_mask)) * val[1];
            normal_derivatives[1] = boundary_mask * normal_derivatives[0] +
                                    (Number(1.0) - std::abs(boundary_mask)) * normal_derivatives[1];
          }

          const VectorizedArray<Number> integrate_factor =
            (JxW_xy * h_z) * mapping_info.quad_weights_z[qz];

          // physical terms
          const VectorizedArray<Number> effective_factor =
            integrate_factor * make_vectorized_array<Number>(0.5 * factor_lapl);

          const auto viscous_value_flux =
            (integrate_factor * sigmaF) * (val[0] - val[1]) -
            effective_factor * (normal_derivatives[0] - normal_derivatives[1]);

          const auto viscous_gradient_flux = effective_factor * (val[1] - val[0]);

          // apply test functions via transpose of the above
          for(unsigned int d = 0; d < 2; ++d)
            grads_face[0][q * dim + d] = jac_x_normal[0][d] * viscous_gradient_flux;
          values_face[0][q] = viscous_value_flux;
          if(use_face_buffer)
          {
            for(unsigned int d = 0; d < 2; ++d)
              grads_face[1][q * dim + d] = jac_x_normal[1][d] * (-viscous_gradient_flux);
            values_face[1][q] = -viscous_value_flux;
          }
        }
      }
    }
    else
    {
      const Number normal_sign = (f % 2) ? 1 : -1;

      for(unsigned int q = 0; q < n_q_points_2d; ++q)
      {
        VectorizedArray<Number> JxW_xy;
        JxW_xy.gather(&mapping_info.cell_JxW_xy[q], mapping_info.mapping_data_index[cell].data());

        VectorizedArray<Number>       val[2]  = {values_face[0][q], values_face[1][q]};
        const VectorizedArray<Number> grad[2] = {grads_face[0][q * dim + 2],
                                                 grads_face[1][q * dim + 2]};

        VectorizedArray<Number> normal_derivatives[2];
        for(unsigned int s = 0; s < (compute_exterior ? 2 : 1); ++s)
          normal_derivatives[s] = (h_z_inverse * normal_sign) * grad[s];

        if(!compute_exterior)
        {
          val[1]                = 0;
          normal_derivatives[1] = 0;
        }

        // assume no boundary in z direction

        // physical terms
        const VectorizedArray<Number> effective_factor =
          JxW_xy * make_vectorized_array<Number>(0.5 * factor_lapl);
        const auto viscous_value_flux =
          (JxW_xy * sigmaF) * (val[0] - val[1]) -
          effective_factor * (normal_derivatives[0] + normal_derivatives[1]);

        const auto viscous_gradient_flux = effective_factor * (val[1] - val[0]);

        // transpose of evaluate part
        values_face[0][q]          = viscous_value_flux;
        grads_face[0][q * dim + 2] = normal_sign * h_z_inverse * viscous_gradient_flux;
        if(use_face_buffer)
        {
          values_face[1][q]          = -values_face[0][q];
          grads_face[1][q * dim + 2] = grads_face[0][q * dim + 2];
        }
      }
    }

    if(face_direction < 2)
      for(unsigned int s = 0; s < ((use_face_buffer && f % 2) ? 2 : 1); ++s)
      {
        const unsigned int first  = face_direction == 0 ? 1 : 0;
        const unsigned int second = face_direction == 1 ? 2 : first + 1;
        eval_g.template gradients<0, false, true, dim>(grads_face[s] + first, values_face[s]);
        if(face_direction == 2)
          eval_g.template gradients<1, false, true, dim>(grads_face[s] + second, values_face[s]);
      }

    // fill flux buffer
    if(use_face_buffer && f % 2 == 1)
    {
      vectorized_transpose_and_store(false,
                                     n_q_points_face,
                                     values_face[1],
                                     face_flux_buffer_index[cell][f].data(),
                                     face_flux_buffer.data());
      for(unsigned int q = 0; q < n_q_points_face; ++q)
        values_face[1][q] = grads_face[1][q * dim + face_direction];
      vectorized_transpose_and_store(false,
                                     n_q_points_face,
                                     values_face[1],
                                     face_flux_buffer_index[cell][f].data(),
                                     face_flux_buffer.data() + n_q_points_face);
    }
    if(use_face_buffer && all_left_face_fluxes_from_buffer[cell][face_direction])
    {
      vectorized_load_and_transpose(n_q_points_face,
                                    face_flux_buffer.data() + n_q_points_face,
                                    face_flux_buffer_index[cell][f - 1].data(),
                                    values_face[1]);
      for(unsigned int q = 0; q < n_q_points_face; ++q)
        grads_face[1][q * dim + face_direction] = values_face[1][q];
      vectorized_load_and_transpose(n_q_points_face,
                                    face_flux_buffer.data(),
                                    face_flux_buffer_index[cell][f - 1].data(),
                                    values_face[1]);
    }

    // interpolate from face to interior
    const unsigned int stride = Utilities::pow(n_q_points_1d, face_direction);
    dealii::ndarray<VectorizedArray<Number>, n_q_points_1d, 2> shape_vec;
    for(unsigned int i = 0; i < n_q_points_1d; ++i)
    {
      shape_vec[i][0] = shape[i][0];
      shape_vec[i][1] = shape[i][1];
    }
    if(use_face_buffer && f % 2 == 1 && all_left_face_fluxes_from_buffer[cell][face_direction])
      for(unsigned int i1 = 0; i1 < n_q_points_1d; ++i1)
        for(unsigned int i0 = 0; i0 < n_q_points_1d; ++i0)
        {
          unsigned int idx;
          if(face_direction == 0)
            idx = n_q_points_1d * (n_q_points_1d * i1 + i0);
          else if(face_direction == 1)
            idx = i0 + i1 * n_q_points_1d * n_q_points_1d;
          else
            idx = i0 + i1 * n_q_points_1d;

          VectorizedArray<Number> *     vals      = out_values + idx;
          const unsigned int            val_idx   = i1 * n_q_points_1d + i0;
          const unsigned int            deriv_idx = val_idx * dim + face_direction;
          const VectorizedArray<Number> v0        = values_face[0][val_idx];
          const VectorizedArray<Number> d0        = grads_face[0][deriv_idx];
          const VectorizedArray<Number> vo0       = values_face[1][val_idx];
          const VectorizedArray<Number> do0       = grads_face[1][deriv_idx];
          for(unsigned int i = 0; i < n_q_points_1d; ++i)
            vals[i * stride] += shape_vec[i][0] * v0 + shape_vec[i][1] * d0 +
                                shape_vec[n_q_points_1d - 1 - i][0] * vo0 -
                                shape_vec[n_q_points_1d - 1 - i][1] * do0;
        }
    else
      for(unsigned int i1 = 0; i1 < n_q_points_1d; ++i1)
        for(unsigned int i0 = 0; i0 < n_q_points_1d; ++i0)
        {
          unsigned int idx;
          if(face_direction == 0)
            idx = n_q_points_1d * (n_q_points_1d * i1 + i0);
          else if(face_direction == 1)
            idx = i0 + i1 * n_q_points_1d * n_q_points_1d;
          else
            idx = i0 + i1 * n_q_points_1d;

          VectorizedArray<Number> *     vals      = out_values + idx;
          const unsigned int            val_idx   = i1 * n_q_points_1d + i0;
          const VectorizedArray<Number> v0        = values_face[0][val_idx];
          const unsigned int            deriv_idx = val_idx * dim + face_direction;
          const VectorizedArray<Number> d0        = grads_face[0][deriv_idx];
          for(unsigned int i = 0; i < n_q_points_1d; ++i)
            vals[i * stride] += shape_vec[i][0] * v0 + shape_vec[i][1] * d0;
        }
  }

  template<int degree>
  void
  do_prolongate(const unsigned int                     cell,
                const VectorType &                     fe_correction,
                const LaplaceOperatorFE<dim, Number> & fe_operator,
                VectorType &                           dg_solution) const
  {
    constexpr unsigned int  dofs_per_cell = Utilities::pow(degree + 1, dim);
    VectorizedArray<Number> values_dofs[dofs_per_cell];
    fe_operator.template read_dof_values<degree>(cell, fe_correction, values_dofs);
    vectorized_transpose_and_store(
      true, dofs_per_cell, values_dofs, dof_indices[cell].data(), dg_solution.begin());
  }

  template<int degree>
  void
  do_body_force_on_cell(const unsigned int            cell,
                        const dealii::Function<dim> & rhs_function,
                        VectorType &                  dst) const
  {
    constexpr unsigned int n_q_points_1d = degree + 1;
    constexpr unsigned int nn            = n_q_points_1d;
    constexpr unsigned int n_points      = Utilities::pow(n_q_points_1d, dim);

    VectorizedArray<Number> quad_values[dim * n_points],
      body_force_z[Utilities::pow(n_q_points_1d, dim - 1)];

    std::array<unsigned int, n_lanes> shifted_data_indices;
    for(unsigned int v = 0; v < n_lanes; ++v)
      shifted_data_indices[v] = mapping_info.mapping_data_index[cell][v] * 4;

    const Number h_z         = mapping_info.h_z;
    const Number h_z_inverse = mapping_info.h_z_inverse;

    for(unsigned int qy = 0, q1 = 0; qy < nn; ++qy)
    {
      for(unsigned int qx = 0; qx < nn; ++qx, ++q1)
      {
        Tensor<2, 2, VectorizedArray<Number>> inv_jac_xy;
        vectorized_load_and_transpose(4,
                                      &mapping_info.inv_jacobians_xy[q1][0][0],
                                      shifted_data_indices.data(),
                                      &inv_jac_xy[0][0]);
        VectorizedArray<Number> JxW_xy;
        JxW_xy.gather(&mapping_info.cell_JxW_xy[q1], mapping_info.mapping_data_index[cell].data());

        // Assume body force does not depend on z direction
        Tensor<1, dim, VectorizedArray<Number>> body_force;
        for(unsigned int v = 0; v < n_lanes; ++v)
        {
          Point<dim> point;
          for(unsigned int d = 0; d < 2; ++d)
            point[d] =
              mapping_info.quadrature_points[mapping_info.mapping_data_index[cell][v] + q1][d];
          for(unsigned int d = 0; d < dim; ++d)
            body_force[d][v] = rhs_function.value(point, d);
        }

        // keep z component for face integral later
        body_force_z[q1] = body_force[dim - 1];

        if constexpr(dim == 2)
        {
          const VectorizedArray<Number> s0 =
            inv_jac_xy[0][0] * body_force[0] + inv_jac_xy[1][0] * body_force[1];
          const VectorizedArray<Number> s1 =
            inv_jac_xy[0][1] * body_force[0] + inv_jac_xy[1][1] * body_force[1];

          quad_values[q1 * dim]     = s0 * JxW_xy;
          quad_values[q1 * dim + 1] = s1 * JxW_xy;
        }
        else // now to dim == 3
        {
          for(unsigned int qz = 0, q = q1; qz < nn; ++qz, q += nn * nn)
          {
            const Number                  weight_z = mapping_info.quad_weights_z[qz];
            const VectorizedArray<Number> factor   = JxW_xy * h_z * weight_z;

            VectorizedArray<Number> val_real[dim], value_tmp[dim];
            for(unsigned int d = 0; d < dim; ++d)
              val_real[d] = body_force[d] * factor;
            for(unsigned int d = 0; d < 2; ++d)
              value_tmp[d] = inv_jac_xy[0][d] * val_real[0] + inv_jac_xy[1][d] * val_real[1];
            value_tmp[2] = h_z_inverse * val_real[2];

            quad_values[q + 0 * n_points] = value_tmp[0];
            quad_values[q + 1 * n_points] = value_tmp[1];
            quad_values[q + 2 * n_points] = value_tmp[2];
          }
        }
      }
    }

    {
      AssertThrow(shape_info.data[0].n_q_points_1d == n_q_points_1d, ExcInternalError());
      internal::EvaluatorTensorProduct<internal::evaluate_evenodd,
                                       dim,
                                       n_q_points_1d,
                                       n_q_points_1d,
                                       VectorizedArray<Number>,
                                       Number>
        eval_g({}, shape_info.data[0].shape_gradients_collocation_eo.data(), {});
      eval_g.template gradients<0, false, false, 1>(quad_values, quad_values);
      eval_g.template gradients<1, false, true, 1>(quad_values + 1 * n_points, quad_values);
      if constexpr(dim == 3)
        eval_g.template gradients<2, false, true, 1>(quad_values + 2 * n_points, quad_values);
    }

    for(unsigned int face = 0; face < 4; ++face)
    {
      const unsigned int face_direction = face / 2;

      VectorizedArray<Number> boundary_mask;
      for(unsigned int v = 0; v < n_lanes; ++v)
        if(neighbor_cells[cell][face][v] == numbers::invalid_unsigned_int)
          boundary_mask[v] = 0.;
        else
          boundary_mask[v] = 1.;

      std::array<unsigned int, n_lanes> shifted_data_indices, data_indices;
      for(unsigned int v = 0; v < n_lanes; ++v)
      {
        const unsigned int idx  = mapping_info.face_mapping_data_index[cell][face][0][v];
        shifted_data_indices[v] = idx * 4;
        data_indices[v]         = idx;
      }

      const std::array<Number, 2> * shape = interpolate_quad_to_boundary[face % 2].data();

      for(unsigned int q1 = 0; q1 < nn; ++q1)
      {
        Tensor<2, 2, VectorizedArray<Number>> jac_xy;
        vectorized_load_and_transpose(4,
                                      &mapping_info.face_jacobians_xy[q1][0][0],
                                      shifted_data_indices.data(),
                                      &jac_xy[0][0]);

        const VectorizedArray<Number> area_element_xy =
          std::sqrt(jac_xy[0][1 - face_direction] * jac_xy[0][1 - face_direction] +
                    jac_xy[1][1 - face_direction] * jac_xy[1][1 - face_direction]);
        const Number h_z = mapping_info.h_z;

        VectorizedArray<Number> val;
        for(unsigned int v = 0; v < n_lanes; ++v)
        {
          Point<dim> point;
          for(unsigned int d = 0; d < 2; ++d)
            point[d] = mapping_info.face_quadrature_points[data_indices[v] + q1][d];
          val[v] = 0;
          for(unsigned int e = 0; e < 2; ++e)
            val[v] += rhs_function.value(point, e) *
                      mapping_info.face_normal_vector_xy[data_indices[v] + q1][e];
        }

        for(unsigned int qz = 0, q = q1; qz < nn; ++qz, q += nn)
        {
          const VectorizedArray<Number> flux_term =
            (mapping_info.quad_weights_z[q1] * area_element_xy * h_z) *
            mapping_info.quad_weights_z[qz] * val * boundary_mask;

          unsigned int idx = 0;
          if(face_direction == 0)
            idx = n_q_points_1d * (n_q_points_1d * qz + q1);
          else // if(face_direction == 1)
            idx = q1 + qz * n_q_points_1d * n_q_points_1d;
          const unsigned int stride = Utilities::pow(n_q_points_1d, face_direction);
          for(unsigned int i = 0; i < n_q_points_1d; ++i)
          {
            quad_values[idx + i * stride] -= shape[i][0] * flux_term;
          }
        }
      }
    }
    for(unsigned int face = 4; face < 2 * dim; ++face)
    {
      VectorizedArray<Number> boundary_mask;
      for(unsigned int v = 0; v < n_lanes; ++v)
        if(neighbor_cells[cell][face][v] == numbers::invalid_unsigned_int)
          boundary_mask[v] = 0.;
        else
          boundary_mask[v] = 1.;
      const std::array<Number, 2> * shape = interpolate_quad_to_boundary[face % 2].data();
      for(unsigned int q = 0; q < nn * nn; ++q)
      {
        VectorizedArray<Number> JxW_xy;
        JxW_xy.gather(&mapping_info.cell_JxW_xy[q], mapping_info.mapping_data_index[cell].data());
        VectorizedArray<Number> flux_term = body_force_z[q];
        flux_term *= JxW_xy * mapping_info.quad_weights_xy[q] * boundary_mask;
        const unsigned int stride = nn * nn;
        for(unsigned int i = 0; i < n_q_points_1d; ++i)
        {
          quad_values[q + i * stride] -= shape[i][0] * flux_term;
        }
      }
    }

    integrate_cell_scatter<degree + 1, n_q_points_1d>(dof_indices[cell],
                                                      n_active_entries_per_cell_batch(cell),
                                                      quad_values,
                                                      dst.begin());
  }

  unsigned int
  find_first_zero_bit(const std::uint64_t x)
  {
    std::uint64_t y = ~x;
    if(y == 0)
      return 64;

      // count number of trailing zero bits
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(y);
#else

    unsigned int pos = 0;
    while((y & 1) == 0)
    {
      y >>= 1;
      ++pos;
    }
    return pos;

#endif
  }

  void
  detect_dependencies_of_face_integrals()
  {
    Timer                      time;
    const Triangulation<dim> & tria = dof_handler->get_triangulation();

    std::vector<int>   touch_length(tria.n_raw_faces(), 0);
    const unsigned int n_cell_batches = dof_indices.size();
    for(unsigned int c = 0, count = 1; c < n_cell_batches; ++c)
      for(unsigned int v = 0; v < n_active_entries_per_cell_batch(c); ++v)
      {
        const auto cell = get_cell_iterator(c, v);
        for(unsigned int f = 0; f < 2 * dim; ++f)
        {
          const unsigned int face_idx = cell->face(f)->index();
          const unsigned int periodic_idx =
            (cell->at_boundary(f) && cell->has_periodic_neighbor(f)) ?
              cell->periodic_neighbor(f)->face(cell->periodic_neighbor_face_no(f))->index() :
              face_idx;
          if(touch_length[face_idx] == 0)
          {
            touch_length[face_idx]     = -count;
            touch_length[periodic_idx] = -count;
          }
          else
          {
            AssertThrow(touch_length[face_idx] < 0, ExcInternalError());
            touch_length[face_idx]     = count + touch_length[face_idx];
            touch_length[periodic_idx] = touch_length[face_idx];
          }
        }
        ++count;
      }

#if 0
    // Some debugging to check the distances of faces from first to last touch
    std::map<unsigned int, unsigned int> distances;
    for (const int a : touch_length)
      if (a > 0)
        {
          if (distances.find(a) == distances.end())
            distances[a] = 1;
          else
            ++distances[a];
        }
    std::cout << "faces: ";
    for (const auto it : distances)
      std::cout << it.first << "|" << it.second << "  ";
    std::cout << std::endl;
#endif

    face_flux_buffer_index.resize(dof_indices.size());
    all_left_face_fluxes_from_buffer.resize(dof_indices.size());
    const unsigned int n_data_per_face =
      Utilities::pow(mapping_info.quad_weights_z.size(), dim - 1) * 2;

    constexpr unsigned int     long_range_start = 16;
    std::vector<std::uint64_t> face_storage(1);
    std::vector<unsigned int>  face_index_in_storage(tria.n_raw_faces(), 0);
    unsigned int               size_of_storage = 1;
    for(unsigned int c = 0; c < n_cell_batches; ++c)
      for(unsigned int d = 0; d < dim; ++d)
      {
        for(unsigned int v = 0; v < n_active_entries_per_cell_batch(c); ++v)
        {
          const auto cell = get_cell_iterator(c, v);

          // For cells at the right boundary (including those adjacent to
          // remote MPI processes), set the index to 0 to indicate a sink
          // that will receive vectorized writes but not get picked up
          // again later. This is one single array and thus acceptable.
          if(cell->at_boundary(2 * d + 1) ||
             cell->neighbor(2 * d + 1)->subdomain_id() != cell->subdomain_id())
          {
            face_flux_buffer_index[c][2 * d + 1][v] = 0;
            continue;
          }
          const unsigned int face_idx = cell->face(2 * d + 1)->index();

          const auto add_entry = [&](const unsigned int position) {
            const unsigned int entry_within_vector = position / 64;
            const unsigned int bit_within_entry    = position % 64;

            // flag current bit as 'occupied'
            face_storage[entry_within_vector] |= std::uint64_t(1) << bit_within_entry;
            face_index_in_storage[face_idx]         = position + 1;
            face_flux_buffer_index[c][2 * d + 1][v] = (position + 1) * n_data_per_face;
            size_of_storage                         = std::max(size_of_storage, position + 2);
          };

          // Only short-distance faces are allowed in the first part of
          // the vector to ensure we have a high cache locality in some
          // part of the final data vector for the flux
          // buffer. Furthermore, the complexity of the linear search we
          // do in this array does not grow too large.
          if(touch_length[face_idx] < 40)
          {
            const unsigned int pos = find_first_zero_bit(face_storage[0]);
            if(pos < 64)
            {
              add_entry(pos);
              continue;
            }
          }
          else
          {
            // should only look in later part
            unsigned int pos = find_first_zero_bit(face_storage[0] >> long_range_start);
            pos += long_range_start;
            if(pos < 64)
            {
              add_entry(pos);
              continue;
            }
          }
          // if we did not find it in the first part, we need to perform a
          // linear search in the storage array, which we accelerate by
          // the bit-processing function
          bool added_this = false;
          for(unsigned int s = 1; s < face_storage.size(); ++s)
          {
            const unsigned int pos = find_first_zero_bit(face_storage[s]);
            if(pos < 64)
            {
              add_entry(pos + 64 * s);
              added_this = true;
              break;
            }
          }
          if(!added_this)
          {
            const unsigned int pos = face_storage.size() * 64;
            face_storage.emplace_back(1);
            face_index_in_storage[face_idx]         = pos + 1;
            face_flux_buffer_index[c][2 * d + 1][v] = (pos + 1) * n_data_per_face;
            size_of_storage += 1;
          }
        }
        bool face_missing = false;
        for(unsigned int v = 0; v < n_active_entries_per_cell_batch(c); ++v)
        {
          const auto         cell     = get_cell_iterator(c, v);
          const unsigned int face_idx = cell->face(2 * d)->index();
          if(face_index_in_storage[face_idx] > 0)
          {
            AssertIndexRange(face_index_in_storage[face_idx], size_of_storage);
            face_flux_buffer_index[c][2 * d][v] = face_index_in_storage[face_idx] * n_data_per_face;
            const unsigned int index_in_storage = (face_index_in_storage[face_idx] - 1) / 64;
            const unsigned int bit_index        = (face_index_in_storage[face_idx] - 1) % 64;
            face_storage[index_in_storage] &= ~(std::uint64_t(1) << bit_index);
          }
          else
            face_missing = true;
        }
        all_left_face_fluxes_from_buffer[c][d] = (face_missing == false);
      }
    face_flux_buffer.resize(n_data_per_face * size_of_storage);

    unsigned int count_0 = 0, count_1 = 0;
    for(const auto & a : all_left_face_fluxes_from_buffer)
      for(const char b : a)
        if(b == 0)
          ++count_0;
        else
          ++count_1;

    // debug output to assess sanity of implementation
    if(false)
      Helper::print_time(static_cast<double>(count_1) / (count_0 + count_1),
                         "Face flux buffer efficiency",
                         MPI_COMM_WORLD);

#if 0
    std::cout << "Identified storage of length " << size_of_storage << " with "
              << face_flux_buffer.size() * sizeof(Number) << " bytes, quality: "
              << count_1 << " of " << (count_0 + count_1) << std::endl;

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "Time analyze dependencies: " << time.wall_time() << std::endl;
#endif
  }
};


} // namespace LaplaceOperator
