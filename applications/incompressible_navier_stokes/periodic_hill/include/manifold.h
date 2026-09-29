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

#ifndef APPLICATIONS_PERIODIC_HILL_MANIFOLD_H_
#define APPLICATIONS_PERIODIC_HILL_MANIFOLD_H_

namespace ExaDG
{
double
m_to_mm(double coordinate)
{
  return 1000.0 * coordinate;
}

double
mm_to_m(double coordinate)
{
  return 0.001 * coordinate;
}

/*
 * This function returns the distance by which points are shifted in y-direction due to the hill,
 * i.e., a value of 0 is returned at x=0H, 9H, and a value of -H at x=4.5H.
 */
double
f(double x_m, double const H, double const LENGTH)
{
  if(x_m > LENGTH / 2.0)
    x_m = LENGTH - x_m;

  double x = m_to_mm(x_m);
  double y = 0.0;

  AssertThrow(x_m > -1e-12 and x_m <= LENGTH / 2.0 + 1.e-12,
              dealii::ExcMessage("Parameter out of bounds."));

  if(x <= 9.0)
    y =
      -m_to_mm(H) +
      std::min(m_to_mm(H), m_to_mm(H) + 6.775070969851e-3 * x * x - 2.124527775800e-3 * x * x * x);
  else if(x > 9.0 and x <= 14.0)
    y = -m_to_mm(H) + 2.507355893131e1 + 9.754803562315e-1 * x - 1.016116352781e-1 * x * x +
        1.889794677828e-3 * x * x * x;
  else if(x > 14.0 and x <= 20.0)
    y = -m_to_mm(H) + 2.579601052357e1 + 8.206693007457e-1 * x - 9.055370274339e-2 * x * x +
        1.626510569859e-3 * x * x * x;
  else if(x > 20.0 and x <= 30.0)
    y = -m_to_mm(H) + 4.046435022819e1 - 1.379581654948 * x + 1.945884504128e-2 * x * x -
        2.070318932190e-4 * x * x * x;
  else if(x > 30.0 and x <= 40.0)
    y = -m_to_mm(H) + 1.792461334664e1 + 8.743920332081e-1 * x - 5.567361123058e-2 * x * x +
        6.277731764683e-4 * x * x * x;
  else if(x > 40.0 and x <= 54.0)
    y = -m_to_mm(H) + std::max(0.0,
                               5.639011190988e1 - 2.010520359035 * x + 1.644919857549e-2 * x * x +
                                 2.674976141766e-5 * x * x * x);
  else if(x > 54.0)
    y = -m_to_mm(H);
  else
    AssertThrow(false, dealii::ExcMessage("Not implemented."));

  return mm_to_m(y);
}



/**
 * Manifold to describe the periodic hill geometry, using two distinctive
 * features for the mesh:
 * (1) The mesh is graded towards the boundary with a tanh profile with
 * scaling determined by @p GRID_STRETCH_FACTOR
 * (2) It is possible to manually deform the mesh to improve the density of
 * mesh elements in regions with higher dissipation. This is purely
 * hand-written and might not optimally place points. As an alternative that
 * has been systematically optimized for the actual flow, use the class
 * PeriodicHillManifoldOptimizedMesh instead.
 */
template<int dim>
class PeriodicHillManifoldTanh : public dealii::ChartManifold<dim>
{
public:
  PeriodicHillManifoldTanh(const double H,
                           const double LENGTH,
                           const double HEIGHT,
                           const double GRID_STRETCH_FAC,
                           const bool   apply_manual_deformation)
    : dealii::ChartManifold<dim>(),
      H(H),
      LENGTH(LENGTH),
      HEIGHT(HEIGHT),
      GRID_STRETCH_FAC(GRID_STRETCH_FAC),
      apply_manual_deformation(apply_manual_deformation)
  {
    // Assume we have a slope of -2 (the actual slope extremum is around
    // -0.86, but due to resolution requirements near the hill top choose a
    // higher value), and then transition from a steeper path of curve length
    // to the more flat part. Start by defining a piecewise linear function (a
    // more complicated function to check the actual path was tried but found
    // to not be better but just more expensive) and then transition over a
    // length corresponding to the hill height to the flat part; the actual
    // evaluation will use a quintic Hermite interpolating polynomial.
    const double slope_assumed        = -2;
    const double scaling_curved_start = 0.7 * H;
    const double scaling_curved_end   = 1.8 * H;
    const double curve_length_sloped  = std::sqrt(1. + slope_assumed * slope_assumed);
    const double curve_length =
      0.5 * (scaling_curved_start + scaling_curved_end) * (curve_length_sloped - 1.) + LENGTH / 2;
    x_transition_points = {
      {0.0,
       curve_length_sloped * scaling_curved_start * LENGTH * 0.5 / curve_length,
       LENGTH * 0.5 - (LENGTH * 0.5 - scaling_curved_end) * LENGTH * 0.5 / curve_length,
       LENGTH * 0.5}};
    x_transition_values = {{0.0, scaling_curved_start, scaling_curved_end, LENGTH * 0.5}};
  }

  dealii::Point<dim>
  push_forward(const dealii::Point<dim> & xi_in) const final
  {
    dealii::Point<dim> xi = xi_in;

    const double gamma = GRID_STRETCH_FAC;
    const double y     = 2.0 * (xi[1] - H) / HEIGHT - 1.0;

    if(apply_manual_deformation)
    {
      // Shift elements towards the lower part of the domain
      const double xi_1_hat =
        std::tanh(gamma * (y - 0.08 * std::sin(dealii::numbers::PI * y))) / std::tanh(gamma);
      xi[1] = xi_1_hat * 0.5 + 0.5;

      // Create shift of mesh elements towards the left, where dissipation is
      // higher in the wake of the hill. The idea is to have the element
      // spacing between 5/32 and 14/32 to be 0.7 the size of a uniform
      // distribution. Along the x coordinate, we transition with C^3
      // continuity between the initial size of 1 and then back to a remainder
      // size, which is 19/15 the size of the initial elements. The obtained x
      // shift is further weighted in y direction to not act near the
      // boundaries with 'y_scaling'
      const double x_unit  = xi_in[0] / LENGTH;
      double       x_shift = 0;
      if(x_unit > 2. / 32)
      {
        if(x_unit < 5. / 32.)
        {
          const double t = 32. / 3. * (x_unit - 2. / 32);
          x_shift        = -9. / 320 * t * t * t * t * (2.5 + t * (-3. + t));
        }
        else if(x_unit < 14. / 32.)
        {
          x_shift = 21. / 640 - 0.3 * x_unit;
        }
        else if(x_unit < 17. / 32)
        {
          const double t = 32. / 3. * (x_unit - 14. / 32.);
          x_shift = 21. / 640 - 0.3 * x_unit + 9. / 320 * t * t * t * t * (2.5 + t * (-3. + t));
        }
        else if(x_unit < 20. / 32)
        {
          const double t = 32. / 3. * (x_unit - 17. / 32.);
          x_shift        = -18. / 160 + 1. / 40 * t * t * t * t * (2.5 + t * (-3. + t));
        }
        else
          x_shift = -128. / 480 + 4. / 15 * x_unit;
      }
      double y_scaling = 0;
      if(xi_1_hat < -0.24)
      {
        const double t = (xi_1_hat + 1) * (1. / 0.76);
        y_scaling      = t * t * (3. - 2 * t);
      }
      else
      {
        const double t = (1. - xi_1_hat) * (1. / 1.24);
        y_scaling      = t * t * (3 - 2 * t);
      }
      xi[0] += 0.8 * LENGTH * x_shift * y_scaling;
    }
    else
    {
      const double xi_1_hat = std::tanh(gamma * y) / std::tanh(gamma);
      xi[1]                 = xi_1_hat * 0.5 + 0.5;
    }

    // Finally move the points near the boundary to ensure approximately
    // equal-length elements (apart from at the hill top, where we want finer
    // element distribution). At this point, xi[1] holds the weight in [0, 1]
    // for blending between the bottom and top walls.
    const double xi_0 =
      xi[0] < LENGTH / 2 ? get_scaled_x_point(xi[0]) : LENGTH - get_scaled_x_point(LENGTH - xi[0]);

    dealii::Point<dim> xi_bottom;
    xi_bottom[0] = xi_0;
    xi_bottom[1] = H + f(xi_0, H, LENGTH);
    if(dim > 2)
      xi_bottom[2] = xi[2];

    dealii::Point<dim> xi_top = xi;
    xi_top[1]                 = H + HEIGHT;
    return xi_top * xi[1] + xi_bottom * (1.0 - xi[1]);
  }

  dealii::Point<dim>
  pull_back(const dealii::Point<dim> & x) const final
  {
    AssertThrow(false, ExcNotImplemented());
    return x;
  }

  std::unique_ptr<dealii::Manifold<dim>>
  clone() const final
  {
    return std::make_unique<PeriodicHillManifoldTanh<dim>>(
      H, LENGTH, HEIGHT, GRID_STRETCH_FAC, apply_manual_deformation);
  }

  double
  get_scaled_x_point(const double x_in) const
  {
    if(x_in >= x_transition_points[2])
      return (x_transition_values[2] * (x_transition_points[3] - x_in) +
              x_transition_values[3] * (x_in - x_transition_points[2])) /
             (x_transition_points[3] - x_transition_points[2]);
    else if(x_in < x_transition_points[1])
      return (x_transition_values[1] * x_in) / x_transition_points[1];
    else
    {
      // evaluate Hermite interpolating polynomial between
      // transition_points[1] and transition_points[2]
      const double x0  = x_transition_points[1];
      const double x1  = x_transition_points[2];
      const double y0  = x_transition_values[1];
      const double y1  = x_transition_values[2];
      const double yp0 = (x_transition_values[1] - x_transition_values[0]) /
                         (x_transition_points[1] - x_transition_points[0]);
      const double yp1 = (x_transition_values[3] - x_transition_values[2]) /
                         (x_transition_points[3] - x_transition_points[2]);
      const double t = (x_in - x0) / (x1 - x0);

      // terms for cubic polynomial
      // const double h00 = (1 + 2*t) * (1 - t) * (1 - t);
      // const double h10 = t * (1 - t) * (1 - t);
      // const double h01 = t * t * (3 - 2*t);
      // const double h11 = t * t * (t - 1);

      // define a quintic polynomial with second derivative zero at the end
      // points, so ignore the part involving second derivatives, which
      // leads to four contributions similar to the cubic Hermite polynomial
      const double h00 = 1. + t * t * t * (-10. + t * (15. - 6. * t));
      const double h01 = t + t * t * t * (-6. + t * (8. - 3. * t));
      const double h10 = t * t * t * (10. + t * (-15. + 6. * t));
      const double h11 = t * t * t * (-4. + t * (7. - 3. * t));
      return h00 * y0 + h10 * y1 + (x1 - x0) * (h01 * yp0 + h11 * yp1);
    }
  }

private:
  const double          H, LENGTH, HEIGHT, GRID_STRETCH_FAC;
  const bool            apply_manual_deformation;
  std::array<double, 4> x_transition_points;
  std::array<double, 4> x_transition_values;
};



/**
 * Alternative to the basic PeriodicHillManifoldTanh class, where the mesh
 * displacement is based on an optimization of the mesh. The actual
 * computation was done for a 64 x 48 x 32 mesh of the periodic hill
 * at Reynolds number 10595 with element FE_RaviartThomasNodal(6)
 * (degree 7/6 in normal/tangential part) using statistics of 10
 * flow-through times in the developed phase, and aims to balance (h / eta)^2
 * with the Kolmogorov length eta and mesh size h = vol(K)^(1/3)
 * with the fe_values.jacobian(q).determinant() quantity in the
 * 2D x-y slice of the mesh using a linear elasticity problem with `mu = (h
 * / eta)^2 * fe_values.jacobian(q).determinant()` and `lambda = -0.33 * mu`
 * (almost pure shear problem).
 */
template<int dim>
class PeriodicHillManifoldOptimizedMesh : public dealii::ChartManifold<dim>
{
  // This is the degree of the optimization computation performed, which uses
  // two panels for the left and right half of the hill, respectively (the
  // final distribution is non-symmetric, so use two separate functions).
  static constexpr unsigned int n_shapes = 7;

public:
  PeriodicHillManifoldOptimizedMesh(const double H,
                                    const double LENGTH,
                                    const double HEIGHT,
                                    const double GRID_STRETCH_FAC)
    : dealii::ChartManifold<dim>(),
      H(H),
      LENGTH(LENGTH),
      HEIGHT(HEIGHT),
      GRID_STRETCH_FAC(GRID_STRETCH_FAC),
      INVERSE_LENGTH(1.0 / LENGTH),
      INVERSE_HEIGHT(1.0 / HEIGHT),
      tanh_gamma(std::tanh(GRID_STRETCH_FAC)),
      deriv_tanh_gamma(GRID_STRETCH_FAC /
                       (tanh_gamma * std::cosh(GRID_STRETCH_FAC) * std::cosh(GRID_STRETCH_FAC)))
  {
    // Assume we have a slope of -2 (the actual slope extremum is around
    // -0.86, but due to resolution requirements near the hill top choose a
    // higher value) that transitions from a steeper part to a flatter part.
    // Start by defining a piecewise linear function (a more complicated
    // function to check the actual path was tried, but not found to perform
    // better at a higher cost) and then transition over a length corresponding
    // to the hill height to the flat part; the actual evaluation will use a
    // Hermite interpolating polynomial of degree 7, see below.
    const double slope_assumed        = -2;
    const double scaling_curved_start = 0.7 * H;
    const double scaling_curved_end   = 1.8 * H;
    const double curve_length_sloped  = std::sqrt(1. + slope_assumed * slope_assumed);
    const double curve_length =
      0.5 * (scaling_curved_start + scaling_curved_end) * (curve_length_sloped - 1.) + LENGTH / 2;
    const double x_transition_point_1 =
      curve_length_sloped * scaling_curved_start * LENGTH * 0.5 / curve_length;
    const double x_transition_point_2 =
      LENGTH * 0.5 - (LENGTH * 0.5 - scaling_curved_end) * LENGTH * 0.5 / curve_length;
    const double x_transition_point_3 = LENGTH * 0.5;
    const double x_transition_value_1 = scaling_curved_start;
    const double x_transition_value_2 = scaling_curved_end;
    const double x_transition_value_3 = x_transition_point_3;

    // We use a single 7-th order Hermite polynomial per half length to place
    // the points more densely along the sloped part. The function has zero
    // second derivative on the left and 4 continuous derivatives at x=0.5,
    // where we stitch the halves together. Specifically, the 8 conditions are:
    // f(0) = 0, f'(0) = d0, f''(0) = 0, f(1) = v1, f'(1) = d1, f''(1) = 0,
    // f'''(1) = 0, f''''(1) = 0.  And in the stored coefficients, we drop the
    // two coefficients with value 0.
    const double d0 = x_transition_value_1 / x_transition_point_1 * x_transition_value_3;
    const double v1 = x_transition_value_3;
    const double d1 = (x_transition_value_3 - x_transition_value_2) /
                      (x_transition_point_3 - x_transition_point_2) * x_transition_value_3;

    coefficients_hermite = {{d0,
                             35 * v1 - 20 * d1 - 15 * d0,
                             -105 * v1 + 65 * d1 + 40 * d0,
                             126 * v1 - 81 * d1 - 45 * d0,
                             46 * d1 - 70 * v1 + 24 * d0,
                             15 * v1 - 10 * d1 - 5 * d0}};

    // These coefficients were determined by the offline computation
    // underlying this class.
    const dealii::ndarray<double, n_shapes * n_shapes, 2> panel_left{
      {{{0, 0}},
       {{0.08488805186072, 0}},
       {{0.2655756032646, 0}},
       {{0.5, 0}},
       {{0.7344243967354, 0}},
       {{0.9151119481393, 0}},
       {{1, 0}},
       {{0, 0.0862239029776}},
       {{0.122449463918, 0.08475565616348}},
       {{0.3134130750993, 0.1277273712494}},
       {{0.5161135710844, 0.08528085829237}},
       {{0.7269076827709, 0.07475349773668}},
       {{0.8890713200363, 0.08286858751728}},
       {{0.9669753494535, 0.08895998328973}},
       {{0, 0.2395340672242}},
       {{0.1567898829495, 0.2015207589759}},
       {{0.3475648736373, 0.2255270208406}},
       {{0.5350414216554, 0.2000609619101}},
       {{0.7176881134132, 0.1975089174241}},
       {{0.8647450893527, 0.20346231199}},
       {{0.9344042279655, 0.2119361181576}},
       {{0, 0.412174915857}},
       {{0.156747180515, 0.3748162953837}},
       {{0.3700143749472, 0.3644255436685}},
       {{0.5422340590131, 0.3421917855631}},
       {{0.7142067668018, 0.3486475530682}},
       {{0.8508273942615, 0.3566253088925}},
       {{0.9153207792259, 0.365053764028}},
       {{0, 0.6095545998635}},
       {{0.1451760564156, 0.5902996742813}},
       {{0.3604543869669, 0.5601909811179}},
       {{0.5448217987142, 0.5476012197965}},
       {{0.7126694784277, 0.5549426208589}},
       {{0.8490665609672, 0.5617434021536}},
       {{0.9157871743652, 0.5658120853449}},
       {{0, 0.8506688932468}},
       {{0.1120038403225, 0.8518311328378}},
       {{0.3088727573184, 0.8526504630219}},
       {{0.5168709898329, 0.8612678292511}},
       {{0.7252553157077, 0.8724553309176}},
       {{0.8882268910253, 0.8711988777596}},
       {{0.9662340468823, 0.8724823126874}},
       {{0, 1}},
       {{0.08488805186072, 1}},
       {{0.2655756032646, 1}},
       {{0.5, 1}},
       {{0.7344243967354, 1}},
       {{0.9151119481393, 1}},
       {{1, 1}}}};
    const dealii::ndarray<double, n_shapes * n_shapes, 2> panel_right{
      {{{0, 0}},
       {{0.08488805186072, 0}},
       {{0.2655756032646, 0}},
       {{0.5, 0}},
       {{0.7344243967354, 0}},
       {{0.9151119481393, 0}},
       {{1, 0}},
       {{-0.03302465054651, 0.08895998328973}},
       {{0.04691580988036, 0.08862348622582}},
       {{0.2187412469525, 0.08741524628976}},
       {{0.445047568607, 0.09180405597729}},
       {{0.6639567001611, 0.1179170588696}},
       {{0.8846759872993, 0.1034990038377}},
       {{1, 0.0862239029776}},
       {{-0.06559577203452, 0.2119361181576}},
       {{0.007260341194406, 0.2136388439994}},
       {{0.1639816074322, 0.2202214424149}},
       {{0.3812584675338, 0.2274518303099}},
       {{0.604135471394, 0.2626953353172}},
       {{0.8516725562502, 0.2630417608455}},
       {{1, 0.2395340672242}},
       {{-0.08467922077408, 0.365053764028}},
       {{-0.01707566652446, 0.3710629699774}},
       {{0.1331873442952, 0.3870641262714}},
       {{0.3464694437837, 0.4025826196895}},
       {{0.5794758771119, 0.4322545207644}},
       {{0.8449325350412, 0.4355240694618}},
       {{1, 0.412174915857}},
       {{-0.08421282563484, 0.5658120853449}},
       {{-0.01554677551131, 0.5744901382617}},
       {{0.1374348520456, 0.5858613683461}},
       {{0.3564424845144, 0.604964757936}},
       {{0.6028603037661, 0.6148017303782}},
       {{0.8646451287138, 0.6292913936227}},
       {{1, 0.6095545998635}},
       {{-0.03376595311774, 0.8724823126874}},
       {{0.04546719729648, 0.8729789770786}},
       {{0.2138510363756, 0.8697006184888}},
       {{0.4424169579444, 0.8694820459734}},
       {{0.6785432468803, 0.8584143510768}},
       {{0.8980036122317, 0.8702676427422}},
       {{1, 0.8506688932468}},
       {{0, 1}},
       {{0.08488805186072, 1}},
       {{0.2655756032646, 1}},
       {{0.5, 1}},
       {{0.7344243967354, 1}},
       {{0.9151119481393, 1}},
       {{1, 1}}}};

    interpolation_points[0].resize(panel_left.size());
    for(unsigned int i = 0; i < panel_left.size(); ++i)
      for(unsigned int d = 0; d < 2; ++d)
        interpolation_points[0][i][d] = panel_left[i][d];
    interpolation_points[1].resize(panel_right.size());
    for(unsigned int i = 0; i < panel_right.size(); ++i)
      for(unsigned int d = 0; d < 2; ++d)
        interpolation_points[1][i][d] = panel_right[i][d];
    basis = dealii::Polynomials::generate_complete_Lagrange_basis(
      dealii::QGaussLobatto<1>(n_shapes).get_points());
  }

  dealii::Point<dim>
  push_forward(const dealii::Point<dim> & xi_in) const final
  {
    dealii::Point<dim> xi = xi_in;
    const double       y  = 2.0 * (xi[1] - H) * INVERSE_HEIGHT - 1.0;

    // Cubic Hermite polynomial with same value and slope as
    // tanh(gamma*y)/tanh(gamma) at +1 and -1, with gamma = GRID_STRETCH_FAC.
    // The base mesh used for the optimization had gamma = 2.
    const double xi_1_hat = y * 0.5 * ((3. - deriv_tanh_gamma) + (deriv_tanh_gamma - 1.) * y * y);

    // Option that deforms mesh based on polynomial interpolation from a given
    // set of interpolation points, applied after the cubic Hermite
    // approximation. Note that the two panels are only C^0 continuous at
    // x = LENGTH/2, so this line must coincide with element faces.
    const bool   left_panel = xi[0] < LENGTH / 2;
    const double r0 =
      (left_panel) ? (xi[0] * INVERSE_LENGTH * 2) : (2 * xi[0] - LENGTH) * INVERSE_LENGTH;
    const double r1 = 0.5 * xi_1_hat + 0.5;

    // evaluate polynomial basis in x and y direction with deal.II function
    // of polynomial class
    dealii::ndarray<double, n_shapes, 2, 2> shapes;
    for(unsigned int i = 0; i < n_shapes; ++i)
      basis[i].values_of_array(std::array<double, 2>{{r0, r1}}, 0, shapes[i].data());
    const std::vector<dealii::Tensor<1, 2>> & coeffs = interpolation_points[left_panel ? 0 : 1];

    int                  dummy = 0;
    dealii::Tensor<1, 2> val =
      dealii::internal::do_interpolate_xy_value<2, n_shapes, double, dealii::Tensor<1, 2>, false>(
        coeffs.data(), {}, shapes.data(), 0, dummy);

    xi[0] = LENGTH * 0.5 * val[0] + (left_panel ? 0. : LENGTH * 0.5);
    xi[1] = val[1];

    // Finally move the points near the boundary to ensure approximately
    // equal-length elements and some finer distribution near the hill top.
    const double xi_0 =
      xi[0] < LENGTH / 2 ? get_scaled_x_point(xi[0]) : LENGTH - get_scaled_x_point(LENGTH - xi[0]);

    dealii::Point<dim> xi_bottom;
    xi_bottom[0] = xi_0;
    xi_bottom[1] = H + f(xi_0, H, LENGTH);
    if(dim > 2)
      xi_bottom[2] = xi[2];

    dealii::Point<dim> xi_top = xi;
    xi_top[1]                 = H + HEIGHT;
    return xi_top * xi[1] + xi_bottom * (1.0 - xi[1]);
  }

  dealii::Point<dim>
  pull_back(const dealii::Point<dim> & x) const final
  {
    AssertThrow(false, ExcNotImplemented());
    return x;
  }

  std::unique_ptr<dealii::Manifold<dim>>
  clone() const final
  {
    return std::make_unique<PeriodicHillManifoldOptimizedMesh<dim>>(H,
                                                                    LENGTH,
                                                                    HEIGHT,
                                                                    GRID_STRETCH_FAC);
  }

  double
  get_scaled_x_point(const double x_in) const
  {
    // 7-th order Hermite polynomial, which has 4 continuous derivatives at
    // x=0.5, the position where the halves are stitched together; the
    // coefficients are set up in the constructor; several terms are zero
    // because f(0) = 0, f''(0) = 0.
    const double t = x_in * INVERSE_LENGTH * 2;
    return t * (coefficients_hermite[0] +
                t * t *
                  (coefficients_hermite[1] +
                   t * (coefficients_hermite[2] +
                        t * (coefficients_hermite[3] +
                             t * (coefficients_hermite[4] + t * coefficients_hermite[5])))));
  }

private:
  const double H, LENGTH, HEIGHT, GRID_STRETCH_FAC, INVERSE_LENGTH, INVERSE_HEIGHT;
  const double tanh_gamma;
  const double deriv_tanh_gamma;
  std::array<std::vector<dealii::Tensor<1, 2>>, 2>     interpolation_points;
  std::vector<dealii::Polynomials::Polynomial<double>> basis;

  // description of shift along lower wall to place points more densely near
  // the hill top, described as Hermite polynomial of degree 7 (but with some
  // coefficients zero, meaning that we store only 6 out of 8 coefficients).
  std::array<double, 6> coefficients_hermite;
};



enum class GridDeformationType
{
  GradedMeshUniform,
  GradedMeshManualShift,
  GradedMeshOptimizationShift
};


/**
 * A manifold class used when constructing anisotropically coarsened
 * meshes. We realize this by independent meshes with different numbers of
 * cells in one of the directions (currently hardcoded as the y
 * direction). We then want to match cells in the two triangulations, for
 * which we want to shift the cells to create a nested space. This shift is
 * implemented by the present manifold class.
 */
template<int dim>
class PiecewiseLinearManifold : public dealii::ChartManifold<dim>
{
public:
  PiecewiseLinearManifold(const std::vector<std::array<double, 2>> & pieces) : pieces(pieces)
  {
    AssertThrow(pieces.size() >= 2,
                dealii::ExcMessage("Can only construct a transformation "
                                   "manifold when given at least two pieces."));
    for(unsigned int i = 1; i < pieces.size(); ++i)
    {
      AssertThrow(pieces[i][0] > pieces[i - 1][0],
                  dealii::ExcMessage("Pieces need to be strictly increasing"));
      AssertThrow(pieces[i][1] > pieces[i - 1][1],
                  dealii::ExcMessage("Pieces need to be strictly increasing"));
    }
  }

  virtual std::unique_ptr<dealii::Manifold<dim, dim>>
  clone() const override
  {
    return std::make_unique<PiecewiseLinearManifold<dim>>(pieces);
  }

  virtual dealii::Point<dim>
  pull_back(const dealii::Point<dim> & space_point) const override
  {
    dealii::Point<dim> chart_point = space_point;
    for(unsigned int i = 0; i < pieces.size() - 1; ++i)
      if(space_point[1] < pieces[i + 1][1])
      {
        chart_point[1] = pieces[i][0] + (space_point[1] - pieces[i][1]) *
                                          (pieces[i + 1][0] - pieces[i][0]) /
                                          (pieces[i + 1][1] - pieces[i][1]);
        return chart_point;
      }

    return chart_point;
  }

  virtual dealii::Point<dim>
  push_forward(const dealii::Point<dim> & chart_point) const override
  {
    dealii::Point<dim> space_point = chart_point;

    for(unsigned int i = 0; i < pieces.size() - 1; ++i)
      if(chart_point[1] < pieces[i + 1][0])
      {
        space_point[1] = pieces[i][1] + (chart_point[1] - pieces[i][0]) *
                                          (pieces[i + 1][1] - pieces[i][1]) /
                                          (pieces[i + 1][0] - pieces[i][0]);
        return space_point;
      }

    return space_point;
  }

private:
  const std::vector<std::array<double, 2>> pieces;
};

} // namespace ExaDG


#endif /* APPLICATIONS_PERIODIC_HILL_MANIFOLD_H_ */
