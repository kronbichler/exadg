/*
 * VelocityDivergenceConvectiveTerm.h
 *
 *  Created on: Dec 21, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_VELOCITY_DIVERGENCE_CONVECTIVE_TERM_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_VELOCITY_DIVERGENCE_CONVECTIVE_TERM_H_


#include "../../incompressible_navier_stokes/user_interface/boundary_descriptor.h"
#include "../infrastructure/fe_evaluation_wrapper.h"
#include "operators/base_operator.h"

namespace IncNS
{
template<int dim>
class VelocityDivergenceConvectiveTermData
{
public:
  VelocityDivergenceConvectiveTermData() : dof_index_velocity(0), dof_index_pressure(0)
  {
  }

  unsigned int dof_index_velocity;
  unsigned int dof_index_pressure;

  std::shared_ptr<BoundaryDescriptorU<dim>> bc;
};

template<int dim,
         int fe_degree_u,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename value_type>
class VelocityDivergenceConvectiveTerm : public BaseOperator<dim>
{
public:
  static const bool         is_xwall = (xwall_quad_rule > 1) ? true : false;
  static const unsigned int n_actual_q_points_vel_nonlinear =
    (is_xwall) ? xwall_quad_rule : fe_degree_u + (fe_degree_u + 2) / 2;

  typedef LinearAlgebra::distributed::Vector<value_type> VectorType;

  /*
   * nomenclature typdedef FEEvaluationWrapper:
   * FEEval_name1_name2 : name1 specifies the dof handler, name2 the quadrature formula
   * example: FEEval_Pressure_Velocity_linear: dof handler for pressure (scalar quantity),
   * quadrature formula with fe_degree_velocity+1 quadrature points
   */

  typedef FEFaceEvaluationWrapper<dim,
                                  fe_degree_u,
                                  fe_degree_xwall,
                                  n_actual_q_points_vel_nonlinear,
                                  dim,
                                  value_type,
                                  is_xwall>
    FEFaceEval_Velocity_Velocity_nonlinear;

  typedef FEFaceEvaluationWrapperPressure<dim,
                                          fe_degree_p,
                                          fe_degree_xwall,
                                          n_actual_q_points_vel_nonlinear,
                                          1,
                                          value_type,
                                          is_xwall>
    FEFaceEval_Pressure_Velocity_nonlinear;

  typedef VelocityDivergenceConvectiveTerm<dim,
                                           fe_degree_u,
                                           fe_degree_p,
                                           fe_degree_xwall,
                                           xwall_quad_rule,
                                           value_type>
    This;

  VelocityDivergenceConvectiveTerm() : data(nullptr)
  {
  }

  void
  initialize(MatrixFree<dim, value_type> const &         mf_data,
             VelocityDivergenceConvectiveTermData<dim> & my_data_in)
  {
    this->data = &mf_data;
    my_data    = my_data_in;
  }

  void
  calculate(VectorType & dst, VectorType const & src) const
  {
    this->data->loop(&This::cell_loop, &This::face_loop, &This::boundary_face_loop, this, dst, src);
  }

private:
  void
  cell_loop(MatrixFree<dim, value_type> const &,
            VectorType &,
            VectorType const &,
            std::pair<unsigned int, unsigned int> const &) const
  {
  }

  void
  face_loop(MatrixFree<dim, value_type> const &,
            VectorType &,
            VectorType const &,
            std::pair<unsigned int, unsigned int> const &) const
  {
  }

  void
  boundary_face_loop(MatrixFree<dim, value_type> const &           data,
                     VectorType &                                  dst,
                     VectorType const &                            src,
                     std::pair<unsigned int, unsigned int> const & face_range) const
  {
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval_velocity(data,
                                                            this->fe_param,
                                                            true,
                                                            my_data.dof_index_velocity);
    FEFaceEval_Pressure_Velocity_nonlinear fe_eval_pressure(data,
                                                            this->fe_param,
                                                            true,
                                                            my_data.dof_index_pressure);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval_velocity.reinit(face);
      fe_eval_velocity.read_dof_values(src);
      fe_eval_velocity.evaluate(true, true);

      fe_eval_pressure.reinit(face);

      types::boundary_id boundary_id   = data.get_boundary_id(face);
      BoundaryTypeU      boundary_type = BoundaryTypeU::Undefined;

      if(my_data.bc->dirichlet_bc.find(boundary_id) != my_data.bc->dirichlet_bc.end())
        boundary_type = BoundaryTypeU::Dirichlet;
      else if(my_data.bc->neumann_bc.find(boundary_id) != my_data.bc->neumann_bc.end())
        boundary_type = BoundaryTypeU::Neumann;
      else if(my_data.bc->symmetry_bc.find(boundary_id) != my_data.bc->symmetry_bc.end())
        boundary_type = BoundaryTypeU::Symmetry;

      AssertThrow(boundary_type != BoundaryTypeU::Undefined,
                  ExcMessage("Boundary type of face is invalid or not implemented."));

      for(unsigned int q = 0; q < fe_eval_pressure.n_q_points; ++q)
      {
        if(boundary_type == BoundaryTypeU::Dirichlet)
        {
          Tensor<1, dim, VectorizedArray<value_type>> u      = fe_eval_velocity.get_value(q);
          Tensor<2, dim, VectorizedArray<value_type>> grad_u = fe_eval_velocity.get_gradient(q);
          Tensor<1, dim, VectorizedArray<value_type>> convective_term =
            grad_u * u + fe_eval_velocity.get_divergence(q) * u;

          VectorizedArray<value_type> flux_times_normal =
            convective_term * fe_eval_pressure.get_normal_vector(q);

          fe_eval_pressure.submit_value(flux_times_normal, q);
        }
        else if(boundary_type == BoundaryTypeU::Neumann || boundary_type == BoundaryTypeU::Symmetry)
        {
          // Do nothing on Neumann and Symmetry boundaries.
          // Remark: on symmetry boundaries we prescribe g_u * n = 0, and also g_{u_hat}*n = 0 in
          // case of the dual splitting scheme. This is in contrast to Dirichlet boundaries where we
          // prescribe a consistent boundary condition for g_{u_hat} derived from the convective
          // step of the dual splitting scheme which differs from the DBC g_u. Applying this
          // consistent DBC to symmetry boundaries and using g_u*n=0 as well as exploiting symmetry,
          // we obtain g_{u_hat}*n=0 on symmetry boundaries. Hence, there are no inhomogeneous
          // contributions for g_{u_hat}*n.
          VectorizedArray<value_type> zero = make_vectorized_array<value_type>(0.0);
          fe_eval_pressure.submit_value(zero, q);
        }
        else
        {
          AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
        }
      }
      fe_eval_pressure.integrate(true, false);
      fe_eval_pressure.distribute_local_to_global(dst);
    }
  }

  MatrixFree<dim, value_type> const * data;

  VelocityDivergenceConvectiveTermData<dim> my_data;
};


} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_VELOCITY_DIVERGENCE_CONVECTIVE_TERM_H_ \
        */
