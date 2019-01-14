#include "laplace_operator.h"

#include <navierstokes/config.h>
#include "../../functionalities/evaluate_functions.h"

namespace Poisson
{
template<int dim, int degree, typename Number>
LaplaceOperator<dim, degree, Number>::LaplaceOperator()
  : OperatorBase<dim, degree, Number, LaplaceOperatorData<dim>>()
{
}

template<int dim, int degree, typename Number>
void
LaplaceOperator<dim, degree, Number>::reinit(
  Mapping<dim> const &              mapping,
  MatrixFree<dim, Number> const &   mf_data,
  AffineConstraints<double> const & constraint_matrix,
  LaplaceOperatorData<dim> const &  operator_data_in) const
{
  LaplaceOperatorData<dim> operator_data = operator_data_in;
  operator_data.mapping                  = std::move(mapping.clone());

  this->reinit(mf_data, constraint_matrix, operator_data);
}

template<int dim, int degree, typename Number>
void
LaplaceOperator<dim, degree, Number>::reinit(MatrixFree<dim, Number> const &   mf_data,
                                             AffineConstraints<double> const & constraint_matrix,
                                             LaplaceOperatorData<dim> const &  operator_data) const
{
  Base::reinit(mf_data, constraint_matrix, operator_data);

  // calculate penalty parameters
  IP::calculate_penalty_parameter<dim, degree, Number>(array_penalty_parameter,
                                                       *this->data,
                                                       *operator_data.mapping,
                                                       this->operator_data.dof_index);
}

/*

template<int dim, int degree, typename Number>
void
LaplaceOperator<dim, degree, Number>::vmult(VectorType & dst, VectorType const & src) const
{
  this->apply(dst, src);
}

template<int dim, int degree, typename Number>
void
LaplaceOperator<dim, degree, Number>::vmult_add(VectorType & dst, VectorType const & src) const
{
  this->apply_add(dst, src);
}

template<int dim, int degree, typename Number>
AffineConstraints<double> const &
LaplaceOperator<dim, degree, Number>::get_constraint_matrix() const
{
  return this->do_get_constraint_matrix();
}

template<int dim, int degree, typename Number>
MatrixFree<dim, Number> const &
LaplaceOperator<dim, degree, Number>::get_data() const
{
  return *this->data;
}

template<int dim, int degree, typename Number>
unsigned int
LaplaceOperator<dim, degree, Number>::get_dof_index() const
{
  return this->operator_data.dof_index;
}

template<int dim, int degree, typename Number>
void
LaplaceOperator<dim, degree, Number>::calculate_inverse_diagonal(VectorType & diagonal) const
{
  this->calculate_diagonal(diagonal);
  invert_diagonal(diagonal);
}

template<int dim, int degree, typename Number>
void
LaplaceOperator<dim, degree, Number>::apply_inverse_block_diagonal(VectorType &       dst,
                                                                   VectorType const & src) const
{
  AssertThrow(this->operator_data.implement_block_diagonal_preconditioner_matrix_free == false,
              ExcMessage("Not implemented."));

  this->apply_inverse_block_diagonal_matrix_based(dst, src);
}

template<int dim, int degree, typename Number>
void
LaplaceOperator<dim, degree, Number>::update_block_diagonal_preconditioner() const
{
  this->do_update_block_diagonal_preconditioner();
}
 */

template<int dim, int degree, typename Number>
bool
LaplaceOperator<dim, degree, Number>::is_singular() const
{
  return this->operator_is_singular();
}

template<int dim, int degree, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  LaplaceOperator<dim, degree, Number>::calculate_value_flux(scalar const & jump_value) const
{
  return -0.5 * jump_value;
}

template<int dim, int degree, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  LaplaceOperator<dim, degree, Number>::calculate_interior_value(
    unsigned int const   q,
    FEEvalFace const &   fe_eval,
    OperatorType const & operator_type) const
{
  scalar value_m = make_vectorized_array<Number>(0.0);

  if(operator_type == OperatorType::full || operator_type == OperatorType::homogeneous)
  {
    value_m = fe_eval.get_value(q);
  }
  else if(operator_type == OperatorType::inhomogeneous)
  {
    value_m = make_vectorized_array<Number>(0.0);
  }
  else
  {
    AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
  }

  return value_m;
}

template<int dim, int degree, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  LaplaceOperator<dim, degree, Number>::calculate_exterior_value(
    scalar const &           value_m,
    unsigned int const       q,
    FEEvalFace const &       fe_eval,
    OperatorType const &     operator_type,
    BoundaryType const &     boundary_type,
    types::boundary_id const boundary_id) const
{
  scalar value_p = make_vectorized_array<Number>(0.0);

  if(boundary_type == BoundaryType::dirichlet)
  {
    if(operator_type == OperatorType::full || operator_type == OperatorType::inhomogeneous)
    {
      typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::iterator it =
        this->operator_data.bc->dirichlet_bc.find(boundary_id);
      Point<dim, scalar> q_points = fe_eval.quadrature_point(q);

      scalar g = evaluate_scalar_function(it->second, q_points, this->eval_time);

      value_p = -value_m + 2.0 * g;
    }
    else if(operator_type == OperatorType::homogeneous)
    {
      value_p = -value_m;
    }
    else
    {
      AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
    }
  }
  else if(boundary_type == BoundaryType::neumann)
  {
    value_p = value_m;
  }
  else
  {
    AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
  }

  return value_p;
}

template<int dim, int degree, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  LaplaceOperator<dim, degree, Number>::calculate_gradient_flux(
    scalar const & normal_gradient_m,
    scalar const & normal_gradient_p,
    scalar const & jump_value,
    scalar const & penalty_parameter) const
{
  return 0.5 * (normal_gradient_m + normal_gradient_p) - penalty_parameter * jump_value;
}

template<int dim, int degree, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  LaplaceOperator<dim, degree, Number>::calculate_interior_normal_gradient(
    unsigned int const   q,
    FEEvalFace const &   fe_eval,
    OperatorType const & operator_type) const
{
  scalar normal_gradient_m = make_vectorized_array<Number>(0.0);

  if(operator_type == OperatorType::full || operator_type == OperatorType::homogeneous)
  {
    normal_gradient_m = fe_eval.get_normal_derivative(q);
  }
  else if(operator_type == OperatorType::inhomogeneous)
  {
    normal_gradient_m = make_vectorized_array<Number>(0.0);
  }
  else
  {
    AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
  }

  return normal_gradient_m;
}

template<int dim, int degree, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  VectorizedArray<Number>
  LaplaceOperator<dim, degree, Number>::calculate_exterior_normal_gradient(
    scalar const &           normal_gradient_m,
    unsigned int const       q,
    FEEvalFace const &       fe_eval,
    OperatorType const &     operator_type,
    BoundaryType const &     boundary_type,
    types::boundary_id const boundary_id) const
{
  scalar normal_gradient_p = make_vectorized_array<Number>(0.0);

  if(boundary_type == BoundaryType::dirichlet)
  {
    normal_gradient_p = normal_gradient_m;
  }
  else if(boundary_type == BoundaryType::neumann)
  {
    if(operator_type == OperatorType::full || operator_type == OperatorType::inhomogeneous)
    {
      typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::iterator it =
        this->operator_data.bc->neumann_bc.find(boundary_id);
      Point<dim, scalar> q_points = fe_eval.quadrature_point(q);

      scalar h = evaluate_scalar_function(it->second, q_points, this->eval_time);

      normal_gradient_p = -normal_gradient_m + 2.0 * h;
    }
    else if(operator_type == OperatorType::homogeneous)
    {
      normal_gradient_p = -normal_gradient_m;
    }
    else
    {
      AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
    }
  }
  else
  {
    AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
  }

  return normal_gradient_p;
}

template<int dim, int degree, typename Number>
void
LaplaceOperator<dim, degree, Number>::do_cell_integral(FEEvalCell & fe_eval,
                                                       unsigned int const /*cell*/) const
{
  for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    fe_eval.submit_gradient(fe_eval.get_gradient(q), q);
}

template<int dim, int degree, typename Number>
void
LaplaceOperator<dim, degree, Number>::do_face_integral(FEEvalFace & fe_eval,
                                                       FEEvalFace & fe_eval_neighbor,
                                                       unsigned int const /*face*/) const
{
  scalar tau_IP = std::max(fe_eval.read_cell_data(array_penalty_parameter),
                           fe_eval_neighbor.read_cell_data(array_penalty_parameter)) *
                  IP::get_penalty_factor<Number>(degree, this->operator_data.IP_factor);

  for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
  {
    scalar jump_value = fe_eval.get_value(q) - fe_eval_neighbor.get_value(q);
    scalar value_flux = calculate_value_flux(jump_value);

    scalar normal_gradient_m = fe_eval.get_normal_derivative(q);
    scalar normal_gradient_p = fe_eval_neighbor.get_normal_derivative(q);
    scalar gradient_flux =
      calculate_gradient_flux(normal_gradient_m, normal_gradient_p, jump_value, tau_IP);

    fe_eval.submit_normal_derivative(value_flux, q);
    fe_eval_neighbor.submit_normal_derivative(value_flux, q);

    fe_eval.submit_value(-gradient_flux, q);
    fe_eval_neighbor.submit_value(gradient_flux, q); // + sign since n⁺ = -n⁻
  }
}

template<int dim, int degree, typename Number>
void
LaplaceOperator<dim, degree, Number>::do_face_int_integral(FEEvalFace & fe_eval,
                                                           FEEvalFace & fe_eval_neighbor,
                                                           unsigned int const /*face*/) const
{
  scalar tau_IP = std::max(fe_eval.read_cell_data(array_penalty_parameter),
                           fe_eval_neighbor.read_cell_data(array_penalty_parameter)) *
                  IP::get_penalty_factor<Number>(degree, this->operator_data.IP_factor);

  for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
  {
    // set exterior value to zero
    scalar jump_value = fe_eval.get_value(q);
    scalar value_flux = calculate_value_flux(jump_value);

    // set exterior value to zero
    scalar normal_gradient_m = fe_eval.get_normal_derivative(q);
    scalar normal_gradient_p = make_vectorized_array<Number>(0.0);
    scalar gradient_flux =
      calculate_gradient_flux(normal_gradient_m, normal_gradient_p, jump_value, tau_IP);

    fe_eval.submit_normal_derivative(value_flux, q);
    fe_eval.submit_value(-gradient_flux, q);
  }
}

template<int dim, int degree, typename Number>
void
LaplaceOperator<dim, degree, Number>::do_face_ext_integral(FEEvalFace & fe_eval,
                                                           FEEvalFace & fe_eval_neighbor,
                                                           unsigned int const /*face*/) const
{
  scalar tau_IP = std::max(fe_eval.read_cell_data(array_penalty_parameter),
                           fe_eval_neighbor.read_cell_data(array_penalty_parameter)) *
                  IP::get_penalty_factor<Number>(degree, this->operator_data.IP_factor);

  for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
  {
    // set value_m to zero
    scalar jump_value = fe_eval_neighbor.get_value(q);
    scalar value_flux = calculate_value_flux(jump_value);

    // set gradient_m to zero
    scalar normal_gradient_m = make_vectorized_array<Number>(0.0);
    // minus sign to get the correct normal vector n⁺ = -n⁻
    scalar normal_gradient_p = -fe_eval_neighbor.get_normal_derivative(q);
    scalar gradient_flux =
      calculate_gradient_flux(normal_gradient_m, normal_gradient_p, jump_value, tau_IP);

    // minus sign since n⁺ = -n⁻
    fe_eval_neighbor.submit_normal_derivative(-value_flux, q);
    fe_eval_neighbor.submit_value(-gradient_flux, q);
  }
}

template<int dim, int degree, typename Number>
void
LaplaceOperator<dim, degree, Number>::do_boundary_integral(FEEvalFace &               fe_eval,
                                                           OperatorType const &       operator_type,
                                                           types::boundary_id const & boundary_id,
                                                           unsigned int const /*face*/) const
{
  BoundaryType boundary_type = this->operator_data.bc->get_boundary_type(boundary_id);

  scalar tau_IP = fe_eval.read_cell_data(array_penalty_parameter) *
                  IP::get_penalty_factor<Number>(degree, this->operator_data.IP_factor);

  for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
  {
    scalar value_m = calculate_interior_value(q, fe_eval, operator_type);
    scalar value_p =
      calculate_exterior_value(value_m, q, fe_eval, operator_type, boundary_type, boundary_id);
    scalar jump_value = value_m - value_p;
    scalar value_flux = calculate_value_flux(jump_value);

    scalar normal_gradient_m = calculate_interior_normal_gradient(q, fe_eval, operator_type);
    scalar normal_gradient_p = calculate_exterior_normal_gradient(
      normal_gradient_m, q, fe_eval, operator_type, boundary_type, boundary_id);
    scalar gradient_flux =
      calculate_gradient_flux(normal_gradient_m, normal_gradient_p, jump_value, tau_IP);

    fe_eval.submit_normal_derivative(value_flux, q);
    fe_eval.submit_value(-gradient_flux, q);
  }
}

template<int dim, int degree, typename Number>
PreconditionableOperator<dim, Number> *
LaplaceOperator<dim, degree, Number>::get_new(unsigned int deg) const
{
  switch(deg)
  {
#if DEGREE_0
    case 0:
      return new LaplaceOperator<dim, 0, Number>();
#endif
#if DEGREE_1
    case 1:
      return new LaplaceOperator<dim, 1, Number>();
#endif
#if DEGREE_2
    case 2:
      return new LaplaceOperator<dim, 2, Number>();
#endif
#if DEGREE_3
    case 3:
      return new LaplaceOperator<dim, 3, Number>();
#endif
#if DEGREE_4
    case 4:
      return new LaplaceOperator<dim, 4, Number>();
#endif
#if DEGREE_5
    case 5:
      return new LaplaceOperator<dim, 5, Number>();
#endif
#if DEGREE_6
    case 6:
      return new LaplaceOperator<dim, 6, Number>();
#endif
#if DEGREE_7
    case 7:
      return new LaplaceOperator<dim, 7, Number>();
#endif
#if DEGREE_8
    case 8:
      return new LaplaceOperator<dim, 8, Number>();
#endif
#if DEGREE_9
    case 9:
      return new LaplaceOperator<dim, 9, Number>();
#endif
#if DEGREE_10
    case 10:
      return new LaplaceOperator<dim, 10, Number>();
#endif
#if DEGREE_11
    case 11:
      return new LaplaceOperator<dim, 11, Number>();
#endif
#if DEGREE_12
    case 12:
      return new LaplaceOperator<dim, 12, Number>();
#endif
#if DEGREE_13
    case 13:
      return new LaplaceOperator<dim, 13, Number>();
#endif
#if DEGREE_14
    case 14:
      return new LaplaceOperator<dim, 14, Number>();
#endif
#if DEGREE_15
    case 15:
      return new LaplaceOperator<dim, 15, Number>();
#endif
    default:
      AssertThrow(false, ExcMessage("LaplaceOperator not implemented for this degree!"));
      return new LaplaceOperator<dim, 1, Number>(); // dummy return (statement not reached)
  }
}

template<int dim, int degree, typename Number>
void
LaplaceOperator<dim, degree, Number>::do_verify_boundary_conditions(
  types::boundary_id const             boundary_id,
  LaplaceOperatorData<dim> const &     operator_data,
  std::set<types::boundary_id> const & periodic_boundary_ids) const
{
  unsigned int counter = 0;
  if(operator_data.bc->dirichlet_bc.find(boundary_id) != operator_data.bc->dirichlet_bc.end())
    counter++;

  if(operator_data.bc->neumann_bc.find(boundary_id) != operator_data.bc->neumann_bc.end())
    counter++;

  if(periodic_boundary_ids.find(boundary_id) != periodic_boundary_ids.end())
    counter++;

  AssertThrow(counter == 1, ExcMessage("Boundary face with non-unique boundary type found."));
}

} // namespace Poisson

#include "laplace_operator.hpp"
