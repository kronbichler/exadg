/*
 * projection_operator.cpp
 *
 *  Created on: Dec 6, 2018
 *      Author: fehn
 */

#include <deal.II/base/polynomials_raviart_thomas.h>
#include <deal.II/fe/fe_poly_tensor.h>

#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/projection_operator.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/weak_boundary_conditions.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
void
ProjectionOperator<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const &        matrix_free,
  dealii::AffineConstraints<Number> const &      affine_constraints,
  ProjectionOperatorData<dim> const &            data,
  Operators::DivergencePenaltyKernelData const & div_kernel_data,
  Operators::ContinuityPenaltyKernelData const & conti_kernel_data)
{
  std::shared_ptr<Operators::DivergencePenaltyKernel<dim, Number>> div_kernel;
  if(operator_data.use_divergence_penalty)
  {
    div_kernel = std::make_shared<Operators::DivergencePenaltyKernel<dim, Number>>();
    div_kernel->reinit(matrix_free,
                       operator_data.dof_index,
                       operator_data.quad_index,
                       div_kernel_data);
  }

  std::shared_ptr<Operators::ContinuityPenaltyKernel<dim, Number>> conti_kernel;
  if(operator_data.use_continuity_penalty)
  {
    conti_kernel = std::make_shared<Operators::ContinuityPenaltyKernel<dim, Number>>();
    conti_kernel->reinit(matrix_free,
                         operator_data.dof_index,
                         operator_data.quad_index,
                         conti_kernel_data);
  }

  initialize(matrix_free, affine_constraints, data, div_kernel, conti_kernel);
}



namespace
{
std::vector<unsigned int>
get_rt_dpo_vector(const unsigned int dim, const unsigned int degree)
{
  std::vector<unsigned int> dpo(dim + 1);
  dpo[0]                     = 0;
  dpo[1]                     = 0;
  unsigned int dofs_per_face = 1;
  for(unsigned int d = 1; d < dim; ++d)
    dofs_per_face *= (degree + 1);

  dpo[dim - 1] = dofs_per_face;
  dpo[dim]     = dim * (degree - 1) * dofs_per_face;

  return dpo;
}

// Finite element class that is used to define a projection operation into
// Hdiv(Omega); note that for now the element is only used to define the
// degrees of freedom without a proper polynomial space and many functions
// missing - this could be extended later (especially with respect to
// hanging node constraints)
template<int dim>
class FiniteElementProject : public dealii::FE_PolyTensor<dim>
{
public:
  FiniteElementProject(unsigned int const degree)
    : dealii::FE_PolyTensor<dim>(
        dealii::PolynomialsRaviartThomas<dim>(degree),
        dealii::FiniteElementData<dim>(get_rt_dpo_vector(dim, degree),
                                       dim,
                                       degree,
                                       dealii::FiniteElementData<dim>::Hdiv),
        std::vector<bool>(1, false),
        std::vector<dealii::ComponentMask>(dim * dealii::Utilities::pow(degree + 1, dim),
                                           std::vector<bool>(dim, true)))
  {
  }

  std::unique_ptr<dealii::FiniteElement<dim, dim>>
  clone() const
  {
    return std::make_unique<FiniteElementProject<dim>>(*this);
  }

  std::string
  get_name() const
  {
    return "FiniteElementProject<" + std::to_string(dim) + ">(" + std::to_string(this->degree) +
           ")";
  }
};
} // namespace

template<int dim, typename Number>
void
ProjectionOperator<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const &   matrix_free,
  dealii::AffineConstraints<Number> const & affine_constraints,
  ProjectionOperatorData<dim> const &       data,
  std::shared_ptr<DivKernel>                div_penalty_kernel,
  std::shared_ptr<ContiKernel>              conti_penalty_kernel)
{
  operator_data = data;

  Base::reinit(matrix_free, affine_constraints, data);

  div_kernel   = div_penalty_kernel;
  conti_kernel = conti_penalty_kernel;

  // mass operator
  this->integrator_flags.cell_evaluate  = CellFlags(true, false, false);
  this->integrator_flags.cell_integrate = CellFlags(true, false, false);

  // divergence penalty
  if(operator_data.use_divergence_penalty)
    this->integrator_flags = this->integrator_flags || div_kernel->get_integrator_flags();

  // continuity penalty
  if(operator_data.use_continuity_penalty)
    this->integrator_flags = this->integrator_flags || conti_kernel->get_integrator_flags();

  constraints_continuity.clear();
  dealii::IndexSet                relevant_dofs;
  dealii::DoFHandler<dim> const & dof = matrix_free.get_dof_handler(data.dof_index);
  dealii::DoFTools::extract_locally_relevant_dofs(dof, relevant_dofs);
  constraints_continuity.reinit(relevant_dofs);
  std::vector<dealii::types::global_dof_index> dof_indices_mine(dof.get_fe().dofs_per_cell);
  std::vector<dealii::types::global_dof_index> dof_indices_neighbor(dof.get_fe().dofs_per_cell);
  dealii::QGaussLobatto<dim - 1>               quadrature(dof.get_fe().degree + 1);
  dealii::FEFaceValues<dim> fe_face_values(*matrix_free.get_mapping_info().mapping,
                                           dof.get_fe(),
                                           quadrature,
                                           dealii::update_normal_vectors);

  // This code is experimental to build constraints, but not easily
  // generalizable to the parallel case where some constraints in the ghost
  // layer do not get picked up.
  if(false)
  {
    const unsigned int scalar_dofs_per_cell = dealii::Utilities::pow(dof.get_fe().degree + 1, dim);
    const dealii::Table<2, unsigned int> & face_indices =
      matrix_free.get_shape_info(data.dof_index, 0, 0, 0, 0).face_to_cell_index_nodal;
    AssertThrow(face_indices.size(1) == quadrature.size(), dealii::ExcInternalError());
    unsigned int other_indices[dim][dim - 1];
    for(unsigned int d = 0; d < dim; ++d)
      for(unsigned int e = 0, c = 0; e < dim; ++e)
        if(d != e)
          other_indices[d][c++] = e;

    // TODO: maybe it is better to let MatrixFree compute these, given that we
    // have unique faces with correct orientation info and subfaces etc already?
    // The only slight complication is to exchange the constraints near
    // processor boundaries, which should be done by some update_ghost_values or
    // similar
    std::vector<bool> touched(dof.n_dofs());
    for(auto const & cell : dof.active_cell_iterators())
      if(cell->is_locally_owned())
        for(unsigned int face : cell->face_indices())
        {
          fe_face_values.reinit(cell, face);
          cell->get_dof_indices(dof_indices_mine);
          if(!cell->at_boundary(face) || cell->has_periodic_neighbor(face))
          {
            typename dealii::DoFHandler<dim>::cell_iterator neighbor =
              cell->neighbor_or_periodic_neighbor(face);
            if(neighbor < cell)
              continue;
            neighbor->get_dof_indices(dof_indices_neighbor);
            const unsigned int neighbor_face = cell->has_periodic_neighbor(face) ?
                                                 cell->periodic_neighbor_face_no(face) :
                                                 cell->neighbor_face_no(face);
            for(unsigned int q = 0; q < quadrature.size(); ++q)
            {
              std::array<dealii::types::global_dof_index, dim> my_dof_indices;
              for(unsigned int d = 0; d < dim; ++d)
                my_dof_indices[d] =
                  dof_indices_mine[d * scalar_dofs_per_cell + face_indices(face, q)];
              std::array<dealii::types::global_dof_index, dim> neigh_dof_indices;
              for(unsigned int d = 0; d < dim; ++d)
                neigh_dof_indices[d] =
                  dof_indices_neighbor[d * scalar_dofs_per_cell + face_indices(neighbor_face, q)];

              // TODO: fix orientation
              const dealii::Tensor<1, dim> normal             = fe_face_values.normal_vector(q);
              unsigned int                 constraining_index = 0;
              for(unsigned int d = 1; d < dim; ++d)
                if(std::abs(normal[d]) > std::abs(normal[constraining_index]))
                  constraining_index = d;

              // condition: n1 (um1 - up1) + n2 (um2 - up2) + n3 (um3 - up3) = 0
              AssertThrow(touched[my_dof_indices[constraining_index]] == false,
                          dealii::ExcMessage("Already constrained index " +
                                             std::to_string(my_dof_indices[constraining_index]) +
                                             " " + std::to_string(constraining_index) + "  " +
                                             std::to_string(cell->active_cell_index())));
              touched[my_dof_indices[constraining_index]] = true;
              constraints_continuity.add_line(my_dof_indices[constraining_index]);
              constraints_continuity.add_entry(my_dof_indices[constraining_index],
                                               neigh_dof_indices[constraining_index],
                                               1.0);
              for(unsigned int e = 0; e < dim - 1; ++e)
              {
                const unsigned int j      = other_indices[constraining_index][e];
                const double       factor = normal[j] / normal[constraining_index];
                if(std::abs(factor) > 1e-12)
                {
                  constraints_continuity.add_entry(my_dof_indices[constraining_index],
                                                   my_dof_indices[j],
                                                   -factor);
                  constraints_continuity.add_entry(my_dof_indices[constraining_index],
                                                   neigh_dof_indices[j],
                                                   factor);
                }
              }
            }
          }
        }
    std::cout << "Found " << constraints_continuity.n_constraints() << " constraints out of "
              << dof.n_dofs() << std::endl;
  }
  constraints_continuity.close();

  // set up the Hdiv auxiliary space
  fe_penalty = std::make_shared<FiniteElementProject<dim>>(dof.get_fe().degree);
  dof_handler_penalty.reinit(dof.get_triangulation());
  dof_handler_penalty.distribute_dofs(*fe_penalty);
  // todo: make constraints
  // std::cout << "Number of Hdiv DoFs: " << dof_handler_penalty.n_dofs() << std::endl;
  std::array<dealii::types::global_dof_index, 2 * dim + 1> default_entry;
  for(dealii::types::global_dof_index & i : default_entry)
    i = dealii::numbers::invalid_dof_index;

  // Extract degrees of freedom for access to finite element solutions
  const unsigned int dofs_per_face = dealii::Utilities::pow(fe_penalty->degree + 1, dim - 1);
  std::vector<std::array<dealii::types::global_dof_index, 2 * dim + 1>> extracted_indices(
    matrix_free.n_cell_batches() * dealii::VectorizedArray<Number>::size(), default_entry);
  std::vector<dealii::types::global_dof_index> dof_indices(fe_penalty->dofs_per_cell);
  std::vector<dealii::types::global_dof_index> neigh_dof_indices(fe_penalty->dofs_per_cell);
  std::vector<dealii::types::global_dof_index> all_dof_indices;
  for(unsigned int c = 0; c < matrix_free.n_cell_batches(); ++c)
    for(unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(c); ++v)
    {
      const auto cell = matrix_free.get_cell_iterator(c, v);
      typename dealii::DoFHandler<dim>::active_cell_iterator cell_pen(
        &matrix_free.get_dof_handler(0).get_triangulation(),
        cell->level(),
        cell->index(),
        &dof_handler_penalty);
      cell_pen->get_dof_indices(dof_indices);

      // resolve periodic constraints
      for(unsigned int f = 0; f < 2 * dim; ++f)
        if(cell->has_periodic_neighbor(f) && cell->periodic_neighbor(f) < cell)
        {
          cell_pen->periodic_neighbor(f)->get_dof_indices(neigh_dof_indices);
          const unsigned int neighbor_face = cell_pen->periodic_neighbor_face_no(f);
          for(unsigned int c = 0; c < dofs_per_face; ++c)
            dof_indices[f * dofs_per_face + c] =
              neigh_dof_indices[neighbor_face * dofs_per_face + c];
        }
      all_dof_indices.insert(all_dof_indices.end(), dof_indices.begin(), dof_indices.end());
      // extract face dofs and first cell dof
      for(unsigned int f = 0; f < 2 * dim + 1; ++f)
      {
        for(unsigned int c = 0; c < dofs_per_face; ++c)
          Assert(dof_indices[f * dofs_per_face + c] == dof_indices[f * dofs_per_face] + c,
                 dealii::ExcMessage("Wrong numbering!"));
        extracted_indices[c * dealii::VectorizedArray<Number>::size() + v][f] =
          dof_indices[f * dofs_per_face];
      }
    }

  // sort and compress out duplicates
  std::sort(all_dof_indices.begin(), all_dof_indices.end());
  all_dof_indices.erase(std::unique(all_dof_indices.begin(), all_dof_indices.end()),
                        all_dof_indices.end());

  // create vector partitioner based on the numbers appearing on locally owned
  // cells
  dealii::IndexSet accessed_dofs(dof_handler_penalty.n_dofs());
  accessed_dofs.add_indices(all_dof_indices.begin(), all_dof_indices.end());
  accessed_dofs.compress();

  velocity_hdiv.reinit(dof_handler_penalty.locally_owned_dofs(),
                       accessed_dofs,
                       dof_handler_penalty.get_communicator());

  // set up locally owned dofs for the access
  const auto & partitioner = *velocity_hdiv.get_partitioner();
  compressed_dof_indices.resize(extracted_indices.size());
  for(unsigned int i = 0; i < extracted_indices.size(); ++i)
    for(unsigned int f = 0; f < extracted_indices[i].size(); ++f)
      if(extracted_indices[i][f] != dealii::numbers::invalid_dof_index)
      {
        AssertThrow(partitioner.in_local_range(extracted_indices[i][f]) ||
                      partitioner.is_ghost_entry(extracted_indices[i][f]),
                    dealii::ExcMessage("Index " + std::to_string(extracted_indices[i][f]) +
                                       " is neither owned nor ghosted"));
        compressed_dof_indices[i][f] = partitioner.global_to_local(extracted_indices[i][f]);
      }

  // compute touch count for mass matrix preconditioner
  velocity_hdiv = 0;
  for(unsigned int c = 0; c < matrix_free.n_cell_batches(); ++c)
    for(unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(c); ++v)
      for(unsigned int f = 0; f < 2 * dim + 1; ++f)
        velocity_hdiv.local_element(
          compressed_dof_indices[c * dealii::VectorizedArray<Number>::size() + v][f]) += Number(1.);
  velocity_hdiv.compress(dealii::VectorOperation::add);
  velocity_hdiv.update_ghost_values();
  inverse_touch_count.resize(matrix_free.n_cell_batches());
  for(unsigned int c = 0; c < matrix_free.n_cell_batches(); ++c)
    for(unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(c); ++v)
    {
      for(unsigned int f = 0; f < 2 * dim; ++f)
        inverse_touch_count[c][f][v] =
          1. / velocity_hdiv.local_element(
                 compressed_dof_indices[c * dealii::VectorizedArray<Number>::size() + v][f]);
      AssertThrow(
        velocity_hdiv.local_element(
          compressed_dof_indices[c * dealii::VectorizedArray<Number>::size() + v][2 * dim]) == 1.,
        dealii::ExcInternalError());
    }
  velocity_hdiv = 0;

  // deduce the interior dofs, i.e., those not located on the faces for the
  // Hdiv auxiliary space
  interior_dofs.clear();
  const unsigned int            fe_degree = dof.get_fe().degree;
  std::array<unsigned int, dim> stride1;
  stride1[0] = fe_degree + 1;
  for(unsigned int d = 1; d < dim; ++d)
    stride1[d] = 1;
  std::array<unsigned int, dim> stride2;
  for(unsigned int d = 0; d < dim - 1; ++d)
    stride2[d] = dealii::Utilities::pow(fe_degree + 1, 2);
  stride2[dim - 1] = fe_degree + 1;
  for(unsigned int comp = 0; comp < dim; ++comp)
  {
    const unsigned int dir_stride = dealii::Utilities::pow(fe_degree + 1, comp);
    for(unsigned int i2 = 0; i2 < (dim > 2 ? fe_degree + 1 : 1); ++i2)
      for(unsigned int i1 = 0; i1 < (dim > 1 ? fe_degree + 1 : 1); ++i1)
        for(unsigned int i0 = 1; i0 < fe_degree; ++i0)
          interior_dofs.push_back(comp * dealii::Utilities::pow(fe_degree + 1, dim) +
                                  i0 * dir_stride + i1 * stride1[comp] + i2 * stride2[comp]);
  }
}

template<int dim, typename Number>
ProjectionOperatorData<dim>
ProjectionOperator<dim, Number>::get_data() const
{
  return operator_data;
}

template<int dim, typename Number>
Operators::DivergencePenaltyKernelData
ProjectionOperator<dim, Number>::get_divergence_kernel_data() const
{
  if(operator_data.use_divergence_penalty)
    return div_kernel->get_data();
  else
    return Operators::DivergencePenaltyKernelData();
}

template<int dim, typename Number>
Operators::ContinuityPenaltyKernelData
ProjectionOperator<dim, Number>::get_continuity_kernel_data() const
{
  if(operator_data.use_continuity_penalty)
    return conti_kernel->get_data();
  else
    return Operators::ContinuityPenaltyKernelData();
}

template<int dim, typename Number>
double
ProjectionOperator<dim, Number>::get_time_step_size() const
{
  return this->time_step_size;
}

template<int dim, typename Number>
dealii::LinearAlgebra::distributed::Vector<Number> const &
ProjectionOperator<dim, Number>::get_velocity() const
{
  AssertThrow(velocity != nullptr,
              dealii::ExcMessage("Velocity ptr is not initialized in ProjectionOperator."));

  return *velocity;
}

template<int dim, typename Number>
void
ProjectionOperator<dim, Number>::update(VectorType const & velocity, double const & dt)
{
  this->velocity = &velocity;

  if(operator_data.use_divergence_penalty)
    div_kernel->calculate_penalty_parameter(velocity);
  if(operator_data.use_continuity_penalty)
    conti_kernel->calculate_penalty_parameter(velocity);

  time_step_size = dt;
}

template<int dim, typename Number>
void
ProjectionOperator<dim, Number>::reinit_cell(unsigned int const cell) const
{
  Base::reinit_cell(cell);

  if(operator_data.use_divergence_penalty)
    div_kernel->reinit_cell(*this->integrator);
}

template<int dim, typename Number>
void
ProjectionOperator<dim, Number>::reinit_face(unsigned int const face) const
{
  Base::reinit_face(face);

  if(operator_data.use_continuity_penalty)
    conti_kernel->reinit_face(*this->integrator_m, *this->integrator_p);
}

template<int dim, typename Number>
void
ProjectionOperator<dim, Number>::reinit_boundary_face(unsigned int const face) const
{
  Base::reinit_boundary_face(face);

  conti_kernel->reinit_boundary_face(*this->integrator_m);
}

template<int dim, typename Number>
void
ProjectionOperator<dim, Number>::reinit_face_cell_based(
  unsigned int const               cell,
  unsigned int const               face,
  dealii::types::boundary_id const boundary_id) const
{
  Base::reinit_face_cell_based(cell, face, boundary_id);

  if(operator_data.use_continuity_penalty)
    conti_kernel->reinit_face_cell_based(boundary_id, *this->integrator_m, *this->integrator_p);
}

template<int dim, typename Number>
void
ProjectionOperator<dim, Number>::do_cell_integral(IntegratorCell & integrator) const
{
  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    integrator.submit_value(integrator.get_value(q), q);

    if(operator_data.use_divergence_penalty)
      integrator.submit_divergence(time_step_size * div_kernel->get_volume_flux(integrator, q), q);
  }
}

template<int dim, typename Number>
void
ProjectionOperator<dim, Number>::do_face_integral(IntegratorFace & integrator_m,
                                                  IntegratorFace & integrator_p) const
{
  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    vector u_m      = integrator_m.get_value(q);
    vector u_p      = integrator_p.get_value(q);
    vector normal_m = integrator_m.get_normal_vector(q);

    vector flux = time_step_size * conti_kernel->calculate_flux(u_m, u_p, normal_m);

    integrator_m.submit_value(flux, q);
    integrator_p.submit_value(-flux, q);
  }
}

template<int dim, typename Number>
void
ProjectionOperator<dim, Number>::do_face_int_integral(IntegratorFace & integrator_m,
                                                      IntegratorFace & integrator_p) const
{
  (void)integrator_p;

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    vector u_m = integrator_m.get_value(q);
    vector u_p; // set u_p to zero
    vector normal_m = integrator_m.get_normal_vector(q);

    vector flux = time_step_size * conti_kernel->calculate_flux(u_m, u_p, normal_m);

    integrator_m.submit_value(flux, q);
  }
}

template<int dim, typename Number>
void
ProjectionOperator<dim, Number>::do_face_ext_integral(IntegratorFace & integrator_m,
                                                      IntegratorFace & integrator_p) const
{
  (void)integrator_m;

  for(unsigned int q = 0; q < integrator_p.n_q_points; ++q)
  {
    vector u_m; // set u_m to zero
    vector u_p      = integrator_p.get_value(q);
    vector normal_p = -integrator_p.get_normal_vector(q);

    vector flux = time_step_size * conti_kernel->calculate_flux(u_p, u_m, normal_p);

    integrator_p.submit_value(flux, q);
  }
}

template<int dim, typename Number>
void
ProjectionOperator<dim, Number>::do_boundary_integral(
  IntegratorFace &                   integrator_m,
  OperatorType const &               operator_type,
  dealii::types::boundary_id const & boundary_id) const
{
  if(operator_data.use_boundary_data == true)
  {
    BoundaryTypeU boundary_type = operator_data.bc->get_boundary_type(boundary_id);

    for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
    {
      vector u_m      = calculate_interior_value(q, integrator_m, operator_type);
      vector u_p      = calculate_exterior_value(u_m,
                                            q,
                                            integrator_m,
                                            operator_type,
                                            boundary_type,
                                            boundary_id,
                                            operator_data.bc,
                                            this->time);
      vector normal_m = integrator_m.get_normal_vector(q);

      vector flux = time_step_size * conti_kernel->calculate_flux(u_m, u_p, normal_m);

      integrator_m.submit_value(flux, q);
    }
  }
  else
  {
    for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
    {
      vector flux; // continuity penalty term is zero on boundary faces if u_p = u_m

      integrator_m.submit_value(flux, q);
    }
  }
}


template<int dim, typename Number>
void
ProjectionOperator<dim, Number>::vmult_hdiv(VectorType & dst, VectorType const & src) const
{
  src.update_ghost_values();
  IntegratorCell integrator(*this->matrix_free, operator_data.dof_index, operator_data.quad_index);
  std::size_t const fe_degree       = integrator.get_shape_info().data.front().fe_degree;
  std::size_t const dofs_per_face   = dealii::Utilities::pow(fe_degree + 1, dim - 1);
  std::size_t const dofs_per_comp   = dealii::Utilities::pow(fe_degree + 1, dim);
  std::size_t const n_interior_dofs = dim * dofs_per_face * (fe_degree - 1);
  std::size_t const n_q_points      = integrator.n_q_points;
  const dealii::Table<2, unsigned int> & face_indices =
    integrator.get_shape_info().face_to_cell_index_nodal;
  dst = 0;

  for(unsigned int cell = 0; cell < this->matrix_free->n_cell_batches(); ++cell)
  {
    integrator.reinit(cell);
    this->div_kernel->reinit_cell(integrator);
    const dealii::VectorizedArray<Number> tau = this->div_kernel->get_tau();
    // read dof values manually
    for(unsigned int v = 0; v < this->matrix_free->n_active_entries_per_cell_batch(cell); ++v)
    {
      for(unsigned int comp = 0; comp < dim; ++comp)
        for(unsigned int f = 2 * comp; f < 2 * comp + 2; ++f)
          for(unsigned i = 0; i < dofs_per_face; ++i)
            integrator.begin_dof_values()[comp * dofs_per_comp + face_indices(f, i)][v] =
              src.local_element(
                compressed_dof_indices[cell * dealii::VectorizedArray<Number>::size() + v][f] + i);
      for(unsigned int i = 0; i < n_interior_dofs; ++i)
        integrator.begin_dof_values()[interior_dofs[i]][v] = src.local_element(
          compressed_dof_indices[cell * dealii::VectorizedArray<Number>::size() + v][2 * dim] + i);
    }
    integrator.evaluate(dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);
    for(std::size_t q = 0; q < n_q_points; ++q)
    {
      dealii::VectorizedArray<Number> div = integrator.begin_gradients()[q];
      for(unsigned int d = 1; d < dim; ++d)
        div += integrator.begin_gradients()[q + (d * dim + d) * n_q_points];
      for(unsigned int d = 0; d < dim * dim; ++d)
        integrator.begin_gradients()[q + d * n_q_points] = Number(0.);
      const Number q_weight = this->matrix_free->get_mapping_info()
                                .cell_data[operator_data.quad_index]
                                .descriptor[0]
                                .quadrature_weights[q];
      dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> jac =
        invert(integrator.inverse_jacobian(q));
      dealii::VectorizedArray<Number> inv_det = determinant(integrator.inverse_jacobian(q));
      for(unsigned int d = 0; d < dim; ++d)
        integrator.begin_gradients()[q + (d * dim + d) * n_q_points] =
          div * (q_weight * inv_det * tau);
      vector vel    = integrator.get_value(q);
      vector result = (jac * (transpose(jac) * vel)) * (inv_det * q_weight);
      for(unsigned int d = 0; d < dim; ++d)
        integrator.begin_values()[d * n_q_points + q] = result[d];
    }
    integrator.integrate(dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);
    for(unsigned int v = 0; v < this->matrix_free->n_active_entries_per_cell_batch(cell); ++v)
    {
      for(unsigned int comp = 0; comp < dim; ++comp)
        for(unsigned int f = 2 * comp; f < 2 * comp + 2; ++f)
          for(unsigned i = 0; i < dofs_per_face; ++i)
            dst.local_element(
              compressed_dof_indices[cell * dealii::VectorizedArray<Number>::size() + v][f] + i) +=
              integrator.begin_dof_values()[comp * dofs_per_comp + face_indices(f, i)][v];
      for(unsigned int i = 0; i < n_interior_dofs; ++i)
        dst.local_element(
          compressed_dof_indices[cell * dealii::VectorizedArray<Number>::size() + v][2 * dim] + i) =
          integrator.begin_dof_values()[interior_dofs[i]][v];
    }
  }

  dst.compress(dealii::VectorOperation::add);
  src.zero_out_ghost_values();
}


template<int dim, typename Number>
class ProjectionMatrix
{
public:
  ProjectionMatrix(ProjectionOperator<dim, Number> const & projection_operator)
    : projection_operator(projection_operator)
  {
  }

  template<typename VectorType>
  void
  vmult(VectorType & dst, VectorType const & src) const
  {
    projection_operator.vmult_hdiv(dst, src);
  }

private:
  ProjectionOperator<dim, Number> const & projection_operator;
};


template<int dim, typename Number>
class ProjectionPreconditioner
{
public:
  ProjectionPreconditioner(ProjectionOperator<dim, Number> const & projection_operator)
    : projection_operator(projection_operator)
  {
  }

  template<typename VectorType>
  void
  vmult(VectorType & dst, VectorType const & src) const
  {
    projection_operator.vmult_invmass_hdiv(dst, src);
  }

private:
  ProjectionOperator<dim, Number> const & projection_operator;
};

template<int dim, typename Number>
void
ProjectionOperator<dim, Number>::vmult_invmass_hdiv(VectorType & dst, VectorType const & src) const
{
  src.update_ghost_values();
  IntegratorCell integrator(*this->matrix_free, operator_data.dof_index, operator_data.quad_index);
  std::size_t const fe_degree       = integrator.get_shape_info().data.front().fe_degree;
  std::size_t const dofs_per_face   = dealii::Utilities::pow(fe_degree + 1, dim - 1);
  std::size_t const dofs_per_comp   = dealii::Utilities::pow(fe_degree + 1, dim);
  std::size_t const n_interior_dofs = dim * dofs_per_face * (fe_degree - 1);
  const dealii::Table<2, unsigned int> & face_indices =
    integrator.get_shape_info().face_to_cell_index_nodal;
  dst = 0;

  for(unsigned int cell = 0; cell < this->matrix_free->n_cell_batches(); ++cell)
  {
    integrator.reinit(cell);
    // read dof values manually
    for(unsigned int v = 0; v < this->matrix_free->n_active_entries_per_cell_batch(cell); ++v)
    {
      for(unsigned int comp = 0; comp < dim; ++comp)
        for(unsigned int f = 2 * comp; f < 2 * comp + 2; ++f)
          for(unsigned i = 0; i < dofs_per_face; ++i)
            integrator.begin_dof_values()[comp * dofs_per_comp + face_indices(f, i)][v] =
              src.local_element(
                compressed_dof_indices[cell * dealii::VectorizedArray<Number>::size() + v][f] + i);
      for(unsigned int i = 0; i < n_interior_dofs; ++i)
        integrator.begin_dof_values()[interior_dofs[i]][v] = src.local_element(
          compressed_dof_indices[cell * dealii::VectorizedArray<Number>::size() + v][2 * dim] + i);
    }
    for(unsigned int comp = 0; comp < dim; ++comp)
      for(unsigned int f = 2 * comp; f < 2 * comp + 2; ++f)
        for(unsigned i = 0; i < dofs_per_face; ++i)
          integrator.begin_dof_values()[comp * dofs_per_comp + face_indices(f, i)] *=
            inverse_touch_count[cell][f];
    dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_general,
                                             dim,
                                             0,
                                             0,
                                             dealii::VectorizedArray<Number>>
      evaluator(integrator.get_shape_info().data.front().inverse_shape_values,
                dealii::AlignedVector<dealii::VectorizedArray<Number>>(),
                dealii::AlignedVector<dealii::VectorizedArray<Number>>(),
                integrator.get_shape_info().data.front().fe_degree + 1,
                integrator.get_shape_info().data.front().fe_degree + 1);

    for(unsigned int d = 0; d < dim; ++d)
    {
      dealii::VectorizedArray<Number> * out = integrator.begin_dof_values() + d * dofs_per_comp;
      // Need to select 'apply' method with hessian slot because values
      // assume symmetries that do not exist in the inverse shapes
      evaluator.template values<0, true, false>(out, out);
      if(dim > 1)
        evaluator.template values<1, true, false>(out, out);
      if(dim > 2)
        evaluator.template values<2, true, false>(out, out);
    }
    for(unsigned int q = 0; q < dofs_per_comp; ++q)
    {
      vector vel;
      for(unsigned int d = 0; d < dim; ++d)
        vel[d] = integrator.begin_dof_values()[d * dofs_per_comp + q];
      const auto                            inv_jac  = integrator.inverse_jacobian(q);
      const dealii::VectorizedArray<Number> inv_det  = determinant(inv_jac);
      const Number                          q_weight = this->matrix_free->get_mapping_info()
                                .cell_data[operator_data.quad_index]
                                .descriptor[0]
                                .quadrature_weights[q];
      const dealii::VectorizedArray<Number> inv_factor = Number(1.) / (q_weight * inv_det);
      const vector result = (transpose(inv_jac) * (inv_jac * vel)) * inv_factor;
      for(unsigned int d = 0; d < dim; ++d)
        integrator.begin_dof_values()[d * dofs_per_comp + q] = result[d];
    }
    for(unsigned int d = 0; d < dim; ++d)
    {
      dealii::VectorizedArray<Number> * out = integrator.begin_dof_values() + d * dofs_per_comp;
      if(dim > 2)
        evaluator.template values<2, false, false>(out, out);
      if(dim > 1)
        evaluator.template values<1, false, false>(out, out);
      evaluator.template values<0, false, false>(out, out);
    }
    for(unsigned int comp = 0; comp < dim; ++comp)
      for(unsigned int f = 2 * comp; f < 2 * comp + 2; ++f)
        for(unsigned i = 0; i < dofs_per_face; ++i)
          integrator.begin_dof_values()[comp * dofs_per_comp + face_indices(f, i)] *=
            inverse_touch_count[cell][f];
    for(unsigned int v = 0; v < this->matrix_free->n_active_entries_per_cell_batch(cell); ++v)
    {
      for(unsigned int comp = 0; comp < dim; ++comp)
        for(unsigned int f = 2 * comp; f < 2 * comp + 2; ++f)
          for(unsigned i = 0; i < dofs_per_face; ++i)
            dst.local_element(
              compressed_dof_indices[cell * dealii::VectorizedArray<Number>::size() + v][f] + i) +=
              integrator.begin_dof_values()[comp * dofs_per_comp + face_indices(f, i)][v];
      for(unsigned int i = 0; i < n_interior_dofs; ++i)
        dst.local_element(
          compressed_dof_indices[cell * dealii::VectorizedArray<Number>::size() + v][2 * dim] + i) =
          integrator.begin_dof_values()[interior_dofs[i]][v];
    }
  }

  dst.compress(dealii::VectorOperation::add);
  src.zero_out_ghost_values();
}

template<int dim, typename Number>
unsigned int
ProjectionOperator<dim, Number>::solve_hdiv(VectorType & velocity)
{
  VectorType hdiv_rhs;
  hdiv_rhs.reinit(velocity_hdiv);
  IntegratorCell integrator(*this->matrix_free, operator_data.dof_index, operator_data.quad_index);
  std::size_t const fe_degree       = integrator.get_shape_info().data.front().fe_degree;
  std::size_t const dofs_per_face   = dealii::Utilities::pow(fe_degree + 1, dim - 1);
  std::size_t const dofs_per_comp   = dealii::Utilities::pow(fe_degree + 1, dim);
  std::size_t const n_interior_dofs = dim * dofs_per_face * (fe_degree - 1);
  std::size_t const n_q_points      = integrator.n_q_points;
  const dealii::Table<2, unsigned int> & face_indices =
    integrator.get_shape_info().face_to_cell_index_nodal;

  // Compute rhs by testing DG velocity with Hdiv basis functions
  for(unsigned int cell = 0; cell < this->matrix_free->n_cell_batches(); ++cell)
  {
    integrator.reinit(cell);
    integrator.gather_evaluate(velocity, dealii::EvaluationFlags::values);
    for(std::size_t q = 0; q < n_q_points; ++q)
    {
      vector                                                  vel = integrator.get_value(q);
      dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> jac =
        invert(integrator.inverse_jacobian(q));
      const Number q_weight = this->matrix_free->get_mapping_info()
                                .cell_data[operator_data.quad_index]
                                .descriptor[0]
                                .quadrature_weights[q];
      vector result = (jac * vel) * dealii::VectorizedArray<Number>(q_weight);
      for(unsigned int d = 0; d < dim; ++d)
        integrator.begin_values()[d * n_q_points + q] = result[d];
    }
    integrator.integrate(dealii::EvaluationFlags::values);
    for(unsigned int v = 0; v < this->matrix_free->n_active_entries_per_cell_batch(cell); ++v)
    {
      for(unsigned int comp = 0; comp < dim; ++comp)
        for(unsigned int f = 2 * comp; f < 2 * comp + 2; ++f)
          for(unsigned i = 0; i < dofs_per_face; ++i)
            hdiv_rhs.local_element(
              compressed_dof_indices[cell * dealii::VectorizedArray<Number>::size() + v][f] + i) +=
              integrator.begin_dof_values()[comp * dofs_per_comp + face_indices(f, i)][v];
      for(unsigned int i = 0; i < n_interior_dofs; ++i)
        hdiv_rhs.local_element(
          compressed_dof_indices[cell * dealii::VectorizedArray<Number>::size() + v][2 * dim] + i) =
          integrator.begin_dof_values()[interior_dofs[i]][v];
    }
  }

  hdiv_rhs.compress(dealii::VectorOperation::add);
  // std::cout << "L2 norm rhs " << hdiv_rhs.l2_norm() << std::endl;

  ProjectionMatrix<dim, Number>         matrix(*this);
  ProjectionPreconditioner<dim, Number> preconditioner(*this);

  dealii::ReductionControl     control(1000, 1e-6, 1e-16);
  dealii::SolverCG<VectorType> solver(control);
  velocity_hdiv = 0;
  solver.solve(matrix, velocity_hdiv, hdiv_rhs, preconditioner);
  // std::cout << "Number of iterations project: " << control.last_step() << std::endl;
  // std::cout << "Velocity hdiv norm " << velocity_hdiv.l2_norm() << std::endl;

  velocity_hdiv.update_ghost_values();

  dealii::MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, dim, Number> mass_matrix(
    integrator);
  // Interpolate velocity back to DG space
  for(unsigned int cell = 0; cell < this->matrix_free->n_cell_batches(); ++cell)
  {
    integrator.reinit(cell);
    // read dof values manually
    for(unsigned int v = 0; v < this->matrix_free->n_active_entries_per_cell_batch(cell); ++v)
    {
      for(unsigned int comp = 0; comp < dim; ++comp)
        for(unsigned int f = 2 * comp; f < 2 * comp + 2; ++f)
          for(unsigned i = 0; i < dofs_per_face; ++i)
            integrator.begin_dof_values()[comp * dofs_per_comp + face_indices(f, i)][v] =
              velocity_hdiv.local_element(
                compressed_dof_indices[cell * dealii::VectorizedArray<Number>::size() + v][f] + i);
      for(unsigned int i = 0; i < n_interior_dofs; ++i)
        integrator.begin_dof_values()[interior_dofs[i]][v] = velocity_hdiv.local_element(
          compressed_dof_indices[cell * dealii::VectorizedArray<Number>::size() + v][2 * dim] + i);
    }
    integrator.evaluate(dealii::EvaluationFlags::values);
    for(std::size_t q = 0; q < n_q_points; ++q)
    {
      vector                                                  vel = integrator.get_value(q);
      dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> jac =
        invert(integrator.inverse_jacobian(q));
      dealii::VectorizedArray<Number> inv_det = determinant(integrator.inverse_jacobian(q));
      vel                                     = (transpose(jac) * vel) * inv_det;
      for(unsigned int d = 0; d < dim; ++d)
        integrator.begin_values()[d * n_q_points + q] = vel[d];
    }
    mass_matrix.transform_from_q_points_to_basis(dim,
                                                 integrator.begin_values(),
                                                 integrator.begin_dof_values());
    integrator.set_dof_values(velocity);
  }

  return control.last_step();
}



template class ProjectionOperator<2, float>;
template class ProjectionOperator<2, double>;

template class ProjectionOperator<3, float>;
template class ProjectionOperator<3, double>;

} // namespace IncNS
} // namespace ExaDG
