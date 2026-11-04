/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2026 by the ExaDG authors
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

// Test the matrix-vector product of RTOperator::RaviartThomasOperator, the
// momentum operator (mass plus viscous term in Laplace formulation) for
// Raviart-Thomas elements on extruded meshes, against the generic
// IncNS::MomentumOperator based on MatrixFree::loop(). The mesh is
// periodic in x and z direction and has no-slip walls in y direction. Also
// compare vmult_scratch() against vmult() and the Chebyshev iteration based
// on vmult_scratch() against dealii::PreconditionChebyshev.

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/numerics/vector_tools.h>

#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/momentum_operator.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/momentum_operator_rt.h>

#include "extruded_operators_common.h"

using namespace ExtrudedTest;

template<typename Number>
void
test(const unsigned int degree, const unsigned int n_refinements)
{
  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

  const MPI_Comm             comm = MPI_COMM_WORLD;
  dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(comm) == 0);

  dealii::parallel::distributed::Triangulation<dim> tria(comm);
  PeriodicFaces                                     periodic_faces;
  create_grid(tria, periodic_faces, n_refinements);
  const auto mapping = create_mapping(tria, 3);

  dealii::FE_RaviartThomasNodal<dim> fe(degree - 1);
  dealii::DoFHandler<dim>            dof_handler(tria);
  dealii::AffineConstraints<double>  constraints;

  // periodicity and no-slip walls in the same way as in the spatial
  // operator for the HDIV discretization, with a renumbering that sorts the
  // constrained DoFs to the end of the locally owned range
  const auto make_constraints = [&]() {
    constraints.reinit(dof_handler.locally_owned_dofs(),
                       dealii::DoFTools::extract_locally_relevant_dofs(dof_handler));
    dealii::DoFTools::make_periodicity_constraints(dof_handler, 10, 11, 0, constraints);
    dealii::DoFTools::make_periodicity_constraints(dof_handler, 14, 15, 2, constraints);
    dealii::VectorTools::project_boundary_values_div_conforming(
      dof_handler, 0, dealii::Functions::ZeroFunction<dim>(dim), wall_id, constraints, *mapping);
    constraints.close();
  };
  dof_handler.distribute_dofs(fe);
  make_constraints();
  {
    typename dealii::MatrixFree<dim, double>::AdditionalData mf_data;
    mf_data.overlap_communication_computation = false;
    dealii::DoFRenumbering::matrix_free_data_locality(dof_handler, constraints, mf_data);
  }
  make_constraints();

  pcout << "Raviart-Thomas degree " << degree << ", " << number_name<Number>() << ", "
        << tria.n_global_active_cells() << " cells, " << dof_handler.n_dofs() << " DoFs"
        << std::endl;

  RTOperator::RaviartThomasOperator<dim, Number> op;
  op.reinit(*mapping,
            dof_handler,
            constraints,
            Helper::compute_vectorization_category(tria),
            dealii::QGauss<1>(degree + 1),
            true);
  const double ip_factor = 1.3;
  op.set_penalty_parameters(ip_factor);

  // reference operator from ExaDG
  dealii::AffineConstraints<Number> constraints_number;
  constraints_number.copy_from(constraints);
  dealii::MatrixFree<dim, Number> matrix_free;
  matrix_free.reinit(
    *mapping,
    dof_handler,
    constraints_number,
    dealii::QGauss<dim>(degree + 1),
    create_matrix_free_data<Number>(
      ExaDG::MassKernel<dim, Number>::get_mapping_flags() ||
      ExaDG::IncNS::Operators::ViscousKernel<dim, Number>::get_mapping_flags(true, true)));

  auto boundary_descriptor = std::make_shared<ExaDG::IncNS::BoundaryDescriptorU<dim>>();
  boundary_descriptor->dirichlet_bc.emplace(
    wall_id, std::make_shared<dealii::Functions::ZeroFunction<dim>>(dim));

  // the operator works on a vector without the constrained DoFs, which come
  // last in the locally owned range of the vectors of MatrixFree
  VectorType src, dst, src_mf, dst_mf;
  op.initialize_dof_vector(src);
  op.initialize_dof_vector(dst);
  matrix_free.initialize_dof_vector(src_mf);
  matrix_free.initialize_dof_vector(dst_mf);
  fill_vector(src, 0);
  op.copy_this_to_mf_vector(src, src_mf);

  for(const auto & [factor_mass, viscosity] :
      std::vector<std::pair<double, double>>{{1., 0.}, {0., 1.}, {2.5, 0.3}})
  {
    pcout << " factor mass " << factor_mass << ", viscosity " << viscosity << std::endl;

    ExaDG::IncNS::MomentumOperatorData<dim> data;
    data.unsteady_problem              = true;
    data.viscous_problem               = true;
    data.viscous_kernel_data.IP_factor = ip_factor;
    data.viscous_kernel_data.viscosity = viscosity;
    data.viscous_kernel_data.formulation_viscous_term =
      ExaDG::IncNS::FormulationViscousTerm::LaplaceFormulation;
    data.viscous_kernel_data.IP_formulation = ExaDG::IncNS::InteriorPenaltyFormulation::SIPG;
    data.bc                                 = boundary_descriptor;
    ExaDG::IncNS::MomentumOperator<dim, Number> reference_op;
    reference_op.initialize(matrix_free, constraints_number, data, *mapping);
    reference_op.set_scaling_factor_mass_operator(factor_mass);
    reference_op.vmult(dst_mf, src_mf);

    VectorType reference;
    op.initialize_dof_vector(reference);
    op.copy_mf_to_this_vector(dst_mf, reference);

    op.set_parameters(factor_mass, viscosity);
    op.vmult(dst, src);
    dst -= reference;
    print_check(pcout,
                "vmult vs. IncNS::MomentumOperator",
                dst.linfty_norm(),
                reference.linfty_norm(),
                tolerance<Number>());

    op.vmult(reference, src);
    check_vmult_scratch(pcout, op, src, reference, tolerance<Number>());
  }

  op.set_parameters(1., 0.3);
  VectorType inverse_diagonal;
  op.initialize_dof_vector(inverse_diagonal);
  op.compute_diagonal(inverse_diagonal);
  for(Number & entry : inverse_diagonal)
    entry = Number(1.) / entry;
  check_chebyshev(pcout, op, inverse_diagonal, tolerance<Number>());
  pcout << std::endl;
}

int
main(int argc, char ** argv)
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  for(const unsigned int degree : {2, 3})
  {
    test<double>(degree, 2);
    test<float>(degree, 2);
  }
}
