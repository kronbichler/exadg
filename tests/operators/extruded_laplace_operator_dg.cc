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

// Test the matrix-vector product of LaplaceOperator::LaplaceOperatorDG, the
// interior penalty discretization of the pressure Laplacian (plus an
// optional mass term) for DG elements on extruded meshes, against the
// generic Poisson::LaplaceOperator and MassOperator based on
// MatrixFree::loop(). The mesh is periodic in x and z direction and has
// Neumann boundaries in y direction. Also compare vmult_scratch() against
// vmult() and the Chebyshev iteration based on vmult_scratch() against
// dealii::PreconditionChebyshev.

#include <deal.II/fe/fe_dgq.h>

#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/laplace_operator_extruded.h>
#include <exadg/operators/mass_operator.h>
#include <exadg/poisson/spatial_discretization/laplace_operator.h>

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

  dealii::FE_DGQ<dim>     fe(degree);
  dealii::DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  dealii::AffineConstraints<Number> constraints;
  constraints.close();

  pcout << "DG degree " << degree << ", " << number_name<Number>() << ", "
        << tria.n_global_active_cells() << " cells, " << dof_handler.n_dofs() << " DoFs"
        << std::endl;

  LaplaceOperator::LaplaceOperatorDG<dim, Number> op;
  op.reinit(*mapping,
            dof_handler,
            constraints,
            Helper::compute_vectorization_category(tria),
            dealii::QGauss<1>(degree + 1),
            true);
  const double ip_factor = 1.3;
  op.set_penalty_parameters(ip_factor);

  // reference operators from ExaDG
  dealii::MatrixFree<dim, Number> matrix_free;
  matrix_free.reinit(
    *mapping,
    dof_handler,
    constraints,
    dealii::QGauss<dim>(degree + 1),
    create_matrix_free_data<Number>(
      ExaDG::MassKernel<dim, Number>::get_mapping_flags() ||
      ExaDG::Poisson::Operators::LaplaceKernel<dim, Number>::get_mapping_flags(true, true)));

  ExaDG::Poisson::LaplaceOperatorData<0, dim> laplace_data;
  laplace_data.kernel_data.IP_factor = ip_factor;
  auto boundary_descriptor = std::make_shared<ExaDG::Poisson::BoundaryDescriptor<0, dim>>();
  boundary_descriptor->neumann_bc.emplace(wall_id,
                                          std::make_shared<dealii::Functions::ZeroFunction<dim>>());
  laplace_data.bc = boundary_descriptor;
  ExaDG::Poisson::LaplaceOperator<dim, Number, 1> laplace_op;
  laplace_op.initialize(matrix_free, constraints, laplace_data, false);

  ExaDG::MassOperator<dim, 1, Number> mass_op;
  mass_op.initialize(matrix_free, constraints, ExaDG::MassOperatorData<dim, Number>());

  VectorType src, dst, reference, tmp;
  op.initialize_dof_vector(src);
  op.initialize_dof_vector(dst);
  matrix_free.initialize_dof_vector(reference);
  matrix_free.initialize_dof_vector(tmp);
  fill_vector(src, 0);
  VectorType src_mf(reference);
  src_mf.copy_locally_owned_data_from(src);

  // the operator does not implement the pure mass case
  for(const auto & [factor_mass, factor_laplace] :
      std::vector<std::pair<double, double>>{{0., 1.}, {2.5, 0.3}})
  {
    pcout << " factor mass " << factor_mass << ", factor Laplace " << factor_laplace << std::endl;

    laplace_op.vmult(reference, src_mf);
    reference *= factor_laplace;
    mass_op.vmult(tmp, src_mf);
    reference.add(factor_mass, tmp);

    op.set_parameters(factor_mass, factor_laplace);
    op.vmult(dst, src);
    tmp.copy_locally_owned_data_from(dst);
    tmp -= reference;
    print_check(pcout,
                "vmult vs. Poisson::LaplaceOperator",
                tmp.linfty_norm(),
                reference.linfty_norm(),
                tolerance<Number>());

    check_vmult_scratch(pcout, op, src, dst, tolerance<Number>());
  }

  op.set_parameters(0., 1.);
  VectorType inverse_diagonal;
  op.compute_inverse_diagonal(inverse_diagonal);
  // the first Chebyshev iteration is fused into the second one
  static_assert(
    ExaDG::internal::has_vmult_scratch_with_before_loop<std::decay_t<decltype(op)>>::value);
  check_chebyshev(pcout, op, inverse_diagonal, tolerance<Number>());
  pcout << std::endl;
}

int
main(int argc, char ** argv)
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  for(const unsigned int degree : {1, 2, 3})
  {
    test<double>(degree, 2);
    test<float>(degree, 2);
  }
}
