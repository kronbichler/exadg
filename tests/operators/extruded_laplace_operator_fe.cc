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

// Test the matrix-vector product of LaplaceOperator::LaplaceOperatorFE, the
// Laplacian for continuous elements on extruded meshes used on the levels
// of the pressure multigrid, against the generic Poisson::LaplaceOperator
// based on MatrixFree::cell_loop(). The mesh is periodic in x and z
// direction, with periodicity constraints and DoF numbering set up as for
// the multigrid levels, and has Neumann boundaries in y direction. Both
// operators set the constrained entries to the source vector. Also compare
// vmult_scratch() against vmult() and the Chebyshev iteration based on
// vmult_scratch() against dealii::PreconditionChebyshev.

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>

#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/laplace_operator_extruded.h>
#include <exadg/operators/constraints.h>
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

  dealii::FE_Q<dim>                 fe(degree);
  dealii::DoFHandler<dim>           dof_handler(tria);
  dealii::AffineConstraints<Number> constraints;
  const std::vector<unsigned int>   cell_vectorization_category =
    Helper::compute_vectorization_category(tria);

  // periodicity constraints and renumbering as on the multigrid levels in
  // PoissonPreconditionerMG
  std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::DoFHandler<dim>::cell_iterator>>
             periodic_dof_faces;
  const auto make_constraints = [&]() {
    constraints.reinit(dof_handler.locally_owned_dofs(),
                       dealii::DoFTools::extract_locally_relevant_dofs(dof_handler));
    for(const auto & face_pair : periodic_dof_faces)
      ExaDG::Periodicity::make_periodicity_constraints_recursively(
        face_pair.cell[0]->face(face_pair.face_idx[0]),
        face_pair.cell[1]->face(face_pair.face_idx[1]),
        constraints);
    constraints.close();
  };
  dof_handler.distribute_dofs(fe);
  dealii::GridTools::collect_periodic_faces(dof_handler, 10, 11, 0, periodic_dof_faces);
  dealii::GridTools::collect_periodic_faces(dof_handler, 14, 15, 2, periodic_dof_faces);
  make_constraints();
  {
    typename dealii::MatrixFree<dim, Number>::AdditionalData mf_data;
    mf_data.cell_vectorization_category = cell_vectorization_category;
    dealii::DoFRenumbering::matrix_free_data_locality(dof_handler, constraints, mf_data);
  }
  make_constraints();

  pcout << "FE degree " << degree << ", " << number_name<Number>() << ", "
        << tria.n_global_active_cells() << " cells, " << dof_handler.n_dofs() << " DoFs"
        << std::endl;

  LaplaceOperator::LaplaceOperatorFE<dim, Number> op;
  op.reinit(
    *mapping, dof_handler, constraints, cell_vectorization_category, dealii::QGauss<1>(degree + 1));

  // reference operator from ExaDG
  dealii::MatrixFree<dim, Number> matrix_free;
  matrix_free.reinit(
    *mapping,
    dof_handler,
    constraints,
    dealii::QGauss<dim>(degree + 1),
    create_matrix_free_data<Number>(
      ExaDG::Poisson::Operators::LaplaceKernel<dim, Number>::get_mapping_flags(false, true)));

  ExaDG::Poisson::LaplaceOperatorData<0, dim> laplace_data;
  auto boundary_descriptor = std::make_shared<ExaDG::Poisson::BoundaryDescriptor<0, dim>>();
  boundary_descriptor->neumann_bc.emplace(wall_id,
                                          std::make_shared<dealii::Functions::ZeroFunction<dim>>());
  laplace_data.bc = boundary_descriptor;
  ExaDG::Poisson::LaplaceOperator<dim, Number, 1> laplace_op;
  laplace_op.initialize(matrix_free, constraints, laplace_data, false);

  VectorType src, dst, reference;
  op.initialize_dof_vector(src);
  op.initialize_dof_vector(dst);
  matrix_free.initialize_dof_vector(reference);
  fill_vector(src, 0);
  VectorType src_mf(reference);
  src_mf.copy_locally_owned_data_from(src);

  laplace_op.vmult(reference, src_mf);

  op.vmult(dst, src);
  VectorType difference(reference);
  difference.copy_locally_owned_data_from(dst);
  difference -= reference;
  print_check(pcout,
              "vmult vs. Poisson::LaplaceOperator",
              difference.linfty_norm(),
              reference.linfty_norm(),
              tolerance<Number>());

  check_vmult_scratch(pcout, op, src, dst, tolerance<Number>());

  VectorType inverse_diagonal;
  op.compute_inverse_diagonal(inverse_diagonal);
  // the first Chebyshev iteration is fused into the second one, first with
  // 'before_loop' on all entries before the loop, as the mesh is below the
  // default minimal size, then within the cell loop
  static_assert(
    ExaDG::internal::has_vmult_scratch_with_before_loop<std::decay_t<decltype(op)>>::value);
  check_chebyshev(
    pcout, op, inverse_diagonal, tolerance<Number>(), op.get_matrix_free().get_constrained_dofs());
  op.set_min_size_for_pre_list(0);
  pcout << "  with before_loop within the cell loop" << std::endl;
  check_chebyshev(
    pcout, op, inverse_diagonal, tolerance<Number>(), op.get_matrix_free().get_constrained_dofs());
  pcout << std::endl;
}

int
main(int argc, char ** argv)
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  for(const unsigned int degree : {1, 2, 3, 4})
  {
    test<double>(degree, 2);
    test<float>(degree, 2);
  }
}
