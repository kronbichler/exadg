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

// Wrapper of the block-Jacobi preconditioner of LaplaceOperatorDG for
// dealii::PreconditionChebyshev
template<typename Operator>
struct BlockInverse
{
  template<typename VectorType>
  void
  vmult(VectorType & dst, const VectorType & src) const
  {
    op->apply_block_inverse(dst, src);
  }
  const Operator * op;
};

template<typename Operator>
void
check_block_inverse_properties(dealii::ConditionalOStream & pcout, const Operator & op)
{
  using VectorType = typename Operator::VectorType;
  using Number     = typename VectorType::value_type;
  VectorType u, v, inverse_u, inverse_v;
  op.initialize_dof_vector(u);
  op.initialize_dof_vector(v);
  op.initialize_dof_vector(inverse_u);
  op.initialize_dof_vector(inverse_v);
  fill_vector(u, 1);
  fill_vector(v, 2);
  op.apply_block_inverse(inverse_u, u);
  op.apply_block_inverse(inverse_v, v);
  const double uv = inverse_u * v;
  const double vu = inverse_v * u;
  print_check(pcout,
              "block inverse symmetric",
              std::abs(uv - vu),
              std::abs(uv) + std::abs(vu),
              10. * tolerance<Number>());
  const double uu = inverse_u * u;
  pcout << "  " << std::left << std::setw(46) << "block inverse positive"
        << ": " << (uu > 0 ? "OK" : "FAILED") << std::endl;
}

template<typename Operator>
void
check_chebyshev_block_inverse(dealii::ConditionalOStream & pcout, const Operator & op)
{
  using VectorType = typename Operator::VectorType;
  using Number     = typename VectorType::value_type;
  using Scratch    = ExaDG::PreconditionChebyshevScratch<Operator>;
  using DealII     = dealii::PreconditionChebyshev<Operator, VectorType, BlockInverse<Operator>>;

  VectorType rhs, initial, reference, result;
  op.initialize_dof_vector(rhs);
  op.initialize_dof_vector(initial);
  fill_vector(rhs, 1);
  fill_vector(initial, 2);

  for(const auto type : {Scratch::PolynomialType::first_kind, Scratch::PolynomialType::fourth_kind})
  {
    typename DealII::AdditionalData data_dealii;
    data_dealii.preconditioner      = std::make_shared<BlockInverse<Operator>>();
    data_dealii.preconditioner->op  = &op;
    data_dealii.degree              = 4;
    data_dealii.smoothing_range     = 20.;
    data_dealii.max_eigenvalue      = 2.;
    data_dealii.eig_cg_n_iterations = 0;
    data_dealii.polynomial_type     = type == Scratch::PolynomialType::first_kind ?
                                        DealII::AdditionalData::PolynomialType::first_kind :
                                        DealII::AdditionalData::PolynomialType::fourth_kind;
    DealII chebyshev_dealii;
    chebyshev_dealii.initialize(op, data_dealii);

    typename Scratch::AdditionalData data;
    data.polynomial_type        = type;
    data.use_cell_block_inverse = true;
    data.degree                 = data_dealii.degree;
    data.smoothing_range        = data_dealii.smoothing_range;
    data.max_eigenvalue         = data_dealii.max_eigenvalue;
    Scratch chebyshev;
    chebyshev.initialize(op, data);

    for(const unsigned int degree : {1, 2, 4})
    {
      chebyshev_dealii.set_degree(degree);
      chebyshev.set_degree(degree);
      const std::string name = std::string("Chebyshev block ") +
                               (type == Scratch::PolynomialType::first_kind ? "1st" : "4th") +
                               " kind degree " + std::to_string(degree) + " ";

      op.initialize_dof_vector(reference);
      op.initialize_dof_vector(result);
      chebyshev_dealii.vmult(reference, rhs);
      chebyshev.vmult(result, rhs);
      result -= reference;
      print_check(
        pcout, name + "vmult", result.linfty_norm(), reference.linfty_norm(), tolerance<Number>());

      reference = initial;
      result    = initial;
      chebyshev_dealii.step(reference, rhs);
      chebyshev.step(result, rhs);
      result -= reference;
      print_check(
        pcout, name + "step", result.linfty_norm(), reference.linfty_norm(), tolerance<Number>());

      op.initialize_dof_vector(reference);
      op.initialize_dof_vector(result);
      const double norm_reference = chebyshev_dealii.vmult_with_last_residual_norm(reference, rhs);
      const double norm           = chebyshev.vmult_with_last_residual_norm(result, rhs);
      print_check(pcout,
                  name + "last residual norm",
                  std::abs(norm - norm_reference),
                  norm_reference,
                  tolerance<Number>());
    }
  }
}

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

    VectorType inverse_diagonal, reference_diagonal;
    op.compute_inverse_diagonal(inverse_diagonal);
    laplace_op.calculate_diagonal(reference_diagonal);
    reference_diagonal *= factor_laplace;
    mass_op.calculate_diagonal(tmp);
    reference_diagonal.add(factor_mass, tmp);
    for(unsigned int i = 0; i < reference_diagonal.locally_owned_size(); ++i)
      reference_diagonal.local_element(i) = Number(1.) / reference_diagonal.local_element(i);
    tmp.copy_locally_owned_data_from(inverse_diagonal);
    tmp -= reference_diagonal;
    print_check(pcout,
                "inverse diagonal vs. Poisson::LaplaceOperator",
                tmp.linfty_norm(),
                reference_diagonal.linfty_norm(),
                tolerance<Number>());
  }

  op.set_parameters(0., 1.);
  VectorType inverse_diagonal;
  op.compute_inverse_diagonal(inverse_diagonal);
  // the first Chebyshev iteration is fused into the second one
  static_assert(
    ExaDG::internal::has_vmult_scratch_with_before_loop<std::decay_t<decltype(op)>>::value);
  check_chebyshev(pcout, op, inverse_diagonal, tolerance<Number>());
  check_chebyshev_fourth_kind(pcout, op, inverse_diagonal, tolerance<Number>());

  // block-Jacobi preconditioner with fast diagonalization on the curved
  // mesh, where the Cartesian surrogate is only an approximation of the
  // cell blocks: symmetry and positivity, and the cell-wise Chebyshev
  // iteration against dealii::PreconditionChebyshev with the same
  // preconditioner applied on the whole vector
  op.setup_block_inverse();
  {
    // the correction steps with the actual cell block reduce the defect of
    // the inverse of the Cartesian surrogate
    const double defect_0 = op.compute_block_inverse_defect(0);
    const double defect_1 = op.compute_block_inverse_defect(1);
    const double defect_2 = op.compute_block_inverse_defect(2);
    pcout << "  " << std::left << std::setw(46) << "block inverse defect reduced by corrections"
          << ": " << (defect_1 < 0.5 * defect_0 && defect_2 < 0.5 * defect_1 ? "OK" : "FAILED")
          << std::endl;
  }
  check_block_inverse_properties(pcout, op);
  check_chebyshev_block_inverse(pcout, op);
  pcout << std::endl;
}

// On a Cartesian mesh, the surrogate block is the exact cell block of the
// operator, so the block-Jacobi preconditioner with fast diagonalization
// must agree with the matrix-based block-Jacobi preconditioner of
// Poisson::LaplaceOperator
template<typename Number>
void
test_block_inverse_cartesian(const unsigned int degree, const unsigned int n_refinements)
{
  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

  const MPI_Comm             comm = MPI_COMM_WORLD;
  dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(comm) == 0);

  dealii::parallel::distributed::Triangulation<dim> tria(comm);
  PeriodicFaces                                     periodic_faces;
  create_grid(tria, periodic_faces, n_refinements);
  const dealii::MappingQ<dim> mapping(1);

  dealii::FE_DGQ<dim>     fe(degree);
  dealii::DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  dealii::AffineConstraints<Number> constraints;
  constraints.close();

  pcout << "Cartesian mesh, DG degree " << degree << ", " << number_name<Number>() << ", "
        << tria.n_global_active_cells() << " cells, " << dof_handler.n_dofs() << " DoFs"
        << std::endl;

  LaplaceOperator::LaplaceOperatorDG<dim, Number> op;
  op.reinit(mapping,
            dof_handler,
            constraints,
            Helper::compute_vectorization_category(tria),
            dealii::QGauss<1>(degree + 1),
            true);
  const double ip_factor = 1.3;
  op.set_penalty_parameters(ip_factor);
  op.set_parameters(0., 1.);
  op.setup_block_inverse();

  dealii::MatrixFree<dim, Number> matrix_free;
  matrix_free.reinit(
    mapping,
    dof_handler,
    constraints,
    dealii::QGauss<dim>(degree + 1),
    create_matrix_free_data<Number>(
      ExaDG::Poisson::Operators::LaplaceKernel<dim, Number>::get_mapping_flags(true, true)));
  ExaDG::Poisson::LaplaceOperatorData<0, dim> laplace_data;
  laplace_data.kernel_data.IP_factor = ip_factor;
  auto boundary_descriptor = std::make_shared<ExaDG::Poisson::BoundaryDescriptor<0, dim>>();
  boundary_descriptor->neumann_bc.emplace(wall_id,
                                          std::make_shared<dealii::Functions::ZeroFunction<dim>>());
  laplace_data.bc = boundary_descriptor;
  ExaDG::Poisson::LaplaceOperator<dim, Number, 1> laplace_op;
  laplace_op.initialize(matrix_free, constraints, laplace_data, false);
  laplace_op.initialize_block_diagonal_preconditioner(true);

  VectorType src, dst, reference, src_mf;
  op.initialize_dof_vector(src);
  op.initialize_dof_vector(dst);
  matrix_free.initialize_dof_vector(reference);
  matrix_free.initialize_dof_vector(src_mf);
  fill_vector(src, 0);
  src_mf.copy_locally_owned_data_from(src);

  const double tolerance_inverse =
    (std::is_same_v<Number, double> ? 1e3 : 1e1) * tolerance<Number>();
  laplace_op.apply_inverse_block_diagonal(reference, src_mf);
  VectorType block_times_inverse;
  op.initialize_dof_vector(block_times_inverse);
  // the inverse of the surrogate is exact, so correction steps with the
  // actual cell block do not change it
  for(const unsigned int n_iterations : {0, 1})
  {
    op.set_block_inverse_iterations(n_iterations);
    op.apply_block_inverse(dst, src);
    VectorType difference(reference);
    difference.copy_locally_owned_data_from(dst);
    difference -= reference;
    print_check(pcout,
                "block inverse (" + std::to_string(n_iterations) +
                  " corrections) vs. Poisson::LaplaceOperator",
                difference.linfty_norm(),
                reference.linfty_norm(),
                tolerance_inverse);
  }

  // the cell block of the operator applied to the inverse gives identity
  op.apply_block_diagonal(block_times_inverse, dst);
  block_times_inverse -= src;
  print_check(pcout,
              "cell block times block inverse = identity",
              block_times_inverse.linfty_norm(),
              src.linfty_norm(),
              tolerance_inverse);
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
  // the matrix-based block diagonal of the reference operator is only
  // available in serial
  if(dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1)
    for(const unsigned int degree : {1, 3})
    {
      test_block_inverse_cartesian<double>(degree, 1);
      test_block_inverse_cartesian<float>(degree, 1);
    }
}
