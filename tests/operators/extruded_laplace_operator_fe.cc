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

// Wrapper of the block-Jacobi preconditioner of LaplaceOperatorFE for
// dealii::PreconditionChebyshev
template<typename Operator>
struct BlockInverseFE
{
  template<typename VectorType>
  void
  vmult(VectorType & dst, const VectorType & src) const
  {
    op->apply_block_inverse(dst, src);
  }
  const Operator * op;
};

// Checks of the block-Jacobi preconditioner of LaplaceOperatorFE: the
// blocks form a partition of the unconstrained unknowns, the inverse is
// symmetric and positive, the vector-based Chebyshev iteration agrees with
// dealii::PreconditionChebyshev, and on a Cartesian mesh (in serial) the
// inverse of the surrogate is the exact inverse of the blocks
template<typename Operator>
void
check_block_inverse_fe(dealii::ConditionalOStream & pcout,
                       const Operator &             op,
                       const bool                   check_exactness)
{
  using VectorType = typename Operator::VectorType;
  using Number     = typename VectorType::value_type;

  // partition of the unknowns
  {
    VectorType count;
    op.initialize_dof_vector(count);
    for(unsigned int b = 0; b < op.n_blocks(); ++b)
      for(const unsigned int i : op.get_block_indices(b))
        count.local_element(i) += 1;
    count.compress(dealii::VectorOperation::add);
    std::vector<bool> constrained(count.locally_owned_size(), false);
    for(const unsigned int i : op.get_matrix_free().get_constrained_dofs())
      constrained[i] = true;
    unsigned int n_wrong = 0;
    for(unsigned int i = 0; i < count.locally_owned_size(); ++i)
      if(count.local_element(i) != (constrained[i] ? 0 : 1))
        ++n_wrong;
    n_wrong = dealii::Utilities::MPI::sum(n_wrong, count.get_mpi_communicator());
    pcout << "  " << std::left << std::setw(46) << "blocks: unknowns not in exactly one block"
          << ": " << n_wrong << std::endl;
  }

  // symmetry and positivity
  VectorType u, v, inverse_u, inverse_v;
  op.initialize_dof_vector(u);
  op.initialize_dof_vector(v);
  op.initialize_dof_vector(inverse_u);
  op.initialize_dof_vector(inverse_v);
  fill_vector(u, 1);
  fill_vector(v, 2);
  for(const unsigned int i : op.get_matrix_free().get_constrained_dofs())
  {
    u.local_element(i) = 0;
    v.local_element(i) = 0;
  }
  op.apply_block_inverse(inverse_u, u);
  op.apply_block_inverse(inverse_v, v);
  const double uv = inverse_u * v;
  const double vu = inverse_v * u;
  print_check(pcout,
              "block inverse symmetric",
              std::abs(uv - vu),
              std::abs(uv) + std::abs(vu),
              10. * tolerance<Number>());
  pcout << "  " << std::left << std::setw(46) << "block inverse positive"
        << ": " << ((inverse_u * u) > 0 ? "OK" : "FAILED") << std::endl;

  // exactness on single blocks
  if(check_exactness && dealii::Utilities::MPI::n_mpi_processes(u.get_mpi_communicator()) == 1)
  {
    double max_error = 0, max_value = 0;
    for(const unsigned int b : {0u, op.n_blocks() / 3, (2 * op.n_blocks()) / 3, op.n_blocks() - 1})
    {
      const auto indices = op.get_block_indices(b);
      u                  = 0;
      for(unsigned int i = 0; i < indices.size(); ++i)
        u.local_element(indices[i]) = std::sin(1. + i);
      op.apply_block_inverse(inverse_u, u);
      op.vmult(v, inverse_u);
      for(const unsigned int i : indices)
      {
        max_error = std::max(max_error, std::abs<double>(v.local_element(i) - u.local_element(i)));
        max_value = std::max(max_value, std::abs<double>(u.local_element(i)));
      }
    }
    print_check(pcout,
                "block times block inverse = identity",
                max_error,
                max_value,
                (std::is_same_v<Number, double> ? 1e3 : 1e1) * tolerance<Number>());
  }

  // vector-based Chebyshev iteration
  using Scratch = ExaDG::PreconditionChebyshevScratch<Operator>;
  using DealII  = dealii::PreconditionChebyshev<Operator, VectorType, BlockInverseFE<Operator>>;
  VectorType rhs, initial, reference, result;
  op.initialize_dof_vector(rhs);
  op.initialize_dof_vector(initial);
  fill_vector(rhs, 1);
  fill_vector(initial, 2);
  for(const unsigned int i : op.get_matrix_free().get_constrained_dofs())
  {
    rhs.local_element(i)     = 0;
    initial.local_element(i) = 0;
  }
  for(const auto type : {Scratch::PolynomialType::first_kind, Scratch::PolynomialType::fourth_kind})
  {
    typename DealII::AdditionalData data_dealii;
    data_dealii.preconditioner      = std::make_shared<BlockInverseFE<Operator>>();
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
    data.polynomial_type       = type;
    data.global_preconditioner = [&op](VectorType & dst, const VectorType & src) {
      op.apply_block_inverse(dst, src);
    };
    data.degree          = data_dealii.degree;
    data.smoothing_range = data_dealii.smoothing_range;
    data.max_eigenvalue  = data_dealii.max_eigenvalue;
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
    }
  }
}

// Wrapper of the vectorized block-Jacobi preconditioner of
// LaplaceOperatorFE for dealii::PreconditionChebyshev
template<typename Operator>
struct BlockInverseVectorizedFE
{
  template<typename VectorType>
  void
  vmult(VectorType & dst, const VectorType & src) const
  {
    op->apply_block_inverse_vectorized(dst, src);
  }
  const Operator * op;
};

// Checks of the vectorized block-Jacobi preconditioner embedded in the
// cell loop: in serial, the application on whole vectors agrees with the
// basic version of setup_block_inverse(), which must be set up (in
// parallel, the vectorized version drops the unknowns of other
// processes); it is symmetric and positive; and the Chebyshev iteration
// with the embedded block updates agrees with dealii::PreconditionChebyshev
// with the vectorized version applied on whole vectors
template<typename Operator>
void
check_embedded_block_inverse_fe(dealii::ConditionalOStream & pcout, const Operator & op)
{
  using VectorType = typename Operator::VectorType;
  using Number     = typename VectorType::value_type;

  VectorType u, reference, result;
  op.initialize_dof_vector(u);
  op.initialize_dof_vector(reference);
  op.initialize_dof_vector(result);
  fill_vector(u, 3);
  for(const unsigned int i : op.get_matrix_free().get_constrained_dofs())
    u.local_element(i) = 0;
  if(dealii::Utilities::MPI::n_mpi_processes(u.get_mpi_communicator()) == 1)
  {
    op.apply_block_inverse(reference, u);
    op.apply_block_inverse_vectorized(result, u);
    result -= reference;
    print_check(pcout,
                "vectorized vs. basic block inverse",
                result.linfty_norm(),
                reference.linfty_norm(),
                10. * tolerance<Number>());
  }
  {
    VectorType v, inverse_u, inverse_v;
    op.initialize_dof_vector(v);
    op.initialize_dof_vector(inverse_u);
    op.initialize_dof_vector(inverse_v);
    fill_vector(v, 4);
    for(const unsigned int i : op.get_matrix_free().get_constrained_dofs())
      v.local_element(i) = 0;
    op.apply_block_inverse_vectorized(inverse_u, u);
    op.apply_block_inverse_vectorized(inverse_v, v);
    const double uv = inverse_u * v;
    const double vu = inverse_v * u;
    print_check(pcout,
                "vectorized block inverse symmetric",
                std::abs(uv - vu),
                std::abs(uv) + std::abs(vu),
                10. * tolerance<Number>());
    pcout << "  " << std::left << std::setw(46) << "vectorized block inverse positive"
          << ": " << ((inverse_u * u) > 0 ? "OK" : "FAILED") << std::endl;
  }

  using Scratch = ExaDG::PreconditionChebyshevScratch<Operator>;
  using DealII =
    dealii::PreconditionChebyshev<Operator, VectorType, BlockInverseVectorizedFE<Operator>>;
  VectorType rhs, initial;
  op.initialize_dof_vector(rhs);
  op.initialize_dof_vector(initial);
  fill_vector(rhs, 1);
  fill_vector(initial, 2);
  for(const unsigned int i : op.get_matrix_free().get_constrained_dofs())
  {
    rhs.local_element(i)     = 0;
    initial.local_element(i) = 0;
  }
  for(const auto type : {Scratch::PolynomialType::first_kind, Scratch::PolynomialType::fourth_kind})
  {
    typename DealII::AdditionalData data_dealii;
    data_dealii.preconditioner      = std::make_shared<BlockInverseVectorizedFE<Operator>>();
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
    data.polynomial_type            = type;
    data.use_embedded_block_inverse = true;
    data.degree                     = data_dealii.degree;
    data.smoothing_range            = data_dealii.smoothing_range;
    data.max_eigenvalue             = data_dealii.max_eigenvalue;
    Scratch chebyshev;
    chebyshev.initialize(op, data);

    for(const unsigned int degree : {1, 2, 4})
    {
      chebyshev_dealii.set_degree(degree);
      chebyshev.set_degree(degree);
      const std::string name = std::string("Chebyshev embedded block ") +
                               (type == Scratch::PolynomialType::first_kind ? "1st" : "4th") +
                               " kind degree " + std::to_string(degree) + " ";
      op.initialize_dof_vector(reference);
      op.initialize_dof_vector(result);
      chebyshev_dealii.vmult(reference, rhs);
      chebyshev.vmult(result, rhs);
      result -= reference;
      print_check(pcout,
                  name + "vmult",
                  result.linfty_norm(),
                  reference.linfty_norm(),
                  10. * tolerance<Number>());

      reference = initial;
      result    = initial;
      chebyshev_dealii.step(reference, rhs);
      chebyshev.step(result, rhs);
      result -= reference;
      print_check(pcout,
                  name + "step",
                  result.linfty_norm(),
                  reference.linfty_norm(),
                  10. * tolerance<Number>());

      op.initialize_dof_vector(reference);
      op.initialize_dof_vector(result);
      const double norm_reference = chebyshev_dealii.vmult_with_last_residual_norm(reference, rhs);
      const double norm           = chebyshev.vmult_with_last_residual_norm(result, rhs);
      print_check(pcout,
                  name + "last residual norm",
                  std::abs(norm - norm_reference),
                  norm_reference,
                  10. * tolerance<Number>());
    }
  }
}

template<typename Number>
void
test(const unsigned int degree, const unsigned int n_refinements, const bool cartesian = false)
{
  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

  const MPI_Comm             comm = MPI_COMM_WORLD;
  dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(comm) == 0);

  dealii::parallel::distributed::Triangulation<dim> tria(comm);
  PeriodicFaces                                     periodic_faces;
  create_grid(tria, periodic_faces, n_refinements);
  const std::shared_ptr<dealii::Mapping<dim>> mapping =
    cartesian ? std::make_shared<dealii::MappingQ<dim>>(1) : create_mapping(tria, 3);

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

  pcout << (cartesian ? "Cartesian mesh, " : "") << "FE degree " << degree << ", "
        << number_name<Number>() << ", " << tria.n_global_active_cells() << " cells, "
        << dof_handler.n_dofs() << " DoFs" << std::endl;

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

  // diagonal, with the constrained entries set to one in both cases
  VectorType inverse_diagonal, reference_diagonal;
  op.compute_inverse_diagonal(inverse_diagonal);
  laplace_op.calculate_diagonal(reference_diagonal);
  difference.copy_locally_owned_data_from(inverse_diagonal);
  for(const unsigned int i : matrix_free.get_constrained_dofs())
    reference_diagonal.local_element(i) = 1.;
  for(unsigned int i = 0; i < reference_diagonal.locally_owned_size(); ++i)
    reference_diagonal.local_element(i) = Number(1.) / reference_diagonal.local_element(i);
  difference -= reference_diagonal;
  print_check(pcout,
              "inverse diagonal vs. Poisson::LaplaceOperator",
              difference.linfty_norm(),
              reference_diagonal.linfty_norm(),
              tolerance<Number>());

  // the first Chebyshev iteration is fused into the second one, first with
  // 'before_loop' on all entries before the loop, as the mesh is below the
  // default minimal size, then within the cell loop
  static_assert(
    ExaDG::internal::has_vmult_scratch_with_before_loop<std::decay_t<decltype(op)>>::value);
  check_chebyshev(
    pcout, op, inverse_diagonal, tolerance<Number>(), op.get_matrix_free().get_constrained_dofs());
  check_chebyshev_fourth_kind(
    pcout, op, inverse_diagonal, tolerance<Number>(), op.get_matrix_free().get_constrained_dofs());
  op.set_min_size_for_pre_list(0);
  pcout << "  with before_loop within the cell loop" << std::endl;
  check_chebyshev(
    pcout, op, inverse_diagonal, tolerance<Number>(), op.get_matrix_free().get_constrained_dofs());
  check_chebyshev_fourth_kind(
    pcout, op, inverse_diagonal, tolerance<Number>(), op.get_matrix_free().get_constrained_dofs());

  // block-Jacobi preconditioner with fast diagonalization
  if(degree > 1)
  {
    const bool supported = op.setup_block_inverse(constraints);
    pcout << "  " << std::left << std::setw(46) << "block inverse set up"
          << ": " << (supported ? "OK" : "FAILED") << std::endl;
    if(supported)
      check_block_inverse_fe(pcout, op, cartesian);

    // vectorized blocks embedded in the cell loop
    const bool supported_embedded = op.setup_block_batches();
    pcout << "  " << std::left << std::setw(46) << "vectorized block inverse set up"
          << ": " << (supported_embedded ? "OK" : "FAILED") << std::endl;
    if(supported && supported_embedded)
      check_embedded_block_inverse_fe(pcout, op);

    // blocks on patches of 2^dim cells
    const bool supported_patch = op.setup_patch_block_inverse(constraints);
    pcout << "  " << std::left << std::setw(46) << "patch block inverse set up"
          << ": " << (supported_patch ? "OK" : "not supported") << std::endl;
    if(supported_patch)
      check_block_inverse_fe(pcout, op, cartesian);
  }
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
  for(const unsigned int degree : {2, 3})
  {
    test<double>(degree, 2, true);
    test<float>(degree, 2, true);
  }
}
