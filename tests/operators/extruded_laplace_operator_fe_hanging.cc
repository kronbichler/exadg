/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2025 by the ExaDG authors
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

// Test LaplaceOperator::LaplaceOperatorFE on a mesh with hanging nodes: the
// channel of the other extruded tests, where the cells close to the bottom
// wall are refined once more, giving hanging nodes on faces and edges also
// across the periodic boundaries, and cells of different extent in z
// direction. The cell-local values read by the operator, which include the
// hanging-node interpolation, are compared to the values of
// FEEvaluation::read_dof_values() on a separate MatrixFree object. Also
// check that the indices without the hanging-node replacement, as used on
// the fine side of transfer operators, visit all unconstrained DoFs. The
// matrix-vector product and the diagonal are compared to the generic
// Poisson::LaplaceOperator, and vmult_scratch() and the Chebyshev iteration
// are checked as in the test without hanging nodes.

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/laplace_operator_extruded.h>
#include <exadg/operators/constraints.h>
#include <exadg/poisson/spatial_discretization/laplace_operator.h>

#include "extruded_operators_common.h"

#include <map>

using namespace ExtrudedTest;

template<typename Number>
void
test(const unsigned int degree, const unsigned int n_refinements)
{
  using VectorType               = dealii::LinearAlgebra::distributed::Vector<Number>;
  constexpr unsigned int n_lanes = dealii::VectorizedArray<Number>::size();

  const MPI_Comm             comm = MPI_COMM_WORLD;
  dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(comm) == 0);

  dealii::parallel::distributed::Triangulation<dim> tria(comm);
  PeriodicFaces                                     periodic_faces;
  create_grid(tria, periodic_faces, n_refinements);
  for(const auto & cell : tria.active_cell_iterators())
    if(cell->is_locally_owned() && cell->center()[1] < 0.3 * length_y)
      cell->set_refine_flag();
  tria.execute_coarsening_and_refinement();
  const auto mapping = create_mapping(tria, 3);

  dealii::FE_Q<dim>                 fe(degree);
  dealii::DoFHandler<dim>           dof_handler(tria);
  dealii::AffineConstraints<Number> constraints;

  // vectorization category and constraints as on the multigrid levels in
  // PoissonPreconditionerMG
  std::vector<unsigned int> cell_vectorization_category(tria.n_active_cells());
  {
    unsigned int count = 0;
    for(const auto & cell : tria.active_cell_iterators())
      if(cell->is_locally_owned())
        cell_vectorization_category[cell->active_cell_index()] = count++;
  }
  std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::DoFHandler<dim>::cell_iterator>>
             periodic_dof_faces;
  const auto make_constraints = [&]() {
    constraints.reinit(dof_handler.locally_owned_dofs(),
                       dealii::DoFTools::extract_locally_relevant_dofs(dof_handler));
    dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);
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

  const auto & masks           = op.get_hanging_node_masks();
  unsigned int n_hanging_cells = 0;
  for(const auto & batch : masks)
    for(const auto mask : batch)
      n_hanging_cells +=
        (mask != dealii::internal::MatrixFreeFunctions::unconstrained_compressed_constraint_kind);
  pcout << "  has hanging nodes                             : " << op.has_hanging_nodes()
        << std::endl;
  pcout << "  cells with hanging nodes                      : "
        << dealii::Utilities::MPI::sum(n_hanging_cells, comm) << std::endl;

  // reference values from FEEvaluation, which applies the hanging-node
  // constraints in read_dof_values(), stored by cell id because the cell
  // order differs between the two MatrixFree objects
  dealii::MatrixFree<dim, Number> matrix_free;
  matrix_free.reinit(
    *mapping,
    dof_handler,
    constraints,
    dealii::QGauss<dim>(degree + 1),
    create_matrix_free_data<Number>(
      ExaDG::Poisson::Operators::LaplaceKernel<dim, Number>::get_mapping_flags(false, true)));

  VectorType src, src_mf;
  op.initialize_dof_vector(src);
  matrix_free.initialize_dof_vector(src_mf);
  fill_vector(src, 1);
  // constrained entries are not read by either implementation
  src_mf.copy_locally_owned_data_from(src);
  src.update_ghost_values();
  src_mf.update_ghost_values();

  std::map<dealii::CellId, std::vector<Number>> reference_values;
  {
    dealii::FEEvaluation<dim, -1, 0, 1, Number> eval(matrix_free);
    for(unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
    {
      eval.reinit(cell);
      eval.read_dof_values(src_mf);
      for(unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(cell); ++v)
      {
        std::vector<Number> values(eval.dofs_per_cell);
        for(unsigned int i = 0; i < eval.dofs_per_cell; ++i)
          values[i] = eval.begin_dof_values()[i][v];
        reference_values[matrix_free.get_cell_iterator(cell, v)->id()] = std::move(values);
      }
    }
  }

  const auto &                                 mf_op         = op.get_matrix_free();
  const unsigned int                           dofs_per_cell = fe.dofs_per_cell;
  double                                       max_error = 0, max_value = 0;
  std::vector<dealii::VectorizedArray<Number>> values(dofs_per_cell);
  // the single-lane read includes the hanging-node interpolation
  std::vector<Number> lane_values(dofs_per_cell);
  for(unsigned int cell = 0; cell < mf_op.n_cell_batches(); ++cell)
  {
    for(unsigned int v = 0; v < n_lanes; ++v)
    {
      if(v < mf_op.n_active_entries_per_cell_batch(cell))
        op.read_dof_values_compressed(src, cell, v, lane_values.data());
      else
        std::fill(lane_values.begin(), lane_values.end(), Number());
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        values[i][v] = lane_values[i];
    }

    for(unsigned int v = 0; v < mf_op.n_active_entries_per_cell_batch(cell); ++v)
    {
      const std::vector<Number> & reference =
        reference_values.at(mf_op.get_cell_iterator(cell, v)->id());
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        max_error = std::max<double>(max_error, std::abs(values[i][v] - reference[i]));
        max_value = std::max<double>(max_value, std::abs(reference[i]));
      }
    }
  }
  print_check(pcout,
              "read with hanging nodes vs. FEEvaluation",
              dealii::Utilities::MPI::max(max_error, comm),
              dealii::Utilities::MPI::max(max_value, comm),
              tolerance<Number>());

  // The indices without hanging-node replacement must visit each locally
  // owned unconstrained DoF and no constrained one.
  {
    const std::vector<unsigned int> own_indices =
      op.get_compressed_dof_indices_without_hanging_nodes();
    const unsigned int n_owned = src.locally_owned_size();
    std::vector<char>  visited(n_owned, 0);
    // the compressed format stores the first index of each of the 3^dim
    // entities (2^dim vertices for degree 1), whose length is (degree-1) to
    // the power of the number of directions in the interior of the cell
    const unsigned int n_stored = dealii::Utilities::pow(degree == 1 ? 2 : 3, dim);
    for(unsigned int k = 0; k < own_indices.size(); ++k)
    {
      unsigned int length = 1;
      if(degree > 1)
        for(unsigned int d = 0, j = (k / n_lanes) % n_stored; d < dim; ++d, j /= 3)
          if(j % 3 == 1)
            length *= degree - 1;
      for(unsigned int i = 0; i < length; ++i)
        if(own_indices[k] + i < n_owned)
          visited[own_indices[k] + i] = 1;
    }
    std::vector<char> is_constrained(n_owned, 0);
    for(const unsigned int index : mf_op.get_constrained_dofs())
      is_constrained[index] = 1;
    unsigned int n_errors = 0;
    for(unsigned int i = 0; i < n_owned; ++i)
      if(visited[i] == is_constrained[i])
        ++n_errors;
    pcout << "  own indices: wrongly visited/missed DoFs      : "
          << dealii::Utilities::MPI::sum(n_errors, comm) << std::endl;
  }

  // reference operator from ExaDG, which treats the hanging nodes via
  // FEEvaluation
  ExaDG::Poisson::LaplaceOperatorData<0, dim> laplace_data;
  auto boundary_descriptor = std::make_shared<ExaDG::Poisson::BoundaryDescriptor<0, dim>>();
  boundary_descriptor->neumann_bc.emplace(wall_id,
                                          std::make_shared<dealii::Functions::ZeroFunction<dim>>());
  laplace_data.bc = boundary_descriptor;
  ExaDG::Poisson::LaplaceOperator<dim, Number, 1> laplace_op;
  laplace_op.initialize(matrix_free, constraints, laplace_data, false);

  src.zero_out_ghost_values();
  src_mf.zero_out_ghost_values();
  VectorType dst, reference;
  op.initialize_dof_vector(dst);
  matrix_free.initialize_dof_vector(reference);
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

  check_vmult_scratch(pcout, op, src, dst, tolerance<Number>());
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
    test<double>(degree, 1);
    test<float>(degree, 1);
  }
}
