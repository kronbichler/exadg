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

// Test the multigrid transfer operators of the pressure Poisson solver on
// extruded meshes with hanging nodes, based on the compressed DoF indices
// of LaplaceOperator::LaplaceOperatorFE:
// - MGTwoLevelTransferFromOperator for the polynomial transfer on a mesh
//   with hanging nodes, compared to dealii::MGTwoLevelTransfer,
// - MGTwoLevelTransferAnisotropicNested between two separate, nested
//   triangulations that both have hanging nodes, compared to
//   dealii::MGTwoLevelTransferNonNested.
// The fine mesh has two global refinements and one more refinement close to
// the bottom wall, the coarse mesh one global refinement and one more
// refinement in a wider region close to the bottom wall, such that cells
// of the fine mesh are either identical to a cell of the coarse mesh or a
// child of it, with hanging nodes on both sides. The comparison is done on
// the unconstrained entries, and the constrained entries of the fine vector
// are set to zero before the restriction.

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

#include <exadg/incompressible_navier_stokes/time_integration/poisson_solver.h>
#include <exadg/operators/constraints.h>

#include "extruded_operators_common.h"

using namespace ExtrudedTest;

// Mesh as in create_grid() with the cells below the given height refined
// once more
void
create_refined_grid(dealii::parallel::distributed::Triangulation<dim> & tria,
                    const unsigned int                                  n_refinements,
                    const double                                        refine_below)
{
  PeriodicFaces periodic_faces;
  create_grid(tria, periodic_faces, n_refinements);
  for(const auto & cell : tria.active_cell_iterators())
    if(cell->is_locally_owned() && cell->center()[1] < refine_below)
      cell->set_refine_flag();
  tria.execute_coarsening_and_refinement();
}

// DoFHandler, constraints and operator of one multigrid level, set up as in
// PoissonPreconditionerMG
template<typename Number>
struct Level
{
  Level(const dealii::Triangulation<dim> & tria,
        const dealii::Mapping<dim> &       mapping,
        const unsigned int                 degree)
    : fe(degree), dof_handler(tria)
  {
    // Vectorization categories as in PoissonPreconditionerMG, but in reverse
    // order, such that the refined cells, which contain the hanging nodes,
    // are visited before the cells on the coarse side of the hanging-node
    // constraints. This checks that the transfers do not rely on a
    // particular order of the cells, as MatrixFree may reorder them.
    std::vector<unsigned int> cell_vectorization_category(tria.n_active_cells());
    unsigned int              count = 0;
    for(const auto & cell : tria.active_cell_iterators())
      count += cell->is_locally_owned();
    for(const auto & cell : tria.active_cell_iterators())
      if(cell->is_locally_owned())
        cell_vectorization_category[cell->active_cell_index()] = --count;

    std::vector<
      dealii::GridTools::PeriodicFacePair<typename dealii::DoFHandler<dim>::cell_iterator>>
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

    op.reinit(mapping,
              dof_handler,
              constraints,
              cell_vectorization_category,
              dealii::QGauss<1>(degree + 1));
  }

  dealii::FE_Q<dim>                               fe;
  dealii::DoFHandler<dim>                         dof_handler;
  dealii::AffineConstraints<Number>               constraints;
  LaplaceOperator::LaplaceOperatorFE<dim, Number> op;
};

// Compare the unconstrained entries of two vectors with the same locally
// owned range
template<typename VectorType>
void
compare_unconstrained(dealii::ConditionalOStream &      pcout,
                      const std::string &               name,
                      const VectorType &                result,
                      const VectorType &                reference,
                      const std::vector<unsigned int> & constrained_dofs,
                      const double                      tolerance)
{
  std::vector<char> is_constrained(result.locally_owned_size(), 0);
  for(const unsigned int i : constrained_dofs)
    is_constrained[i] = 1;
  double max_error = 0, max_value = 0;
  for(unsigned int i = 0; i < result.locally_owned_size(); ++i)
    if(!is_constrained[i])
    {
      max_error =
        std::max<double>(max_error, std::abs(result.local_element(i) - reference.local_element(i)));
      max_value = std::max<double>(max_value, std::abs(reference.local_element(i)));
    }
  const MPI_Comm comm = result.get_mpi_communicator();
  print_check(pcout,
              name,
              dealii::Utilities::MPI::max(max_error, comm),
              dealii::Utilities::MPI::max(max_value, comm),
              tolerance);
}

// Run prolongation and restriction of the given transfer and the reference
template<typename Number>
void
check_transfer(
  dealii::ConditionalOStream & pcout,
  const dealii::MGTwoLevelTransferBase<dim, dealii::LinearAlgebra::distributed::Vector<Number>> &
    transfer,
  const dealii::MGTwoLevelTransferBase<dim, dealii::LinearAlgebra::distributed::Vector<Number>> &
                        reference_transfer,
  const Level<Number> & fine,
  const Level<Number> & coarse)
{
  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

  VectorType src_coarse, dst_fine, reference_fine;
  coarse.op.initialize_dof_vector(src_coarse);
  fine.op.initialize_dof_vector(dst_fine);
  fine.op.initialize_dof_vector(reference_fine);
  fill_vector(src_coarse, 2);
  transfer.prolongate_and_add(dst_fine, src_coarse);
  reference_transfer.prolongate_and_add(reference_fine, src_coarse);
  compare_unconstrained(pcout,
                        "prolongate",
                        dst_fine,
                        reference_fine,
                        fine.op.get_matrix_free().get_constrained_dofs(),
                        tolerance<Number>());

  VectorType src_fine, dst_coarse, reference_coarse;
  fine.op.initialize_dof_vector(src_fine);
  coarse.op.initialize_dof_vector(dst_coarse);
  coarse.op.initialize_dof_vector(reference_coarse);
  fill_vector(src_fine, 3);
  for(const unsigned int i : fine.op.get_matrix_free().get_constrained_dofs())
    src_fine.local_element(i) = 0.;
  transfer.restrict_and_add(dst_coarse, src_fine);
  reference_transfer.restrict_and_add(reference_coarse, src_fine);
  compare_unconstrained(pcout,
                        "restrict",
                        dst_coarse,
                        reference_coarse,
                        coarse.op.get_matrix_free().get_constrained_dofs(),
                        tolerance<Number>());
}

template<typename Number>
void
test(const unsigned int degree)
{
  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

  const MPI_Comm             comm = MPI_COMM_WORLD;
  dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(comm) == 0);

  dealii::parallel::distributed::Triangulation<dim> tria_fine(comm), tria_coarse(comm);
  create_refined_grid(tria_fine, 2, 0.3 * length_y);
  create_refined_grid(tria_coarse, 1, 0.3 * length_y);
  const auto mapping_fine   = create_mapping(tria_fine, 3);
  const auto mapping_coarse = create_mapping(tria_coarse, 3);

  pcout << "FE degree " << degree << ", " << number_name<Number>() << ", "
        << tria_fine.n_global_active_cells() << " fine cells, "
        << tria_coarse.n_global_active_cells() << " coarse cells" << std::endl;

  const Level<Number> fine(tria_fine, *mapping_fine, degree);
  AssertThrow(fine.op.has_hanging_nodes(), dealii::ExcInternalError());

  // polynomial transfer on the fine mesh
  if(degree > 1)
  {
    const unsigned int  degree_coarse = LaplaceOperator::get_coarser_fe_degree(degree);
    const Level<Number> coarse_p(tria_fine, *mapping_fine, degree_coarse);
    LaplaceOperator::MGTwoLevelTransferFromOperator<dim, Number> transfer(fine.op, coarse_p.op);
    dealii::MGTwoLevelTransfer<dim, VectorType>                  reference;
    reference.reinit_polynomial_transfer(fine.dof_handler,
                                         coarse_p.dof_handler,
                                         fine.constraints,
                                         coarse_p.constraints);
    pcout << " p-transfer to degree " << degree_coarse << std::endl;
    check_transfer(pcout, transfer, reference, fine, coarse_p);
  }

  // anisotropic nested transfer between the two meshes
  {
    const Level<Number> coarse_h(tria_coarse, *mapping_coarse, degree);
    AssertThrow(coarse_h.op.has_hanging_nodes(), dealii::ExcInternalError());
    LaplaceOperator::MGTwoLevelTransferAnisotropicNested<dim, Number> transfer(fine.op,
                                                                               coarse_h.op);
    dealii::MGTwoLevelTransferNonNested<dim, VectorType>              reference;
    dealii::MappingQ1<dim>                                            mapping_q1;
    reference.reinit(fine.dof_handler,
                     coarse_h.dof_handler,
                     mapping_q1,
                     mapping_q1,
                     fine.constraints,
                     coarse_h.constraints);
    pcout << " h-transfer to coarse mesh" << std::endl;
    check_transfer(pcout, transfer, reference, fine, coarse_h);
  }
  pcout << std::endl;
}

int
main(int argc, char ** argv)
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  for(const unsigned int degree : {1, 2, 3, 4})
  {
    test<double>(degree);
    test<float>(degree);
  }
}
