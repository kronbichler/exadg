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

// Test RTOperator::RaviartThomasOperator::compute_diagonal() against the
// diagonal obtained by applying vmult() to each unit vector. The mesh is
// periodic in x and z direction and has no-slip walls in y direction, with a
// mapping that deforms the xy plane. The error is reported separately for
// the different kinds of DoFs, because the face integrals contribute
// differently to them: the DoFs of the normal component on faces are shared
// between the two adjacent cells, and the interior DoFs of cells at the walls
// see the boundary condition of the viscous term.

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/numerics/vector_tools.h>

#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/momentum_operator_rt.h>

#include "extruded_operators_common.h"

#include <algorithm>
#include <array>
#include <limits>
#include <tuple>

using namespace ExtrudedTest;

enum DoFKind : unsigned int
{
  interior_away_from_walls,
  interior_at_walls,
  face_normal_x,
  face_normal_y,
  face_normal_z,
  n_kinds
};

const std::array<std::string, n_kinds> kind_names = {{"interior DoFs, cells away from walls",
                                                      "interior DoFs, cells at walls",
                                                      "face DoFs, normal x",
                                                      "face DoFs, normal y",
                                                      "face DoFs, normal z"}};

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

  // same constraints and renumbering as in extruded_momentum_operator_rt.cc
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
  // the face DoFs in the order of the RT operator
  Helper::renumber_rt_face_dofs_in_operator_order(dof_handler);
  make_constraints();

  RTOperator::RaviartThomasOperator<dim, Number> op;
  op.reinit(*mapping,
            dof_handler,
            constraints,
            Helper::compute_vectorization_category(tria),
            dealii::QGauss<1>(degree + 1),
            true);
  op.set_penalty_parameters(1.3);

  VectorType src, dst, diagonal, reference;
  op.initialize_dof_vector(src);
  op.initialize_dof_vector(dst);
  op.initialize_dof_vector(diagonal);
  op.initialize_dof_vector(reference);

  pcout << "Raviart-Thomas degree " << degree << ", " << number_name<Number>() << ", "
        << tria.n_global_active_cells() << " cells, " << src.size() << " unconstrained DoFs"
        << std::endl;

  // The vector of the operator holds the locally owned unconstrained DoFs,
  // which come first in the locally owned range of the DoFHandler after the
  // renumbering above, so the n-th locally owned index of the DoFHandler
  // corresponds to the n-th locally owned entry of the vector
  const dealii::IndexSet &  owned_dofs = dof_handler.locally_owned_dofs();
  std::vector<unsigned int> kind(src.locally_owned_size(), dealii::numbers::invalid_unsigned_int);
  const unsigned int        dofs_per_face = fe.n_dofs_per_face();
  std::vector<dealii::types::global_dof_index> dof_indices(fe.n_dofs_per_cell());
  for(const auto & cell : dof_handler.active_cell_iterators())
    if(cell->is_locally_owned())
    {
      cell->get_dof_indices(dof_indices);
      bool at_wall = false;
      for(const unsigned int f : {2U, 3U})
        if(cell->at_boundary(f) && cell->face(f)->boundary_id() == wall_id)
          at_wall = true;
      for(unsigned int i = 0; i < dof_indices.size(); ++i)
        if(owned_dofs.is_element(dof_indices[i]))
        {
          const unsigned int local = owned_dofs.index_within_set(dof_indices[i]);
          if(local >= kind.size())
            continue;
          const unsigned int face = i / dofs_per_face;
          kind[local]             = (face < 2 * dim) ? face_normal_x + face / 2 :
                                                       (at_wall ? interior_at_walls : interior_away_from_walls);
        }
    }
  for(const unsigned int k : kind)
    AssertThrow(k != dealii::numbers::invalid_unsigned_int,
                dealii::ExcMessage("Could not classify locally owned DoF"));

  // Reference: entry i of the operator applied to the i-th unit vector. To
  // keep the run time in debug mode acceptable, only every third DoF is
  // checked, and the operator being linear in the two factors, the
  // reference of the mass and Laplace parts is computed once and combined
  // for the different factors.
  constexpr unsigned int sample_stride = 3;
  VectorType             reference_mass, reference_laplace;
  op.initialize_dof_vector(reference_mass);
  op.initialize_dof_vector(reference_laplace);
  for(const auto & [factor_mass, factor_laplace, result] :
      {std::tuple<double, double, VectorType *>{1., 0., &reference_mass},
       std::tuple<double, double, VectorType *>{0., 1., &reference_laplace}})
  {
    op.set_parameters(factor_mass, factor_laplace);
    for(dealii::types::global_dof_index i = 0; i < src.size(); i += sample_stride)
    {
      src = 0;
      if(src.in_local_range(i))
        src(i) = 1;
      op.vmult(dst, src);
      if(dst.in_local_range(i))
        (*result)(i) = dst(i);
    }
  }

  std::array<unsigned int, n_kinds> count{};
  for(const unsigned int k : kind)
    ++count[k];
  for(unsigned int k = 0; k < n_kinds; ++k)
    count[k] = dealii::Utilities::MPI::sum(count[k], comm);

  for(const auto & [factor_mass, factor_laplace] :
      std::vector<std::pair<double, double>>{{1., 0.}, {0., 1.}, {2.5, 0.3}})
  {
    pcout << " factor mass " << factor_mass << ", factor Laplace " << factor_laplace << std::endl;
    op.set_parameters(factor_mass, factor_laplace);

    op.compute_diagonal(diagonal);
    reference.equ(factor_mass, reference_mass);
    reference.add(factor_laplace, reference_laplace);

    std::array<double, n_kinds> max_error{}, max_reference{};
    std::array<double, n_kinds> min_ratio, max_ratio;
    min_ratio.fill(std::numeric_limits<double>::max());
    max_ratio.fill(-std::numeric_limits<double>::max());
    for(unsigned int i = 0; i < kind.size(); ++i)
    {
      if(src.get_partitioner()->local_to_global(i) % sample_stride != 0)
        continue;
      const unsigned int k   = kind[i];
      const double       ref = reference.local_element(i);
      const double       val = diagonal.local_element(i);
      max_error[k]           = std::max(max_error[k], std::abs(val - ref));
      max_reference[k]       = std::max(max_reference[k], std::abs(ref));
      min_ratio[k]           = std::min(min_ratio[k], val / ref);
      max_ratio[k]           = std::max(max_ratio[k], val / ref);
    }
    for(unsigned int k = 0; k < n_kinds; ++k)
    {
      max_error[k]     = dealii::Utilities::MPI::max(max_error[k], comm);
      max_reference[k] = dealii::Utilities::MPI::max(max_reference[k], comm);
      min_ratio[k]     = dealii::Utilities::MPI::min(min_ratio[k], comm);
      max_ratio[k]     = dealii::Utilities::MPI::max(max_ratio[k], comm);
    }

    for(unsigned int k = 0; k < n_kinds; ++k)
    {
      if(count[k] == 0)
        continue;
      print_check(pcout,
                  kind_names[k] + " (" + std::to_string(count[k]) + ")",
                  max_error[k],
                  max_reference[k],
                  tolerance<Number>());
      pcout << "    ratio diagonal / reference in [" << std::fixed << std::setprecision(3)
            << min_ratio[k] << ", " << max_ratio[k] << "]" << std::defaultfloat
            << std::setprecision(6) << std::endl;
    }
  }
  pcout << std::endl;
}

int
main(int argc, char ** argv)
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  for(const unsigned int degree : {2, 3})
  {
    test<double>(degree, 1);
    test<float>(degree, 1);
  }
}
