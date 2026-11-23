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

#pragma once

// Common setup for the tests of the operators on extruded meshes used by
// the consistent splitting scheme with Raviart-Thomas velocity: a channel
// with curved walls in y direction that is periodic in x and z direction,
// where the geometry does not depend on z

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q_cache.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <exadg/operators/mapping_flags.h>
#include <exadg/solvers_and_preconditioners/preconditioners/chebyshev_scratch.h>

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace ExtrudedTest
{
constexpr unsigned int dim = 3;

constexpr double length_x = 3.0;
constexpr double length_y = 2.0;
constexpr double length_z = 1.5;

// boundary id of the walls in y direction, the periodic boundaries in x and
// z direction get ids 10, 11 and 14, 15
constexpr dealii::types::boundary_id wall_id = 0;

using PeriodicFaces = std::vector<
  dealii::GridTools::PeriodicFacePair<typename dealii::Triangulation<dim>::cell_iterator>>;

inline void
create_grid(dealii::parallel::distributed::Triangulation<dim> & tria,
            PeriodicFaces &                                     periodic_faces,
            const unsigned int                                  n_refinements)
{
  dealii::GridGenerator::subdivided_hyper_rectangle(
    tria, {3, 2, 2}, dealii::Point<dim>(), dealii::Point<dim>(length_x, length_y, length_z));
  for(const auto & cell : tria.cell_iterators())
    for(const unsigned int f : cell->face_indices())
      if(cell->at_boundary(f))
        cell->face(f)->set_boundary_id(f == 2 || f == 3 ? wall_id : 10 + f);

  dealii::GridTools::collect_periodic_faces(tria, 10, 11, 0, periodic_faces);
  dealii::GridTools::collect_periodic_faces(tria, 14, 15, 2, periodic_faces);
  tria.add_periodicity(periodic_faces);
  tria.refine_global(n_refinements);
}

// High-order mapping deforming the xy plane independently of z, with a
// curved wall at the bottom and periodic in x direction
inline std::shared_ptr<dealii::Mapping<dim>>
create_mapping(const dealii::Triangulation<dim> & tria, const unsigned int degree)
{
  auto mapping = std::make_shared<dealii::MappingQCache<dim>>(degree);
  mapping->initialize(
    dealii::MappingQ<dim>(1),
    tria,
    [](const typename dealii::Triangulation<dim>::cell_iterator &, const dealii::Point<dim> & p) {
      const double       pi = dealii::numbers::PI;
      dealii::Point<dim> displacement;
      displacement[0] = 0.05 * std::sin(2. * pi * p[0] / length_x) * std::sin(pi * p[1] / length_y);
      displacement[1] = 0.15 * std::cos(2. * pi * p[0] / length_x) * (1. - p[1] / length_y);
      return displacement;
    },
    true);
  return mapping;
}

template<typename Number>
typename dealii::MatrixFree<dim, Number>::AdditionalData
create_matrix_free_data(const ExaDG::MappingFlags & flags)
{
  typename dealii::MatrixFree<dim, Number>::AdditionalData data;
  data.tasks_parallel_scheme               = dealii::MatrixFree<dim, Number>::AdditionalData::none;
  data.mapping_update_flags                = flags.cells;
  data.mapping_update_flags_inner_faces    = flags.inner_faces;
  data.mapping_update_flags_boundary_faces = flags.boundary_faces;
  return data;
}

// Fill the locally owned entries with values in [-0.5, 0.5) depending on
// the global index only
template<typename VectorType>
void
fill_vector(VectorType & vector, const unsigned int seed)
{
  for(unsigned int i = 0; i < vector.locally_owned_size(); ++i)
  {
    const std::uint64_t index = vector.get_partitioner()->local_to_global(i);
    const std::uint64_t hash  = (index * 2654435761U + seed * 40503U) % 1024U;
    vector.local_element(i)   = static_cast<double>(hash) / 1024. - 0.5;
  }
}

// Relative tolerance of comparisons between different implementations
template<typename Number>
constexpr double
tolerance()
{
  return std::is_same_v<Number, double> ? 1e-12 : 1e-5;
}

inline void
print_check(dealii::ConditionalOStream & pcout,
            const std::string &          name,
            const double                 difference,
            const double                 reference,
            const double                 tolerance)
{
  pcout << "  " << std::left << std::setw(46) << name << ": ";
  if(difference <= tolerance * reference)
    pcout << "OK" << std::endl;
  else
    pcout << "FAILED, error " << difference << " vs. reference " << reference << std::endl;
}

template<typename Number>
std::string
number_name()
{
  return std::is_same_v<Number, double> ? "double" : "float";
}

// Compare vmult_scratch() of the operator against vmult() and check that
// each locally owned entry is passed to 'after_loop' exactly once in
// ranges starting at multiples of the chunk size with aligned pointers
template<typename Operator, typename VectorType>
void
check_vmult_scratch(dealii::ConditionalOStream & pcout,
                    const Operator &             op,
                    const VectorType &           src,
                    const VectorType &           reference,
                    const double                 tolerance)
{
  using Number                      = typename VectorType::value_type;
  constexpr unsigned int chunk_size = ExaDG::ScratchMatrixVectorProduct<Number>::chunk_size;

  const unsigned int        n_owned = reference.locally_owned_size();
  VectorType                result(reference);
  std::vector<unsigned int> n_finalized(n_owned, 0);
  unsigned int              n_misaligned = 0;
  const auto                after_loop =
    [&](const unsigned int begin, const unsigned int end, const Number * values) {
      if(begin % chunk_size != 0 || (end % chunk_size != 0 && end != n_owned) ||
         reinterpret_cast<std::uintptr_t>(values) % 64 != 0)
        ++n_misaligned;
      for(unsigned int i = begin; i < end; ++i)
      {
        result.local_element(i) = values[i - begin];
        ++n_finalized[i];
      }
    };
  op.vmult_scratch(src, after_loop);

  unsigned int n_wrong_count = 0;
  for(const unsigned int count : n_finalized)
    if(count != 1)
      ++n_wrong_count;
  const MPI_Comm comm = reference.get_mpi_communicator();
  n_wrong_count       = dealii::Utilities::MPI::sum(n_wrong_count, comm);
  n_misaligned        = dealii::Utilities::MPI::sum(n_misaligned, comm);

  pcout << "  " << std::left << std::setw(46) << "vmult_scratch: entries not finalized once"
        << ": " << n_wrong_count << std::endl;
  pcout << "  " << std::left << std::setw(46) << "vmult_scratch: misaligned ranges"
        << ": " << n_misaligned << std::endl;

  result -= reference;
  print_check(
    pcout, "vmult_scratch vs. vmult", result.linfty_norm(), reference.linfty_norm(), tolerance);
}

// Compare ExaDG::PreconditionChebyshevScratch against
// dealii::PreconditionChebyshev for the functions vmult(), step() and
// vmult_with_last_residual_norm(). The right hand side and initial guess
// are zero in the constrained entries: In the matrix-vector product with
// 'operation_after_loop' of dealii::MatrixFree::cell_loop(), the source
// values are copied into the constrained entries only before the last cell
// range, so ranges of constrained entries finalized earlier see a zero
// result, whereas vmult_scratch() consistently uses the source values.
// Both variants agree if all constrained entries are zero. For float, the
// ExaDG variant uses the diagonal in BF16 format and deal.II the same
// rounded values in float.
template<typename Operator, typename VectorType>
void
check_chebyshev(dealii::ConditionalOStream &      pcout,
                const Operator &                  op,
                const VectorType &                inverse_diagonal,
                const double                      tolerance,
                const std::vector<unsigned int> & constrained_dofs = {})
{
  using PreconditionerType = dealii::DiagonalMatrix<VectorType>;
  using Number             = typename VectorType::value_type;

  typename ExaDG::PreconditionChebyshevScratch<Operator>::AdditionalData data;

  typename dealii::PreconditionChebyshev<Operator, VectorType, PreconditionerType>::AdditionalData
    data_dealii;
  data_dealii.preconditioner               = std::make_shared<PreconditionerType>();
  data_dealii.preconditioner->get_vector() = inverse_diagonal;
  if constexpr(std::is_same_v<Number, float>)
    ExaDG::round_to_bf16(data_dealii.preconditioner->get_vector(), data.preconditioner_bf16);
  data_dealii.degree              = 4;
  data_dealii.smoothing_range     = 20.;
  data_dealii.max_eigenvalue      = 2.5;
  data_dealii.eig_cg_n_iterations = 0;
  dealii::PreconditionChebyshev<Operator, VectorType, PreconditionerType> chebyshev_dealii;
  chebyshev_dealii.initialize(op, data_dealii);

  if constexpr(!std::is_same_v<Number, float>)
    data.preconditioner = data_dealii.preconditioner;
  data.degree          = data_dealii.degree;
  data.smoothing_range = data_dealii.smoothing_range;
  data.max_eigenvalue  = data_dealii.max_eigenvalue;
  ExaDG::PreconditionChebyshevScratch<Operator> chebyshev;
  chebyshev.initialize(op, data);

  VectorType rhs, initial, reference, result;
  op.initialize_dof_vector(rhs);
  op.initialize_dof_vector(initial);
  fill_vector(rhs, 1);
  fill_vector(initial, 2);
  for(const unsigned int i : constrained_dofs)
  {
    rhs.local_element(i)     = 0;
    initial.local_element(i) = 0;
  }

  for(const unsigned int degree : {1, 2, 4})
  {
    chebyshev_dealii.set_degree(degree);
    chebyshev.set_degree(degree);

    const std::string name = "Chebyshev degree " + std::to_string(degree) + " ";

    op.initialize_dof_vector(reference);
    op.initialize_dof_vector(result);
    chebyshev_dealii.vmult(reference, rhs);
    chebyshev.vmult(result, rhs);
    result -= reference;
    print_check(pcout, name + "vmult", result.linfty_norm(), reference.linfty_norm(), tolerance);

    reference = initial;
    result    = initial;
    chebyshev_dealii.step(reference, rhs);
    chebyshev.step(result, rhs);
    result -= reference;
    print_check(pcout, name + "step", result.linfty_norm(), reference.linfty_norm(), tolerance);

    op.initialize_dof_vector(reference);
    op.initialize_dof_vector(result);
    const double norm_reference = chebyshev_dealii.vmult_with_last_residual_norm(reference, rhs);
    const double norm           = chebyshev.vmult_with_last_residual_norm(result, rhs);
    result -= reference;
    print_check(pcout,
                name + "vmult_with_residual",
                result.linfty_norm(),
                reference.linfty_norm(),
                tolerance);
    print_check(pcout,
                name + "last residual norm",
                std::abs(norm - norm_reference),
                norm_reference,
                tolerance);
  }
}

// Compare the fourth-kind variants of ExaDG::PreconditionChebyshevScratch:
// the plain fourth kind against the fourth_kind variant of
// dealii::PreconditionChebyshev, and the variant with optimized weights
// against a direct implementation of the algorithm with the vectors x, r,
// d as in Lottes (2023) and PETSc's KSP_CHEBYSHEV_OPT_FOURTH. The vectors
// are set up as in check_chebyshev().
template<typename Operator, typename VectorType>
void
check_chebyshev_fourth_kind(dealii::ConditionalOStream &      pcout,
                            const Operator &                  op,
                            const VectorType &                inverse_diagonal,
                            const double                      tolerance,
                            const std::vector<unsigned int> & constrained_dofs = {})
{
  using PreconditionerType = dealii::DiagonalMatrix<VectorType>;
  using Number             = typename VectorType::value_type;
  using Scratch            = ExaDG::PreconditionChebyshevScratch<Operator>;

  typename Scratch::AdditionalData data;
  typename dealii::PreconditionChebyshev<Operator, VectorType, PreconditionerType>::AdditionalData
    data_dealii;
  data_dealii.preconditioner               = std::make_shared<PreconditionerType>();
  data_dealii.preconditioner->get_vector() = inverse_diagonal;
  if constexpr(std::is_same_v<Number, float>)
    ExaDG::round_to_bf16(data_dealii.preconditioner->get_vector(), data.preconditioner_bf16);
  data_dealii.degree              = 5;
  data_dealii.max_eigenvalue      = 2.5;
  data_dealii.smoothing_range     = 20.;
  data_dealii.eig_cg_n_iterations = 0;
  data_dealii.polynomial_type =
    dealii::PreconditionChebyshev<Operator, VectorType, PreconditionerType>::AdditionalData::
      PolynomialType::fourth_kind;
  dealii::PreconditionChebyshev<Operator, VectorType, PreconditionerType> chebyshev_dealii;
  chebyshev_dealii.initialize(op, data_dealii);

  if constexpr(!std::is_same_v<Number, float>)
    data.preconditioner = data_dealii.preconditioner;
  data.degree          = data_dealii.degree;
  data.smoothing_range = data_dealii.smoothing_range;
  data.max_eigenvalue  = data_dealii.max_eigenvalue;

  VectorType rhs, initial, reference, result;
  op.initialize_dof_vector(rhs);
  op.initialize_dof_vector(initial);
  fill_vector(rhs, 1);
  fill_vector(initial, 2);
  for(const unsigned int i : constrained_dofs)
  {
    rhs.local_element(i)     = 0;
    initial.local_element(i) = 0;
  }

  // the optimized fourth-kind iteration with x, r, d
  const VectorType & diagonal = data_dealii.preconditioner->get_vector();
  const auto         reference_optimized =
    [&](VectorType & x, const bool zero_initial_guess, const unsigned int degree) {
      const std::vector<double> betas = Scratch::get_optimized_fourth_kind_weights(degree);
      const double              rho   = data.max_eigenvalue;
      VectorType                r, d, tmp;
      op.initialize_dof_vector(r);
      op.initialize_dof_vector(d);
      op.initialize_dof_vector(tmp);
      if(zero_initial_guess)
      {
        x = 0;
        r = rhs;
      }
      else
      {
        op.vmult(r, x);
        r.sadd(-1., 1., rhs);
      }
      d.equ(4. / (3. * rho), r);
      d.scale(diagonal);
      for(unsigned int i = 1; i < degree; ++i)
      {
        x.add(betas[i - 1], d);
        op.vmult(tmp, d);
        r -= tmp;
        tmp = r;
        tmp.scale(diagonal);
        d.sadd((2. * i - 1.) / (2. * i + 3.), (8. * i + 4.) / ((2. * i + 3.) * rho), tmp);
      }
      x.add(betas[degree - 1], d);
    };

  for(const auto type :
      {Scratch::PolynomialType::fourth_kind, Scratch::PolynomialType::fourth_kind_optimized})
  {
    data.polynomial_type = type;
    Scratch chebyshev;
    chebyshev.initialize(op, data);
    const std::string kind =
      type == Scratch::PolynomialType::fourth_kind ? "4th kind" : "opt. 4th kind";

    for(const unsigned int degree : {1, 2, 5})
    {
      chebyshev_dealii.set_degree(degree);
      chebyshev.set_degree(degree);
      const std::string name = "Chebyshev " + kind + " degree " + std::to_string(degree) + " ";

      op.initialize_dof_vector(reference);
      op.initialize_dof_vector(result);
      if(type == Scratch::PolynomialType::fourth_kind)
        chebyshev_dealii.vmult(reference, rhs);
      else
        reference_optimized(reference, true, degree);
      chebyshev.vmult(result, rhs);
      result -= reference;
      print_check(pcout, name + "vmult", result.linfty_norm(), reference.linfty_norm(), tolerance);

      reference = initial;
      result    = initial;
      if(type == Scratch::PolynomialType::fourth_kind)
        chebyshev_dealii.step(reference, rhs);
      else
        reference_optimized(reference, false, degree);
      chebyshev.step(result, rhs);
      result -= reference;
      print_check(pcout, name + "step", result.linfty_norm(), reference.linfty_norm(), tolerance);
    }
  }
}

} // namespace ExtrudedTest
