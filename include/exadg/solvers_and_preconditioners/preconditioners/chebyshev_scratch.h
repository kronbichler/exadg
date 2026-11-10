/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2026 by Martin Kronbichler, Shubham Goswami,
 *  Richard Schussnig
 *
 *  This file is dual-licensed under the Apache-2.0 with LLVM Exception (see
 *  https://spdx.org/licenses/Apache-2.0.html and
 *  https://spdx.org/licenses/LLVM-exception.html) and the GNU General Public
 *  License as published by the Free Software Foundation, either version 3 of
 *  the License, or (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License in the top-level LICENSE file for
 *  more details.
 *  ______________________________________________________________________
 */

#pragma once

#include <deal.II/base/mpi.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <cmath>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

namespace ExaDG
{
namespace internal
{
// Detect operators that let other processes read the source vector of
// vmult_scratch() until the next call, see
// RTOperator::RaviartThomasOperator::vmult_scratch()
template<typename OperatorType, typename = void>
struct has_pending_source_reads : std::false_type
{
};

template<typename OperatorType>
struct has_pending_source_reads<
  OperatorType,
  std::void_t<decltype(std::declval<const OperatorType &>().wait_until_source_vector_free())>>
  : std::true_type
{
};
} // namespace internal

/**
 * Chebyshev iteration of the first kind with a point-Jacobi preconditioner,
 * with the same coefficients as dealii::PreconditionChebyshev for given
 * bounds of the eigenvalues. The difference is in the vector updates: The
 * operator provides a function template
 *
 *   vmult_scratch(src, after_loop(begin, end, values))
 *
 * that only accumulates the result of the matrix-vector product in a small
 * scratch array (see ScratchMatrixVectorProduct) and passes it to
 * 'after_loop' once all contributions to the range of entries [begin, end)
 * have been collected, where it is directly consumed by the update of the
 * solution. The new solution overwrites the solution from two iterations
 * ago, which is read at the same position, so the iteration only needs two
 * vectors and avoids the global temporary vector holding the result of the
 * matrix-vector product.
 */
template<typename OperatorType>
class PreconditionChebyshevScratch
{
public:
  using VectorType = typename OperatorType::VectorType;
  using Number     = typename VectorType::value_type;

  struct AdditionalData
  {
    // inverse of the diagonal
    std::shared_ptr<dealii::DiagonalMatrix<VectorType>> preconditioner;

    // diagonal in BF16 format
    dealii::AlignedVector<std::uint16_t> preconditioner_bf16;

    // upper bound of the eigenvalues of the preconditioned matrix
    double max_eigenvalue = 1.;

    // the lower bound of the eigenvalues is max_eigenvalue/smoothing_range
    double smoothing_range = 1.;

    // number of matrix-vector products and preconditioner applications
    unsigned int degree = 1;
  };

  void
  initialize(const OperatorType & matrix, const AdditionalData & additional_data)
  {
    AssertThrow(additional_data.preconditioner.get() != nullptr,
                dealii::ExcMessage("Chebyshev iteration needs a preconditioner"));
    // solution_old may hold the source vector of the last product, which
    // other processes may still read
    if constexpr(internal::has_pending_source_reads<OperatorType>::value)
      if(this->matrix != nullptr)
        this->matrix->wait_until_source_vector_free();
    this->matrix = &matrix;
    data         = additional_data;

    // lower end of the eigenvalue range as in dealii::PreconditionChebyshev
    const double alpha = data.smoothing_range > 1. ? data.max_eigenvalue / data.smoothing_range :
                                                     0.9 * data.max_eigenvalue;
    set_eigenvalue_range(alpha, data.max_eigenvalue);

    matrix.initialize_dof_vector(solution_old);
  }

  // Change the interval [lower, upper] the Chebyshev polynomial is built for,
  // without touching the vectors, so it can be called before every solve
  void
  set_eigenvalue_range(const double lower, const double upper)
  {
    Assert(lower > 0 && upper > lower,
           dealii::ExcMessage("Invalid eigenvalue range [" + std::to_string(lower) + ", " +
                              std::to_string(upper) + "]"));
    data.max_eigenvalue  = upper;
    data.smoothing_range = upper / lower;
    delta                = (upper - lower) * 0.5;
    theta                = (upper + lower) * 0.5;
  }

  void
  set_degree(const unsigned int degree)
  {
    Assert(degree > 0, dealii::ExcMessage("The degree must be positive"));
    data.degree = degree;
  }

  // Run the Chebyshev iteration with zero initial guess, like
  // dealii::PreconditionChebyshev::vmult()
  void
  vmult(VectorType & solution, const VectorType & rhs) const
  {
    apply(true, solution, rhs, false);
  }

  // Run the Chebyshev iteration with the content of solution as initial
  // guess, like dealii::PreconditionChebyshev::step()
  void
  step(VectorType & solution, const VectorType & rhs) const
  {
    apply(false, solution, rhs, false);
  }

  // Run the Chebyshev iteration with zero initial guess and return the norm
  // of the residual before the last update of the solution, like
  // dealii::PreconditionChebyshev::vmult_with_last_residual_norm()
  double
  vmult_with_last_residual_norm(VectorType & solution, const VectorType & rhs) const
  {
    return apply(true, solution, rhs, true);
  }

  AdditionalData data;

private:
  double
  apply(const bool         zero_initial_guess,
        VectorType &       solution,
        const VectorType & rhs,
        const bool         compute_final_norm) const
  {
    const Number *        rhs_ptr       = rhs.begin();
    const Number *        diagonal      = data.preconditioner->get_vector().begin();
    const std::uint16_t * diagonal_bf16 = data.preconditioner_bf16.data();
    const unsigned int    n_owned       = solution.locally_owned_size();

    // One iteration with the matrix-vector product, where the first update
    // after a zero initial guess does not read the old solution. Returns the
    // residual norm if requested.
    const auto iterate = [&](const bool   first_update,
                             const bool   compute_norm,
                             const Number factor1,
                             const Number factor2) {
      const Number * solution_ptr = solution.begin();
      Number *       new_solution = solution_old.begin();
      double         residual_sqr = 0;
      const auto     run          = [&](auto first_tag, auto norm_tag) {
        matrix->vmult_scratch(
          solution, [&](const unsigned int begin, const unsigned int end, const Number * matvec) {
            const Number sum =
              update<decltype(first_tag)::value, decltype(norm_tag)::value>(begin,
                                                                            end,
                                                                            factor1,
                                                                            factor2,
                                                                            matvec,
                                                                            rhs_ptr,
                                                                            diagonal,
                                                                            diagonal_bf16,
                                                                            solution_ptr,
                                                                            new_solution);
            if constexpr(decltype(norm_tag)::value)
              residual_sqr += sum;
          });
      };
      if(first_update)
      {
        if(compute_norm)
          run(std::true_type(), std::true_type());
        else
          run(std::true_type(), std::false_type());
      }
      else
      {
        if(compute_norm)
          run(std::false_type(), std::true_type());
        else
          run(std::false_type(), std::false_type());
      }

      solution.swap(solution_old);
      if(compute_norm)
        return std::sqrt(
          dealii::Utilities::MPI::sum(residual_sqr, solution.get_mpi_communicator()));
      else
        return 0.;
    };

    // first iteration: x^1 = x^0 + 1/theta * P^{-1} (b - A x^0)
    double residual_norm = 0;
    if(zero_initial_guess)
    {
      const Number factor       = 1. / theta;
      Number *     solution_ptr = solution.begin();
      DEAL_II_OPENMP_SIMD_PRAGMA
      for(unsigned int i = 0; i < n_owned; ++i)
        solution_ptr[i] = factor * diagonal[i] * rhs_ptr[i];
      if(compute_final_norm && data.degree < 2)
        residual_norm = rhs.l2_norm();
    }
    else
      residual_norm = iterate(true, compute_final_norm && data.degree < 2, 0., 1. / theta);

    // if delta is zero, we do not need to iterate because the updates will
    // be zero
    if(data.degree < 2 || std::abs(delta) < 1e-40)
      return residual_norm;

    double       rhok  = delta / theta;
    const double sigma = theta / delta;
    for(unsigned int k = 0; k < data.degree - 1; ++k)
    {
      const double rhokp = 1. / (2. * sigma - rhok);
      const Number factor1(rhokp * rhok);
      const Number factor2(2. * rhokp / delta);
      rhok = rhokp;

      residual_norm = iterate(zero_initial_guess && k == 0,
                              compute_final_norm && k + 2 == data.degree,
                              factor1,
                              factor2);
    }

    return residual_norm;
  }

  // Compute x^{n+1} = x^n + f_1 (x^n - x^{n-1}) + f_2 P^{-1} (b - A x^n) on
  // the range [begin, end), where the vector new_solution holds x^{n-1} on
  // entry (not read in the first update, where f_1 is zero) and x^{n+1} on
  // exit, and matvec points to the entry 'begin' of A x^n. The range starts
  // at a multiple of the SIMD width and only the last range of the vector
  // can have a remainder. Returns the sum of the squares of the residual
  // entries if requested.
  template<bool first_update, bool compute_norm>
  static Number
  update(const unsigned int    begin,
         const unsigned int    end,
         const Number          factor1,
         const Number          factor2,
         const Number *        matvec,
         const Number *        rhs,
         const Number *        diagonal,
         const std::uint16_t * diagonal_bf16,
         const Number *        solution,
         Number *              new_solution)
  {
    constexpr unsigned int          n_lanes        = dealii::VectorizedArray<Number>::size();
    const Number                    factor1_plus_1 = 1. + factor1;
    const unsigned int              end_regular    = begin + (end - begin) / n_lanes * n_lanes;
    dealii::VectorizedArray<Number> sum            = 0;
    for(unsigned int i = begin; i < end_regular; i += n_lanes)
    {
      dealii::VectorizedArray<Number> rhs_i, sol_i, tmp_i, diag_i;
      rhs_i.load(rhs + i);
      if constexpr(std::is_same_v<Number, float> && dealii::VectorizedArray<Number>::size() == 8)
        diag_i.data = reinterpret_cast<__m256>(_mm256_slli_epi32(
          _mm256_cvtepu16_epi32(*reinterpret_cast<const __m128i *>(diagonal_bf16 + i)), 16));
      else if constexpr(std::is_same_v<Number, float> &&
                        dealii::VectorizedArray<Number>::size() == 16)
        diag_i.data = reinterpret_cast<__m512>(_mm512_slli_epi32(
          _mm512_cvtepu16_epi32(*reinterpret_cast<const __m256i *>(diagonal_bf16 + i)), 16));
      else
        diag_i.load(diagonal + i);
      sol_i.load(solution + i);
      tmp_i.load(matvec + i - begin);
      const dealii::VectorizedArray<Number> residual_i = rhs_i - tmp_i;
      if constexpr(compute_norm)
        sum += residual_i * residual_i;
      dealii::VectorizedArray<Number> result_i;
      if constexpr(first_update)
        result_i = factor1_plus_1 * sol_i + factor2 * diag_i * residual_i;
      else
      {
        dealii::VectorizedArray<Number> sol_old_i;
        sol_old_i.load(new_solution + i);
        result_i = factor1_plus_1 * sol_i - factor1 * sol_old_i + factor2 * diag_i * residual_i;
      }
      result_i.store(new_solution + i);
    }
    for(unsigned int i = end_regular; i < end; ++i)
    {
      const Number residual_i = rhs[i] - matvec[i - begin];
      if constexpr(compute_norm)
        sum[i - end_regular] += residual_i * residual_i;
      if constexpr(first_update)
        new_solution[i] = factor1_plus_1 * solution[i] + factor2 * diagonal[i] * residual_i;
      else
        new_solution[i] = factor1_plus_1 * solution[i] - factor1 * new_solution[i] +
                          factor2 * diagonal[i] * residual_i;
    }
    if constexpr(compute_norm)
      return sum.sum();
    else
      return Number();
  }

  const OperatorType * matrix = nullptr;
  double               delta  = 1.;
  double               theta  = 1.;

  // holds x^{n-1} on entry to the matrix-vector product and the new
  // solution x^{n+1} on exit
  mutable VectorType solution_old;
};

} // namespace ExaDG
