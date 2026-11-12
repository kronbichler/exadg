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

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <bit>
#include <cmath>
#include <cstdint>
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

// Detect operators with the variant vmult_scratch(src, before_loop,
// after_loop) that runs 'before_loop' on the entries of src before their
// first read
template<typename OperatorType, typename = void>
struct has_vmult_scratch_with_before_loop : std::false_type
{
};

template<typename OperatorType>
struct has_vmult_scratch_with_before_loop<
  OperatorType,
  std::void_t<decltype(std::declval<const OperatorType &>().vmult_scratch(
    std::declval<const typename OperatorType::VectorType &>(),
    std::declval<void (*)(const unsigned int, const unsigned int)>(),
    std::declval<void (*)(const unsigned int,
                          const unsigned int,
                          const typename OperatorType::VectorType::value_type *)>()))>>
  : std::true_type
{
};
} // namespace internal

/**
 * Round a float / VectorizedArray<float> number to the BF16 format (to
 * nearest, ties to even) and store the upper 16 bits of the result at
 * store_location. Infinity and NaN are not treated specially.
 */
template<typename Number>
void
store_float_as_bf16(const Number a, std::uint16_t * store_location)
{
  static_assert(std::is_same_v<Number, float> ||
                  std::is_same_v<Number, dealii::VectorizedArray<float>>,
                "Only conversion float -> BF16 is supported.");
  if constexpr(std::is_same_v<Number, float>)
  {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(a);
    // round to nearest, ties to even, on the upper 16 bits
    bits += 0x7fffu + ((bits >> 16) & 1u);
    store_location[0] = static_cast<std::uint16_t>(bits >> 16);
  }
  else
  {
#ifdef __AVX2__
    if constexpr(Number::size() == 8)
    {
      const __m256i bits = _mm256_castps_si256(a.data);
      const __m256i odd  = _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1));
      const __m256i rounded =
        _mm256_add_epi32(bits, _mm256_add_epi32(_mm256_set1_epi32(0x7fff), odd));
      const __m256i upper = _mm256_srli_epi32(rounded, 16);
      // AVX2 has no 32->16 bit truncation; pack the two 128-bit halves, which
      // keeps the order (packus saturates, but all values are in [0, 0xffff])
      const __m128i reduced =
        _mm_packus_epi32(_mm256_castsi256_si128(upper), _mm256_extracti128_si256(upper, 1));
      _mm_storeu_si128(reinterpret_cast<__m128i *>(store_location), reduced);
    }
    else
#endif
#ifdef __AVX512F__
      if constexpr(Number::size() == 16)
    {
      const __m512i bits = _mm512_castps_si512(a.data);
      const __m512i odd  = _mm512_and_si512(_mm512_srli_epi32(bits, 16), _mm512_set1_epi32(1));
      const __m512i rounded =
        _mm512_add_epi32(bits, _mm512_add_epi32(_mm512_set1_epi32(0x7fff), odd));
      // values are below 2^16 after the shift, so truncation is exact
      const __m256i reduced = _mm512_cvtepi32_epi16(_mm512_srli_epi32(rounded, 16));
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(store_location), reduced);
    }
    else
#endif
      for(unsigned int i = 0; i < Number::size(); ++i)
        store_float_as_bf16(a[i], store_location + i);
  }
}

/**
 * Load data item from a std::uint16_t array (representing an FP type) into a
 * float/VectorizedArray<float> field, selected by the template argument. This
 * works by a left-shift of the loaded uint-16 to the left 16 bits of a 32-bit
 * integer, which is then reinterpreted as float.
 */
template<typename Number>
Number
load_bf16_into_float(const std::uint16_t * ptr)
{
  static_assert(std::is_same_v<Number, float> ||
                  std::is_same_v<Number, dealii::VectorizedArray<float>>,
                "Only conversion BF16 -> float is supported.");

  if constexpr(std::is_same_v<Number, float>)
    return std::bit_cast<float>(static_cast<std::uint32_t>(ptr[0]) << 16);
  else
  {
    Number result;
#ifdef __AVX2__
    if constexpr(Number::size() == 8)
    {
      result.data = _mm256_castsi256_ps(_mm256_slli_epi32(
        _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(ptr))), 16));
      return result;
    }
#endif
#ifdef __AVX512F__
    if constexpr(Number::size() == 16)
    {
      result.data = _mm512_castsi512_ps(_mm512_slli_epi32(
        _mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr))), 16));
      return result;
    }
#endif
    for(unsigned int i = 0; i < Number::size(); ++i)
      result[i] = load_bf16_into_float<float>(ptr + i);
    return result;
  }
}

/**
 * Round the locally owned entries of 'diagonal' to BF16 format, store them
 * in 'diagonal_bf16', and write the rounded values back to 'diagonal'. This
 * way, both representations describe the same preconditioner, e.g. for an
 * eigenvalue estimate with the float vector before Chebyshev runs with the
 * BF16 version.
 */
inline void
round_to_bf16(dealii::LinearAlgebra::distributed::Vector<float> & diagonal,
              dealii::AlignedVector<std::uint16_t> &              diagonal_bf16)
{
  using VectorizedArrayType      = dealii::VectorizedArray<float>;
  constexpr unsigned int n_lanes = VectorizedArrayType::size();

  const unsigned int owned_size   = diagonal.locally_owned_size();
  const unsigned int regular_size = owned_size / n_lanes * n_lanes;
  float *            values       = diagonal.begin();
  diagonal_bf16.resize_fast(owned_size);
  std::uint16_t * values_bf16 = diagonal_bf16.data();
  for(unsigned int i = 0; i < regular_size; i += n_lanes)
  {
    VectorizedArrayType value;
    value.load(values + i);
    store_float_as_bf16(value, values_bf16 + i);
    load_bf16_into_float<VectorizedArrayType>(values_bf16 + i).store(values + i);
  }
  for(unsigned int i = regular_size; i < owned_size; ++i)
  {
    store_float_as_bf16(values[i], values_bf16 + i);
    values[i] = load_bf16_into_float<float>(values_bf16 + i);
  }
}



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
    // inverse of the diagonal; if not set, the iteration uses
    // 'preconditioner_bf16' instead
    std::shared_ptr<dealii::DiagonalMatrix<VectorType>> preconditioner;

    // inverse of the diagonal in BF16 format (only for float), see
    // round_to_bf16(), with one entry per locally owned index. The choice
    // between the two formats depends on 'preconditioner' only, not on the
    // size of this field, which is empty on processes without locally owned
    // indices.
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
    matrix.initialize_dof_vector(solution_old);
    // without 'preconditioner', the BF16 diagonal is used, which is only
    // available for float
    constexpr bool is_float = std::is_same_v<Number, float>;
    AssertThrow(additional_data.preconditioner.get() != nullptr || is_float,
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
    const Number * rhs_ptr = rhs.begin();
    const Number * diagonal =
      data.preconditioner ? data.preconditioner->get_vector().begin() : nullptr;
    const std::uint16_t * diagonal_bf16 = data.preconditioner_bf16.data();
    const unsigned int    n_owned       = solution.locally_owned_size();
    const bool            use_bf16      = diagonal == nullptr;
    AssertThrow(!use_bf16 || data.preconditioner_bf16.size() == n_owned,
                dealii::ExcMessage("The BF16 diagonal has " +
                                   std::to_string(data.preconditioner_bf16.size()) +
                                   " entries, but the vector " + std::to_string(n_owned)));

    // With a zero initial guess, the first iteration x^1 = 1/theta P^{-1} b
    // does not need a matrix-vector product. If the operator supports it, we
    // compute it on the fly in the matrix-vector product of the second
    // iteration, which reads x^1, before each entry is read for the first
    // time. This saves a separate pass through the vectors.
    constexpr bool can_fuse_first_step =
      internal::has_vmult_scratch_with_before_loop<OperatorType>::value;
    const bool fuse_first_step =
      can_fuse_first_step && zero_initial_guess && data.degree >= 2 && std::abs(delta) >= 1e-40;
    const Number factor_first_step = 1. / theta;
    const auto   run_first_step    = [&](const unsigned int begin, const unsigned int end) {
      if constexpr(std::is_same_v<Number, float>)
        if(use_bf16)
        {
          first_step<true>(
            begin, end, factor_first_step, rhs_ptr, diagonal, diagonal_bf16, solution.begin());
          return;
        }
      first_step<false>(
        begin, end, factor_first_step, rhs_ptr, diagonal, diagonal_bf16, solution.begin());
    };

    // One iteration with the matrix-vector product, where the first update
    // after a zero initial guess does not read the old solution. Returns the
    // residual norm if requested.
    const auto iterate = [&](const bool   first_update,
                             const bool   compute_norm,
                             const Number factor1,
                             const Number factor2,
                             const bool   with_first_step = false) {
      const Number * solution_ptr = solution.begin();
      Number *       new_solution = solution_old.begin();
      double         residual_sqr = 0;
      const auto     product      = [&](auto first_tag, auto norm_tag, auto bf16_tag) {
        const auto after_loop =
          [&](const unsigned int begin, const unsigned int end, const Number * matvec) {
            const Number sum = update<decltype(first_tag)::value,
                                      decltype(norm_tag)::value,
                                      decltype(bf16_tag)::value>(begin,
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
          };
        if constexpr(can_fuse_first_step && decltype(first_tag)::value)
          if(with_first_step)
          {
            matrix->vmult_scratch(
              solution,
              [&](const unsigned int begin, const unsigned int end) {
                first_step<decltype(bf16_tag)::value>(begin,
                                                      end,
                                                      factor_first_step,
                                                      rhs_ptr,
                                                      diagonal,
                                                      diagonal_bf16,
                                                      solution.begin());
              },
              after_loop);
            return;
          }
        matrix->vmult_scratch(solution, after_loop);
      };
      const auto run = [&](auto first_tag, auto norm_tag) {
        if constexpr(std::is_same_v<Number, float>)
          if(use_bf16)
          {
            product(first_tag, norm_tag, std::true_type());
            return;
          }
        product(first_tag, norm_tag, std::false_type());
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
      if(!fuse_first_step)
        run_first_step(0, n_owned);
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
                              factor2,
                              fuse_first_step && k == 0);
    }

    return residual_norm;
  }

  // Compute x^1 = f P^{-1} b on the range [begin, end), the first iteration
  // with zero initial guess
  template<bool use_bf16>
  static void
  first_step(const unsigned int    begin,
             const unsigned int    end,
             const Number          factor,
             const Number *        rhs,
             const Number *        diagonal,
             const std::uint16_t * diagonal_bf16,
             Number *              solution)
  {
    constexpr unsigned int n_lanes     = dealii::VectorizedArray<Number>::size();
    const unsigned int     end_regular = begin + (end - begin) / n_lanes * n_lanes;
    for(unsigned int i = begin; i < end_regular; i += n_lanes)
    {
      dealii::VectorizedArray<Number> rhs_i, diag_i;
      rhs_i.load(rhs + i);
      if constexpr(use_bf16)
        diag_i = load_bf16_into_float<dealii::VectorizedArray<Number>>(diagonal_bf16 + i);
      else
        diag_i.load(diagonal + i);
      (factor * diag_i * rhs_i).store(solution + i);
    }
    for(unsigned int i = end_regular; i < end; ++i)
    {
      if constexpr(use_bf16)
        solution[i] = factor * load_bf16_into_float<Number>(diagonal_bf16 + i) * rhs[i];
      else
        solution[i] = factor * diagonal[i] * rhs[i];
    }
  }

  // Compute x^{n+1} = x^n + f_1 (x^n - x^{n-1}) + f_2 P^{-1} (b - A x^n) on
  // the range [begin, end), where the vector new_solution holds x^{n-1} on
  // entry (not read in the first update, where f_1 is zero) and x^{n+1} on
  // exit, and matvec points to the entry 'begin' of A x^n. The range starts
  // at a multiple of the SIMD width and only the last range of the vector
  // can have a remainder. Returns the sum of the squares of the residual
  // entries if requested. With use_bf16, the inverse diagonal is read from
  // diagonal_bf16, otherwise from diagonal.
  template<bool first_update, bool compute_norm, bool use_bf16>
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
      if constexpr(use_bf16)
        diag_i = load_bf16_into_float<dealii::VectorizedArray<Number>>(diagonal_bf16 + i);
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
      Number diag_i;
      if constexpr(use_bf16)
        diag_i = load_bf16_into_float<Number>(diagonal_bf16 + i);
      else
        diag_i = diagonal[i];

      if constexpr(first_update)
        new_solution[i] = factor1_plus_1 * solution[i] + factor2 * diag_i * residual_i;
      else
        new_solution[i] =
          factor1_plus_1 * solution[i] - factor1 * new_solution[i] + factor2 * diag_i * residual_i;
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
