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
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

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

// Detect operators that provide a block-Jacobi preconditioner applied cell
// batch by cell batch, see LaplaceOperatorDG::apply_block_inverse_cell()
template<typename OperatorType, typename = void>
struct has_cell_block_inverse : std::false_type
{
};

template<typename OperatorType>
struct has_cell_block_inverse<
  OperatorType,
  std::void_t<decltype(std::declval<const OperatorType &>().apply_block_inverse_cell(
    0u,
    std::declval<dealii::VectorizedArray<typename OperatorType::VectorType::value_type> *>()))>>
  : std::true_type
{
};
// Detect operators with a block-Jacobi preconditioner embedded in the cell
// loop of vmult_scratch(), see
// LaplaceOperatorFE::vmult_scratch_with_block_update()
template<typename OperatorType, typename = void>
struct has_block_update : std::false_type
{
};

template<typename OperatorType>
struct has_block_update<
  OperatorType,
  std::void_t<decltype(std::declval<const OperatorType &>().apply_block_inverse_vectorized(
    std::declval<typename OperatorType::VectorType &>(),
    std::declval<const typename OperatorType::VectorType &>()))>> : std::true_type
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



// Polynomial of the iteration in PreconditionChebyshevScratch:
// - first_kind: Chebyshev polynomial of the first kind on the interval
//   [max_eigenvalue / smoothing_range, max_eigenvalue], as in
//   dealii::PreconditionChebyshev;
// - fourth_kind: Chebyshev polynomial of the fourth kind on the interval
//   [0, max_eigenvalue] (Lottes, Numer. Linear Algebra Appl. 30:e2518,
//   2023), identical to the fourth_kind variant of
//   dealii::PreconditionChebyshev; smoothing_range is not used;
// - fourth_kind_optimized: the fourth-kind polynomial with the weights
//   beta_i optimized for multigrid smoothing by Lottes (2023), with the
//   values from PETSc's KSP_CHEBYSHEV_OPT_FOURTH, for degree <= 12. It
//   needs one additional vector to accumulate the weighted updates.
enum class ChebyshevPolynomialType
{
  first_kind,
  fourth_kind,
  fourth_kind_optimized
};



/**
 * Chebyshev iteration with a point-Jacobi preconditioner and a polynomial
 * of the kind given by ChebyshevPolynomialType, with the same coefficients
 * as dealii::PreconditionChebyshev for given bounds of the eigenvalues for
 * the first and fourth kind. The difference is in the vector updates: The
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

  using PolynomialType = ChebyshevPolynomialType;

  struct AdditionalData
  {
    PolynomialType polynomial_type = PolynomialType::first_kind;

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

    // Use the block-Jacobi preconditioner of the operator applied cell by
    // cell instead of the diagonal: the operator provides
    // vmult_cellwise(src, cell_function(cell, values)), handing over the
    // matrix-vector product of each cell batch, read_cell_dofs(),
    // write_cell_dofs(), apply_block_inverse_cell(), n_cell_batches(),
    // n_active_lanes() and n_dofs_per_cell(), see LaplaceOperatorDG. Then,
    // 'preconditioner' and 'preconditioner_bf16' are not used, the optimized
    // fourth kind is not available, and step_with_first_product() passes a
    // cell function instead of 'after_loop'.
    bool use_cell_block_inverse = false;

    // If set, a preconditioner dst = P^{-1} src applied on whole vectors,
    // used with the plain vector-based Chebyshev iteration with the
    // operator's vmult(): a basic variant for assessing preconditioners,
    // without the scratch layout and without step_with_first_product() or
    // the optimized fourth kind
    std::function<void(VectorType &, const VectorType &)> global_preconditioner;

    // Use the block-Jacobi preconditioner embedded in the cell loop of the
    // operator, with blocks spanning several cells: the operator provides
    // vmult_scratch_with_block_update(src, after_loop, rhs, residual,
    // target, factor), which adds factor * P^{-1} (rhs - A src) into target
    // as soon as the result is complete on a block,
    // get_block_residual_chunks(), and apply_block_inverse_vectorized(dst,
    // src), see LaplaceOperatorFE. The update in 'after_loop' then stores
    // the part of the new iterate without the preconditioned residual, and
    // the residual for the chunks the operator needs after the loop. Not
    // available with the optimized fourth kind or step_with_first_product().
    bool use_embedded_block_inverse = false;
  };

  void
  initialize(const OperatorType & matrix, const AdditionalData & additional_data)
  {
    matrix.initialize_dof_vector(solution_old);
    // without 'preconditioner', the BF16 diagonal is used, which is only
    // available for float
    constexpr bool is_float = std::is_same_v<Number, float>;
    AssertThrow(additional_data.preconditioner.get() != nullptr || is_float ||
                  additional_data.use_cell_block_inverse || additional_data.global_preconditioner ||
                  additional_data.use_embedded_block_inverse,
                dealii::ExcMessage("Chebyshev iteration needs a preconditioner"));
    if(additional_data.use_embedded_block_inverse)
    {
      AssertThrow(internal::has_block_update<OperatorType>::value,
                  dealii::ExcMessage("The operator does not provide an embedded block inverse"));
      AssertThrow(additional_data.polynomial_type != PolynomialType::fourth_kind_optimized,
                  dealii::ExcNotImplemented(
                    "Optimized fourth kind with the embedded block inverse"));
      matrix.initialize_dof_vector(global_residual);
    }
    if(additional_data.global_preconditioner)
    {
      AssertThrow(additional_data.polynomial_type != PolynomialType::fourth_kind_optimized,
                  dealii::ExcNotImplemented(
                    "Optimized fourth kind with the global preconditioner"));
      matrix.initialize_dof_vector(global_residual);
      matrix.initialize_dof_vector(global_update);
    }
    if(additional_data.use_cell_block_inverse)
    {
      AssertThrow(internal::has_cell_block_inverse<OperatorType>::value,
                  dealii::ExcMessage("The operator does not provide a cell-wise block inverse"));
      AssertThrow(additional_data.polynomial_type != PolynomialType::fourth_kind_optimized,
                  dealii::ExcNotImplemented(
                    "Optimized fourth kind with the cell-wise block inverse"));
    }
    // solution_old may hold the source vector of the last product, which
    // other processes may still read
    if constexpr(internal::has_pending_source_reads<OperatorType>::value)
      if(this->matrix != nullptr)
        this->matrix->wait_until_source_vector_free();
    this->matrix = &matrix;
    data         = additional_data;
    if(data.polynomial_type == PolynomialType::fourth_kind_optimized)
    {
      AssertThrow(data.degree <= max_optimized_degree,
                  dealii::ExcMessage("Optimized fourth-kind weights are only available up to "
                                     "degree " +
                                     std::to_string(max_optimized_degree)));
      matrix.initialize_dof_vector(accumulated_update);
    }
    else
      accumulated_update.reinit(0);

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
    AssertThrow(data.polynomial_type != PolynomialType::fourth_kind_optimized ||
                  degree <= max_optimized_degree,
                dealii::ExcMessage("Optimized fourth-kind weights are only available up to "
                                   "degree " +
                                   std::to_string(max_optimized_degree)));
    data.degree = degree;
  }

  static constexpr unsigned int max_optimized_degree = 12;

  // The weights beta_1, ..., beta_degree of the optimized fourth-kind
  // Chebyshev smoother of the given degree (Lottes 2023, Table in PETSc's
  // src/ksp/ksp/impls/cheby/betas.c)
  static std::vector<double>
  get_optimized_fourth_kind_weights(const unsigned int degree)
  {
    static const std::vector<std::vector<double>> weights = {
      {1.12500000000000},
      {1.02387287570313, 1.26408905371085},
      {1.00842544782028, 1.08867839208730, 1.33753125909618},
      {1.00391310427285, 1.04035811188593, 1.14863498546254, 1.38268869241000},
      {1.00212930146164, 1.02173711549260, 1.07872433192603, 1.19810065292663, 1.41322542791682},
      {1.00128517255940,
       1.01304293035233,
       1.04678215124113,
       1.11616489419675,
       1.23829020218444,
       1.43524297106744},
      {1.00083464397912,
       1.00843949430122,
       1.03008707768713,
       1.07408384092003,
       1.15036186707366,
       1.27116474046139,
       1.45186658649364},
      {1.00057246631197,
       1.00577427662415,
       1.02050187922941,
       1.05019803444565,
       1.10115572984941,
       1.18086042806856,
       1.29838585382576,
       1.46486073151099},
      {1.00040960072832,
       1.00412439506106,
       1.01460212148266,
       1.03561113626671,
       1.07139972529194,
       1.12688273710962,
       1.20785219140729,
       1.32121930716746,
       1.47529642820699},
      {1.00030312229652,
       1.00304840660796,
       1.01077022715387,
       1.02619011597640,
       1.05231724933755,
       1.09255743207549,
       1.15083376663972,
       1.23172250870894,
       1.34060802024460,
       1.48386124407011},
      {1.00023058595209,
       1.00231675024028,
       1.00817245396304,
       1.01982986566342,
       1.03950210235324,
       1.06965042700541,
       1.11305754295742,
       1.17290876275564,
       1.25288300576792,
       1.35725579919519,
       1.49101672564139},
      {1.00017947200828,
       1.00180189139619,
       1.00634861907307,
       1.01537864566306,
       1.03056942830760,
       1.05376019693943,
       1.08699862592072,
       1.13259183097913,
       1.19316273358172,
       1.27171293675110,
       1.37169337969799,
       1.49708418575562}};
    AssertThrow(degree >= 1 && degree <= weights.size(),
                dealii::ExcMessage("No optimized fourth-kind weights for degree " +
                                   std::to_string(degree)));
    return weights[degree - 1];
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

  // Variant of step() where the matrix-vector product of the first
  // iteration is computed by 'first_product(after_loop)' instead of
  // matrix.vmult_scratch(solution, after_loop), e.g. to modify solution
  // within the product before its entries are read. The function must call
  // 'after_loop' like vmult_scratch() and leave the final content of the
  // entries of solution passed to 'after_loop', which is the initial guess
  // of the iteration. With 'use_cell_block_inverse', the function is
  // called with the cell function of vmult_cellwise() instead, i.e.,
  // 'first_product(cell_function)', and must hand over the product of each
  // cell batch to it like vmult_cellwise().
  template<typename FirstProduct>
  void
  step_with_first_product(VectorType &         solution,
                          const VectorType &   rhs,
                          const FirstProduct & first_product) const
  {
    apply(false, solution, rhs, false, &first_product);
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
  // How the vector update accumulates the weighted updates of the optimized
  // fourth-kind iteration, see update()
  struct Accumulation
  {
    int    mode           = 0;
    bool   read           = false;
    Number scale_solution = 0;
    Number beta_minus_1   = 0;
  };

  template<typename FirstProduct = std::nullptr_t>
  double
  apply(const bool           zero_initial_guess,
        VectorType &         solution,
        const VectorType &   rhs,
        const bool           compute_final_norm,
        const FirstProduct * first_product = nullptr) const
  {
    constexpr bool has_first_product = !std::is_same_v<FirstProduct, std::nullptr_t>;
    if constexpr(has_first_product)
      AssertThrow(!zero_initial_guess, dealii::ExcNotImplemented());
    if(data.use_embedded_block_inverse)
    {
      if constexpr(has_first_product || !internal::has_block_update<OperatorType>::value)
      {
        AssertThrow(false, dealii::ExcNotImplemented());
        return 0.;
      }
      else
        return apply_embedded_block(zero_initial_guess, solution, rhs, compute_final_norm);
    }
    if(data.global_preconditioner)
    {
      if constexpr(has_first_product)
      {
        AssertThrow(false, dealii::ExcNotImplemented());
        return 0.;
      }
      else
        return apply_global(zero_initial_guess, solution, rhs, compute_final_norm);
    }
    if(data.use_cell_block_inverse)
    {
      if constexpr(!internal::has_cell_block_inverse<OperatorType>::value)
      {
        AssertThrow(false, dealii::ExcNotImplemented());
        return 0.;
      }
      else
        return apply_block(zero_initial_guess, solution, rhs, compute_final_norm, first_product);
    }
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

    // The fourth-kind iteration of Lottes (2023) with the vectors x, r, d,
    //   d_0 = 4/(3 rho) P^{-1} r_0,  y_{i} = y_{i-1} + d_{i-1},
    //   d_i = (2i-1)/(2i+3) d_{i-1} + (8i+4)/((2i+3) rho) P^{-1} (b - A y_i),
    // is written as the same three-term recurrence on y as for the first
    // kind, using d_{i-1} = y_i - y_{i-1}. With optimized weights, the result
    // is x = x_0 + sum_i beta_i d_{i-1} = y_degree + sum_i (beta_i - 1)
    // d_{i-1}, where the sum is accumulated in a separate vector.
    const bool fourth_kind = data.polynomial_type != PolynomialType::first_kind;
    const bool optimized   = data.polynomial_type == PolynomialType::fourth_kind_optimized;
    const std::vector<double> betas =
      optimized ? get_optimized_fourth_kind_weights(data.degree) : std::vector<double>();
    const double rho = data.max_eigenvalue;
    const Number factor_first_step =
      fourth_kind ? 4. / (3. * rho) * ((optimized && data.degree == 1) ? betas[0] : 1.) :
                    1. / theta;
    const auto run_first_step = [&](const unsigned int begin, const unsigned int end) {
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
    const auto iterate = [&](const bool           first_update,
                             const bool           compute_norm,
                             const Number         factor1,
                             const Number         factor2,
                             const bool           with_first_step = false,
                             const Accumulation & accumulation    = Accumulation()) {
      const Number * solution_ptr = solution.begin();
      Number *       new_solution = solution_old.begin();
      Number *       accumulated  = accumulated_update.begin();
      double         residual_sqr = 0;
      const auto     product      = [&](auto first_tag, auto norm_tag, auto bf16_tag) {
        const auto after_loop =
          [&](const unsigned int begin, const unsigned int end, const Number * matvec) {
            const auto run_update = [&](auto mode_tag) {
              return update<decltype(first_tag)::value,
                            decltype(norm_tag)::value,
                            decltype(bf16_tag)::value,
                            decltype(mode_tag)::value>(begin,
                                                       end,
                                                       factor1,
                                                       factor2,
                                                       matvec,
                                                       rhs_ptr,
                                                       diagonal,
                                                       diagonal_bf16,
                                                       solution_ptr,
                                                       new_solution,
                                                       accumulation,
                                                       accumulated);
            };
            Number sum;
            if(accumulation.mode == 0)
              sum = run_update(std::integral_constant<int, 0>());
            else if(accumulation.mode == 1)
              sum = run_update(std::integral_constant<int, 1>());
            else
              sum = run_update(std::integral_constant<int, 2>());
            if constexpr(decltype(norm_tag)::value)
              residual_sqr += sum;
          };
        // the first iteration with nonzero initial guess, computed by the
        // given function
        if constexpr(has_first_product)
        {
          if constexpr(decltype(first_tag)::value)
          {
            (*first_product)(after_loop);
            return;
          }
        }
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
    {
      // with optimized weights, start the accumulation of (beta_1 - 1) d_0
      Accumulation accumulation;
      if(optimized && data.degree >= 2)
      {
        accumulation.mode         = 1;
        accumulation.beta_minus_1 = betas[0] - 1.;
      }
      residual_norm = iterate(
        true, compute_final_norm && data.degree < 2, 0., factor_first_step, false, accumulation);
    }

    // if delta is zero, we do not need to iterate because the updates will
    // be zero
    if(data.degree < 2 || std::abs(delta) < 1e-40)
      return residual_norm;

    double       rhok  = delta / theta;
    const double sigma = theta / delta;
    for(unsigned int k = 0; k < data.degree - 1; ++k)
    {
      Number factor1, factor2;
      if(fourth_kind)
      {
        factor1 = (2. * k + 1.) / (2. * k + 5.);
        factor2 = (8. * k + 12.) / ((2. * k + 5.) * rho);
      }
      else
      {
        const double rhokp = 1. / (2. * sigma - rhok);
        factor1            = rhokp * rhok;
        factor2            = 2. * rhokp / delta;
        rhok               = rhokp;
      }

      // with optimized weights, add (beta_{k+2} - 1) d_{k+1} to the
      // accumulated updates, which start from (beta_1 - 1) d_0 = (beta_1 - 1)
      // y_1 after the first step with zero initial guess, and write the
      // final result in the last iteration
      Accumulation accumulation;
      if(optimized)
      {
        accumulation.mode         = k + 2 == data.degree ? 2 : 1;
        accumulation.beta_minus_1 = betas[k + 1] - 1.;
        if(k == 0 && zero_initial_guess)
          accumulation.scale_solution = betas[0] - 1.;
        else
          accumulation.read = true;
      }

      residual_norm = iterate(zero_initial_guess && k == 0,
                              compute_final_norm && k + 2 == data.degree,
                              factor1,
                              factor2,
                              fuse_first_step && k == 0,
                              accumulation);
    }

    return residual_norm;
  }

  // Variant of apply() with the block-Jacobi preconditioner of the operator
  // applied cell batch by cell batch: In each iteration, the operator hands
  // the matrix-vector product of a cell batch to the update, which computes
  // the residual on the cells, applies the inverse of the cell blocks, and
  // writes x^{n+1} = x^n + f_1 (x^n - x^{n-1}) + f_2 P^{-1} (b - A x^n) over
  // x^{n-1} as in update(). This is possible because the cells of a batch
  // only write into their own entries. If given, the product of the first
  // iteration with nonzero initial guess is computed by
  // 'first_product(cell_function)' instead of
  // matrix.vmult_cellwise(solution, cell_function), see
  // step_with_first_product().
  template<typename FirstProduct = std::nullptr_t>
  double
  apply_block(const bool           zero_initial_guess,
              VectorType &         solution,
              const VectorType &   rhs,
              const bool           compute_final_norm,
              const FirstProduct * first_product = nullptr) const
  {
    constexpr bool has_first_product     = !std::is_same_v<FirstProduct, std::nullptr_t>;
    using VectorizedArrayType            = dealii::VectorizedArray<Number>;
    const unsigned int n_dofs            = matrix->n_dofs_per_cell();
    const bool         fourth_kind       = data.polynomial_type != PolynomialType::first_kind;
    const double       rho               = data.max_eigenvalue;
    const Number       factor_first_step = fourth_kind ? 4. / (3. * rho) : 1. / theta;
    if(block_buffer.size() < 3 * n_dofs)
      block_buffer.resize_fast(3 * n_dofs);
    VectorizedArrayType * residual     = block_buffer.data();
    VectorizedArrayType * sol          = block_buffer.data() + n_dofs;
    VectorizedArrayType * sol_old      = block_buffer.data() + 2 * n_dofs;
    const Number *        rhs_ptr      = rhs.begin();
    const unsigned int    n_batches    = matrix->n_cell_batches();
    double                residual_sqr = 0;

    const auto iterate = [&](const bool   first_update,
                             const bool   compute_norm,
                             const Number factor1,
                             const Number factor2) {
      const Number * solution_ptr = solution.begin();
      Number *       new_solution = solution_old.begin();
      residual_sqr                = 0;
      const auto cell_function    = [&](const unsigned int cell, VectorizedArrayType * matvec) {
        matrix->read_cell_dofs(cell, rhs_ptr, residual);
        for(unsigned int i = 0; i < n_dofs; ++i)
          residual[i] -= matvec[i];
        if(compute_norm)
        {
          VectorizedArrayType sum = VectorizedArrayType();
          for(unsigned int i = 0; i < n_dofs; ++i)
            sum += residual[i] * residual[i];
          for(unsigned int v = 0; v < matrix->n_active_lanes(cell); ++v)
            residual_sqr += sum[v];
        }
        matrix->apply_block_inverse_cell(cell, residual);
        matrix->read_cell_dofs(cell, solution_ptr, sol);
        const Number factor1_plus_1 = 1. + factor1;
        if(first_update)
          for(unsigned int i = 0; i < n_dofs; ++i)
            residual[i] = factor1_plus_1 * sol[i] + factor2 * residual[i];
        else
        {
          matrix->read_cell_dofs(cell, new_solution, sol_old);
          for(unsigned int i = 0; i < n_dofs; ++i)
            residual[i] = factor1_plus_1 * sol[i] - factor1 * sol_old[i] + factor2 * residual[i];
        }
        matrix->write_cell_dofs(cell, residual, new_solution);
      };
      if constexpr(has_first_product)
      {
        if(first_update && !zero_initial_guess)
          (*first_product)(cell_function);
        else
          matrix->vmult_cellwise(solution, cell_function);
      }
      else
        matrix->vmult_cellwise(solution, cell_function);
      solution.swap(solution_old);
      return compute_norm ?
               std::sqrt(
                 dealii::Utilities::MPI::sum(residual_sqr, solution.get_mpi_communicator())) :
               0.;
    };

    // first iteration: x^1 = x^0 + f P^{-1} (b - A x^0)
    double residual_norm = 0;
    if(zero_initial_guess)
    {
      for(unsigned int cell = 0; cell < n_batches; ++cell)
      {
        matrix->read_cell_dofs(cell, rhs_ptr, residual);
        matrix->apply_block_inverse_cell(cell, residual);
        for(unsigned int i = 0; i < n_dofs; ++i)
          residual[i] *= factor_first_step;
        matrix->write_cell_dofs(cell, residual, solution.begin());
      }
      if(compute_final_norm && data.degree < 2)
        residual_norm = rhs.l2_norm();
    }
    else
      residual_norm = iterate(true, compute_final_norm && data.degree < 2, 0., factor_first_step);

    if(data.degree < 2 || std::abs(delta) < 1e-40)
      return residual_norm;

    double       rhok  = delta / theta;
    const double sigma = theta / delta;
    for(unsigned int k = 0; k < data.degree - 1; ++k)
    {
      Number factor1, factor2;
      if(fourth_kind)
      {
        factor1 = (2. * k + 1.) / (2. * k + 5.);
        factor2 = (8. * k + 12.) / ((2. * k + 5.) * rho);
      }
      else
      {
        const double rhokp = 1. / (2. * sigma - rhok);
        factor1            = rhokp * rhok;
        factor2            = 2. * rhokp / delta;
        rhok               = rhokp;
      }
      residual_norm = iterate(zero_initial_guess && k == 0,
                              compute_final_norm && k + 2 == data.degree,
                              factor1,
                              factor2);
    }
    return residual_norm;
  }

  // Chebyshev iteration with the block-Jacobi preconditioner embedded in the
  // cell loop of the operator: 'after_loop' stores w = x^n + f_1 (x^n -
  // x^{n-1}) over x^{n-1} (and r = b - A x^n in global_residual for the
  // chunks flagged by the operator), and the operator adds f_2 P^{-1} r on
  // each block once r is complete there, which gives x^{n+1} = w + f_2
  // P^{-1} r.
  double
  apply_embedded_block(const bool         zero_initial_guess,
                       VectorType &       solution,
                       const VectorType & rhs,
                       const bool         compute_final_norm) const
  {
    const bool     fourth_kind       = data.polynomial_type != PolynomialType::first_kind;
    const double   rho               = data.max_eigenvalue;
    const Number   factor_first_step = fourth_kind ? 4. / (3. * rho) : 1. / theta;
    const Number * rhs_ptr           = rhs.begin();

    const auto iterate = [&](const bool   first_update,
                             const bool   compute_norm,
                             const Number factor1,
                             const Number factor2) {
      const Number *         solution_ptr   = solution.begin();
      Number *               new_solution   = solution_old.begin();
      Number *               residual       = global_residual.begin();
      const unsigned char *  residual_flags = matrix->get_block_residual_chunks().data();
      constexpr unsigned int chunk_size     = ScratchMatrixVectorProduct<Number>::chunk_size;
      double                 residual_sqr   = 0;
      const Number           factor1_plus_1 = 1. + factor1;
      const auto             after_loop =
        [&](const unsigned int begin, const unsigned int end, const Number * matvec) {
          for(unsigned int i = begin; i < end; ++i)
          {
            const Number r = rhs_ptr[i] - matvec[i - begin];
            if(residual_flags[i / chunk_size])
              residual[i] = r;
            if(compute_norm)
              residual_sqr += r * r;
            new_solution[i] = first_update ?
                                factor1_plus_1 * solution_ptr[i] :
                                factor1_plus_1 * solution_ptr[i] - factor1 * new_solution[i];
          }
        };
      matrix->vmult_scratch_with_block_update(
        solution, after_loop, rhs, global_residual, solution_old, factor2);
      solution.swap(solution_old);
      return compute_norm ?
               std::sqrt(
                 dealii::Utilities::MPI::sum(residual_sqr, solution.get_mpi_communicator())) :
               0.;
    };

    double residual_norm = 0;
    if(zero_initial_guess)
    {
      matrix->apply_block_inverse_vectorized(solution, rhs);
      solution *= factor_first_step;
      if(compute_final_norm && data.degree < 2)
        residual_norm = rhs.l2_norm();
    }
    else
      residual_norm = iterate(true, compute_final_norm && data.degree < 2, 0., factor_first_step);
    if(data.degree < 2 || std::abs(delta) < 1e-40)
      return residual_norm;

    double       rhok  = delta / theta;
    const double sigma = theta / delta;
    for(unsigned int k = 0; k < data.degree - 1; ++k)
    {
      Number factor1, factor2;
      if(fourth_kind)
      {
        factor1 = (2. * k + 1.) / (2. * k + 5.);
        factor2 = (8. * k + 12.) / ((2. * k + 5.) * rho);
      }
      else
      {
        const double rhokp = 1. / (2. * sigma - rhok);
        factor1            = rhokp * rhok;
        factor2            = 2. * rhokp / delta;
        rhok               = rhokp;
      }
      residual_norm = iterate(zero_initial_guess && k == 0,
                              compute_final_norm && k + 2 == data.degree,
                              factor1,
                              factor2);
    }
    return residual_norm;
  }

  // Plain Chebyshev iteration on whole vectors with the operator's vmult()
  // and data.global_preconditioner, with the same coefficients as apply()
  double
  apply_global(const bool         zero_initial_guess,
               VectorType &       solution,
               const VectorType & rhs,
               const bool         compute_final_norm) const
  {
    const bool   fourth_kind       = data.polynomial_type != PolynomialType::first_kind;
    const double rho               = data.max_eigenvalue;
    const Number factor_first_step = fourth_kind ? 4. / (3. * rho) : 1. / theta;
    double       residual_norm     = 0;

    // first iteration x^1 = x^0 + f P^{-1} (b - A x^0), keeping x^0 in
    // solution_old
    if(zero_initial_guess)
    {
      data.global_preconditioner(global_update, rhs);
      solution.equ(factor_first_step, global_update);
      solution_old = 0;
      if(compute_final_norm && data.degree < 2)
        residual_norm = rhs.l2_norm();
    }
    else
    {
      matrix->vmult(global_residual, solution);
      global_residual.sadd(-1., 1., rhs);
      if(compute_final_norm && data.degree < 2)
        residual_norm = global_residual.l2_norm();
      data.global_preconditioner(global_update, global_residual);
      solution_old = solution;
      solution.add(factor_first_step, global_update);
    }
    if(data.degree < 2 || std::abs(delta) < 1e-40)
      return residual_norm;

    double       rhok  = delta / theta;
    const double sigma = theta / delta;
    for(unsigned int k = 0; k < data.degree - 1; ++k)
    {
      double factor1, factor2;
      if(fourth_kind)
      {
        factor1 = (2. * k + 1.) / (2. * k + 5.);
        factor2 = (8. * k + 12.) / ((2. * k + 5.) * rho);
      }
      else
      {
        const double rhokp = 1. / (2. * sigma - rhok);
        factor1            = rhokp * rhok;
        factor2            = 2. * rhokp / delta;
        rhok               = rhokp;
      }
      matrix->vmult(global_residual, solution);
      global_residual.sadd(-1., 1., rhs);
      if(compute_final_norm && k + 2 == data.degree)
        residual_norm = global_residual.l2_norm();
      data.global_preconditioner(global_update, global_residual);
      // x^{n+1} = (1 + f_1) x^n - f_1 x^{n-1} + f_2 P^{-1} r^n
      solution_old.sadd(-factor1, 1. + factor1, solution);
      solution_old.add(factor2, global_update);
      solution.swap(solution_old);
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
  //
  // For the optimized fourth-kind weights, 'accumulation_mode' selects the
  // treatment of the weighted updates with d = x^{n+1} - x^n:
  // 0: none;
  // 1: e = e_old + (beta - 1) d stored in 'accumulated';
  // 2: like 1, but x^{n+1} + e is stored in new_solution as the final result,
  // where e_old is read from 'accumulated' if accumulation.read is set, and
  // otherwise computed as accumulation.scale_solution * x^n.
  template<bool first_update, bool compute_norm, bool use_bf16, int accumulation_mode>
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
         Number *              new_solution,
         const Accumulation &  accumulation,
         Number *              accumulated)
  {
    constexpr unsigned int          n_lanes        = dealii::VectorizedArray<Number>::size();
    const Number                    factor1_plus_1 = 1. + factor1;
    const unsigned int              end_regular    = begin + (end - begin) / n_lanes * n_lanes;
    dealii::VectorizedArray<Number> sum            = 0;
    const bool                      read_acc       = accumulation.read;
    const Number                    scale_solution = accumulation.scale_solution;
    const Number                    beta_minus_1   = accumulation.beta_minus_1;
    // store the new iterate, including the accumulated weighted updates if
    // requested
    const auto store = [&](const auto & sol_i, const auto & result_i, const unsigned int i) {
      using VectorizedType = std::remove_cv_t<std::remove_reference_t<decltype(sol_i)>>;
      if constexpr(accumulation_mode == 0)
      {
        if constexpr(std::is_same_v<VectorizedType, Number>)
          new_solution[i] = result_i;
        else
          result_i.store(new_solution + i);
      }
      else
      {
        VectorizedType acc_i;
        if constexpr(std::is_same_v<VectorizedType, Number>)
          acc_i = read_acc ? accumulated[i] : scale_solution * sol_i;
        else
        {
          if(read_acc)
            acc_i.load(accumulated + i);
          else
            acc_i = scale_solution * sol_i;
        }
        acc_i += beta_minus_1 * (result_i - sol_i);
        if constexpr(std::is_same_v<VectorizedType, Number>)
        {
          if constexpr(accumulation_mode == 1)
          {
            new_solution[i] = result_i;
            accumulated[i]  = acc_i;
          }
          else
            new_solution[i] = result_i + acc_i;
        }
        else
        {
          if constexpr(accumulation_mode == 1)
          {
            result_i.store(new_solution + i);
            acc_i.store(accumulated + i);
          }
          else
            (result_i + acc_i).store(new_solution + i);
        }
      }
    };
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
      store(sol_i, result_i, i);
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

      Number result_i;
      if constexpr(first_update)
        result_i = factor1_plus_1 * solution[i] + factor2 * diag_i * residual_i;
      else
        result_i =
          factor1_plus_1 * solution[i] - factor1 * new_solution[i] + factor2 * diag_i * residual_i;
      store(solution[i], result_i, i);
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

  // accumulated weighted updates sum_i (beta_i - 1) d_{i-1} of the
  // optimized fourth-kind iteration, empty otherwise
  mutable VectorType accumulated_update;

  // residual and preconditioned residual in apply_global()
  mutable VectorType global_residual;
  mutable VectorType global_update;

  // cell values of the residual, solution and old solution in apply_block()
  mutable dealii::AlignedVector<dealii::VectorizedArray<Number>> block_buffer;
};

} // namespace ExaDG
