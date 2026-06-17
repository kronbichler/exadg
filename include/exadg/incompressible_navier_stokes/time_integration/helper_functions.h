/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2025 by Martin Kronbichler, Shubham Goswami,
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

/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2025 by Martin Kronbichler, Shubham Goswami,
 *  Schussnig
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
template<bool add_into_result, typename Number, typename Number2, typename Number3>
void
extrapolate_vectors_range(
  const std::size_t                                                       start_index,
  const std::size_t                                                       end_index,
  const std::vector<Number3> &                                            factors,
  std::vector<dealii::LinearAlgebra::distributed::Vector<Number>> const & vectors,
  dealii::LinearAlgebra::distributed::Vector<Number2> &                   result)
{
  Assert(result.size() > 0, dealii::ExcMessage("Result vector has not been initialized."));
  for(auto const & vector : vectors)
    Assert(vector.size() > 0,
           dealii::ExcMessage("Vector to extrapolate has not been initialized."));
  AssertIndexRange(start_index, result.locally_owned_size());
  AssertIndexRange(end_index, result.locally_owned_size() + 1);
  if(factors.size() == 1)
  {
    Number const * vec_0  = vectors[0].begin();
    Number2 *      res    = result.begin();
    Number2 const  beta_0 = factors[0];

    DEAL_II_OPENMP_SIMD_PRAGMA
    for(unsigned int i = start_index; i < end_index; ++i)
      res[i] = add_into_result ? (res[i] + beta_0 * vec_0[i]) : (beta_0 * vec_0[i]);
  }
  else if(factors.size() == 2)
  {
    Number const * vec_0  = vectors[0].begin();
    Number const * vec_1  = vectors[1].begin();
    Number2 *      res    = result.begin();
    Number2 const  beta_0 = factors[0];
    Number2 const  beta_1 = factors[1];

    DEAL_II_OPENMP_SIMD_PRAGMA
    for(unsigned int i = start_index; i < end_index; ++i)
      res[i] =
        (add_into_result ? res[i] + beta_0 * vec_0[i] : beta_0 * vec_0[i]) + beta_1 * vec_1[i];
  }
  else if(factors.size() == 3)
  {
    Number const * vec_0  = vectors[0].begin();
    Number const * vec_1  = vectors[1].begin();
    Number const * vec_2  = vectors[2].begin();
    Number2 *      res    = result.begin();
    Number2 const  beta_0 = factors[0];
    Number2 const  beta_1 = factors[1];
    Number2 const  beta_2 = factors[2];

    DEAL_II_OPENMP_SIMD_PRAGMA
    for(unsigned int i = start_index; i < end_index; ++i)
      res[i] = (add_into_result ? res[i] + beta_0 * vec_0[i] : beta_0 * vec_0[i]) +
               beta_1 * vec_1[i] + beta_2 * vec_2[i];
  }
  else if(factors.size() == 4)
  {
    Number const * vec_0  = vectors[0].begin();
    Number const * vec_1  = vectors[1].begin();
    Number const * vec_2  = vectors[2].begin();
    Number const * vec_3  = vectors[3].begin();
    Number2 *      res    = result.begin();
    Number2 const  beta_0 = factors[0];
    Number2 const  beta_1 = factors[1];
    Number2 const  beta_2 = factors[2];
    Number2 const  beta_3 = factors[3];

    DEAL_II_OPENMP_SIMD_PRAGMA
    for(unsigned int i = start_index; i < end_index; ++i)
      res[i] = (add_into_result ? res[i] + beta_0 * vec_0[i] : beta_0 * vec_0[i]) +
               beta_1 * vec_1[i] + beta_2 * vec_2[i] + beta_3 * vec_3[i];
  }
  else if(factors.size() == 5)
  {
    Number const * vec_0  = vectors[0].begin();
    Number const * vec_1  = vectors[1].begin();
    Number const * vec_2  = vectors[2].begin();
    Number const * vec_3  = vectors[3].begin();
    Number const * vec_4  = vectors[4].begin();
    Number2 *      res    = result.begin();
    Number2 const  beta_0 = factors[0];
    Number2 const  beta_1 = factors[1];
    Number2 const  beta_2 = factors[2];
    Number2 const  beta_3 = factors[3];
    Number2 const  beta_4 = factors[4];

    DEAL_II_OPENMP_SIMD_PRAGMA
    for(unsigned int i = start_index; i < end_index; ++i)
      res[i] = (add_into_result ? res[i] + beta_0 * vec_0[i] : beta_0 * vec_0[i]) +
               beta_1 * vec_1[i] + beta_2 * vec_2[i] + beta_3 * vec_3[i] + beta_4 * vec_4[i];
  }
  else
    for(unsigned int i = start_index; i < end_index; ++i)
    {
      Number2 entry = factors[0] * vectors[0].local_element(i);
      for(unsigned int j = 1; j < factors.size(); ++j)
        entry += factors[j] * vectors[j].local_element(i);
      if(add_into_result)
        result.local_element(i) += entry;
      else
        result.local_element(i) = entry;
    }
}

template<typename Number, typename Number2, typename Number3>
void
extrapolate_vectors(std::vector<Number3> const &                                            factors,
                    std::vector<dealii::LinearAlgebra::distributed::Vector<Number>> const & vectors,
                    dealii::LinearAlgebra::distributed::Vector<Number2> &                   result)
{
  extrapolate_vectors_range<false>(0, result.locally_owned_size(), factors, vectors, result);
}



template<typename Number, typename Number2, typename Number3>
void
extrapolate_vectors_and_add(
  std::vector<Number3> const &                                            factors,
  std::vector<dealii::LinearAlgebra::distributed::Vector<Number>> const & vectors,
  dealii::LinearAlgebra::distributed::Vector<Number2> &                   result)
{
  extrapolate_vectors_range<true>(0, result.locally_owned_size(), factors, vectors, result);
}



template<typename OperatorType, typename VectorType>
void
compute_least_squares_fit(OperatorType const &            op,
                          std::vector<VectorType> const & vectors,
                          VectorType const &              rhs,
                          VectorType &                    result)
{
  using Number = typename VectorType::value_type;
  std::vector<VectorType>    tmp(vectors.size());
  dealii::FullMatrix<double> matrix(vectors.size(), vectors.size());
  std::vector<Number>        small_vector(vectors.size());

  // This algorithm performs a Cholesky (LDLT) factorization of the normal
  // equations for the minimization problem
  // min_{alpha_i} | sum(alpha_i A x_i) - b |
  // which eventually gives the linear combination sum (alpha_i x_i)
  // minimizing the residual among the given search vectors
  unsigned int i = 0;
  for(; i < vectors.size(); ++i)
  {
    tmp[i].reinit(vectors[0], true);
    op.vmult(tmp[i], vectors[i]);

    std::array<Number *, 11> vec_ptrs = {};
    for(unsigned int j = 0; j <= i; ++j)
      vec_ptrs[j] = tmp[j].begin();
    Number const * rhs_ptr = rhs.begin();

    unsigned int constexpr n_lanes    = dealii::VectorizedArray<Number>::size();
    unsigned int constexpr n_lanes_4  = 4 * n_lanes;
    unsigned int const regular_size_4 = (vectors[0].locally_owned_size()) / n_lanes_4 * n_lanes_4;
    unsigned int const regular_size   = (vectors[0].locally_owned_size()) / n_lanes * n_lanes;

    // compute inner products in normal equations (all at once)
    std::array<dealii::VectorizedArray<Number>, 12> local_sums = {};

    unsigned int k = 0;
    for(; k < regular_size_4; k += n_lanes_4)
    {
      dealii::VectorizedArray<Number> v_k_0, v_k_1, v_k_2, v_k_3;
      v_k_0.load(vec_ptrs[i] + k);
      v_k_1.load(vec_ptrs[i] + k + n_lanes);
      v_k_2.load(vec_ptrs[i] + k + 2 * n_lanes);
      v_k_3.load(vec_ptrs[i] + k + 3 * n_lanes);
      for(unsigned int j = 0; j < i; ++j)
      {
        dealii::VectorizedArray<Number> v_j_k, tmp0;
        v_j_k.load(vec_ptrs[j] + k);
        tmp0 = v_k_0 * v_j_k;
        v_j_k.load(vec_ptrs[j] + k + n_lanes);
        tmp0 += v_k_1 * v_j_k;
        v_j_k.load(vec_ptrs[j] + k + 2 * n_lanes);
        tmp0 += v_k_2 * v_j_k;
        v_j_k.load(vec_ptrs[j] + k + 3 * n_lanes);
        tmp0 += v_k_3 * v_j_k;
        local_sums[j] += tmp0;
      }
      local_sums[i] += v_k_0 * v_k_0 + v_k_1 * v_k_1 + v_k_2 * v_k_2 + v_k_3 * v_k_3;

      dealii::VectorizedArray<Number> rhs_k, tmp0;
      rhs_k.load(rhs_ptr + k);
      tmp0 = rhs_k * v_k_0;
      rhs_k.load(rhs_ptr + k + n_lanes);
      tmp0 += rhs_k * v_k_1;
      rhs_k.load(rhs_ptr + k + 2 * n_lanes);
      tmp0 += rhs_k * v_k_2;
      rhs_k.load(rhs_ptr + k + 3 * n_lanes);
      tmp0 += rhs_k * v_k_3;
      local_sums[i + 1] += tmp0;
    }
    for(; k < regular_size; k += n_lanes)
    {
      dealii::VectorizedArray<Number> v_k;
      v_k.load(vec_ptrs[i] + k);
      for(unsigned int j = 0; j < i; ++j)
      {
        dealii::VectorizedArray<Number> v_j_k;
        v_j_k.load(vec_ptrs[j] + k);
        local_sums[j] += v_k * v_j_k;
      }
      local_sums[i] += v_k * v_k;
      dealii::VectorizedArray<Number> rhs_k;
      rhs_k.load(rhs_ptr + k);
      local_sums[i + 1] += v_k * rhs_k;
    }
    for(; k < vectors[0].locally_owned_size(); ++k)
    {
      for(unsigned int j = 0; j <= i; ++j)
        local_sums[j][k - regular_size] += vec_ptrs[i][k] * vec_ptrs[j][k];
      local_sums[i + 1][k - regular_size] += vec_ptrs[i][k] * rhs_ptr[k];
    }
    std::array<Number, 12> scalar_sums;
    for(unsigned int j = 0; j < i + 2; ++j)
      scalar_sums[j] = local_sums[j].sum();

    dealii::Utilities::MPI::sum(dealii::ArrayView<Number const>(scalar_sums.data(), i + 2),
                                vectors[0].get_mpi_communicator(),
                                dealii::ArrayView<Number>(scalar_sums.data(), i + 2));

    for(unsigned int j = 0; j <= i; ++j)
      matrix(i, j) = scalar_sums[j];

    // update row in Cholesky factorization associated to matrix of normal
    // equations using the diagonal entry D
    for(unsigned int j = 0; j < i; ++j)
    {
      double const inv_entry = matrix(i, j) / matrix(j, j);
      for(unsigned int k = j + 1; k <= i; ++k)
        matrix(i, k) -= matrix(k, j) * inv_entry;
    }
    if(matrix(i, i) < 1e-12 * matrix(0, 0) or matrix(0, 0) < 1e-30)
      break;

    // update for the right hand side (forward substitution)
    small_vector[i] = scalar_sums[i + 1];
    for(unsigned int j = 0; j < i; ++j)
      small_vector[i] -= matrix(i, j) / matrix(j, j) * small_vector[j];
  }

  // backward substitution of Cholesky factorization
  for(unsigned int s = i; s < small_vector.size(); ++s)
    small_vector[s] = 0.;
  for(int s = i - 1; s >= 0; --s)
  {
    double sum = small_vector[s];
    for(unsigned int j = s + 1; j < i; ++j)
      sum -= small_vector[j] * matrix(j, s);
    small_vector[s] = sum / matrix(s, s);
  }
  // if(dealii::Utilities::MPI::this_mpi_process(vectors[0].get_mpi_communicator()) == 0)
  //{
  //  std::cout << "extrapolate " << std::defaultfloat << std::setprecision(3) << result.size()
  //            << ": ";
  //  for(const double a : small_vector)
  //    std::cout << a << " ";
  //  if(i > 0)
  //    std::cout << "i=" << i << " " << matrix(i - 1, i - 1) / matrix(0, 0) << "   ";
  //}
  extrapolate_vectors(small_vector, vectors, result);
}


template<bool combine_two, typename Number, typename Number2>
void
do_compute_inner_products_range(const unsigned int            begin,
                                const unsigned int            end,
                                const unsigned int            n_vectors,
                                const std::array<double, 6> & factors_change_in_matrix,
                                const std::array<const Number *, (combine_two ? 10 : 5)> & vec_ptrs,
                                const Number2 *                                            rhs_ptr,
                                dealii::ndarray<dealii::VectorizedArray<double>, 5, 6> & local_sums)
{
  unsigned int constexpr n_lanes    = dealii::VectorizedArray<double>::size();
  unsigned int constexpr n_lanes_4  = 4 * n_lanes;
  unsigned int const regular_size_4 = end / n_lanes_4 * n_lanes_4;
  unsigned int const regular_size   = end / n_lanes * n_lanes;

  std::array<double, 5> compensate_factor;
  for(unsigned int i = 0; i < n_vectors; ++i)
    compensate_factor[i] = factors_change_in_matrix[0] - factors_change_in_matrix[i + 1];

  unsigned int k = begin;
  for(; k < regular_size_4; k += n_lanes_4)
  {
    for(unsigned int i = 0; i < n_vectors; ++i)
    {
      dealii::VectorizedArray<double> v_k_0, v_k_1, v_k_2, v_k_3, tmp;
      if(combine_two)
      {
        v_k_0.load(vec_ptrs[2 * i] + k);
        v_k_1.load(vec_ptrs[2 * i] + k + n_lanes);
        v_k_2.load(vec_ptrs[2 * i] + k + 2 * n_lanes);
        v_k_3.load(vec_ptrs[2 * i] + k + 3 * n_lanes);
        tmp.load(vec_ptrs[2 * i + 1] + k);
        v_k_0 += compensate_factor[i] * tmp;
        tmp.load(vec_ptrs[2 * i + 1] + k + n_lanes);
        v_k_1 += compensate_factor[i] * tmp;
        tmp.load(vec_ptrs[2 * i + 1] + k + 2 * n_lanes);
        v_k_2 += compensate_factor[i] * tmp;
        tmp.load(vec_ptrs[2 * i + 1] + k + 3 * n_lanes);
        v_k_3 += compensate_factor[i] * tmp;

        for(unsigned int j = 0; j < i; ++j)
        {
          dealii::VectorizedArray<double> v_j_k, tmp0;
          v_j_k.load(vec_ptrs[2 * j] + k);
          tmp.load(vec_ptrs[2 * j + 1] + k);
          v_j_k += compensate_factor[j] * tmp;
          tmp0 = v_k_0 * v_j_k;

          v_j_k.load(vec_ptrs[2 * j] + k + n_lanes);
          tmp.load(vec_ptrs[2 * j + 1] + k + n_lanes);
          v_j_k += compensate_factor[j] * tmp;
          tmp0 += v_k_1 * v_j_k;

          v_j_k.load(vec_ptrs[2 * j] + k + 2 * n_lanes);
          tmp.load(vec_ptrs[2 * j + 1] + k + 2 * n_lanes);
          v_j_k += compensate_factor[j] * tmp;
          tmp0 += v_k_2 * v_j_k;

          v_j_k.load(vec_ptrs[2 * j] + k + 3 * n_lanes);
          tmp.load(vec_ptrs[2 * j + 1] + k + 3 * n_lanes);
          v_j_k += compensate_factor[j] * tmp;
          tmp0 += v_k_3 * v_j_k;

          local_sums[i][j] += tmp0;
        }
      }
      else
      {
        v_k_0.load(vec_ptrs[i] + k);
        v_k_1.load(vec_ptrs[i] + k + n_lanes);
        v_k_2.load(vec_ptrs[i] + k + 2 * n_lanes);
        v_k_3.load(vec_ptrs[i] + k + 3 * n_lanes);
        for(unsigned int j = 0; j < i; ++j)
        {
          dealii::VectorizedArray<double> v_j_k, tmp0;
          v_j_k.load(vec_ptrs[j] + k);
          tmp0 = v_k_0 * v_j_k;
          v_j_k.load(vec_ptrs[j] + k + n_lanes);
          tmp0 += v_k_1 * v_j_k;
          v_j_k.load(vec_ptrs[j] + k + 2 * n_lanes);
          tmp0 += v_k_2 * v_j_k;
          v_j_k.load(vec_ptrs[j] + k + 3 * n_lanes);
          tmp0 += v_k_3 * v_j_k;
          local_sums[i][j] += tmp0;
        }
      }
      local_sums[i][i] += v_k_0 * v_k_0 + v_k_1 * v_k_1 + v_k_2 * v_k_2 + v_k_3 * v_k_3;

      dealii::VectorizedArray<double> rhs_k, tmp0, tmp1;
      rhs_k.load(rhs_ptr + k);
      if(i == 0)
      {
        tmp0 = rhs_k * v_k_0;
        tmp1 = rhs_k * rhs_k;
        rhs_k.load(rhs_ptr + k + n_lanes);
        tmp0 += rhs_k * v_k_1;
        tmp1 += rhs_k * rhs_k;
        rhs_k.load(rhs_ptr + k + 2 * n_lanes);
        tmp0 += rhs_k * v_k_2;
        tmp1 += rhs_k * rhs_k;
        rhs_k.load(rhs_ptr + k + 3 * n_lanes);
        tmp0 += rhs_k * v_k_3;
        tmp1 += rhs_k * rhs_k;
        local_sums[i][i + 1] += tmp0;
        local_sums[0][5] += tmp1;
      }
      else
      {
        tmp0 = rhs_k * v_k_0;
        rhs_k.load(rhs_ptr + k + n_lanes);
        tmp0 += rhs_k * v_k_1;
        rhs_k.load(rhs_ptr + k + 2 * n_lanes);
        tmp0 += rhs_k * v_k_2;
        rhs_k.load(rhs_ptr + k + 3 * n_lanes);
        tmp0 += rhs_k * v_k_3;
        local_sums[i][i + 1] += tmp0;
      }
    }
  }
  for(; k < regular_size; k += n_lanes)
  {
    dealii::VectorizedArray<double> rhs_k = 0;
    for(unsigned int i = 0; i < n_vectors; ++i)
    {
      dealii::VectorizedArray<double> v_k, tmp;
      if(combine_two)
      {
        v_k.load(vec_ptrs[2 * i] + k);
        tmp.load(vec_ptrs[2 * i + 1] + k);
        v_k += compensate_factor[i] * tmp;
        for(unsigned int j = 0; j < i; ++j)
        {
          dealii::VectorizedArray<double> v_j_k;
          v_j_k.load(vec_ptrs[2 * j] + k);
          tmp.load(vec_ptrs[2 * j + 1] + k);
          v_j_k += compensate_factor[j] * tmp;
          local_sums[i][j] += v_k * v_j_k;
        }
      }
      else
      {
        v_k.load(vec_ptrs[i] + k);
        for(unsigned int j = 0; j < i; ++j)
        {
          dealii::VectorizedArray<double> v_j_k;
          v_j_k.load(vec_ptrs[j] + k);
          local_sums[i][j] += v_k * v_j_k;
        }
      }
      local_sums[i][i] += v_k * v_k;
      rhs_k.load(rhs_ptr + k);
      local_sums[i][i + 1] += v_k * rhs_k;
    }
    local_sums[0][5] += rhs_k * rhs_k;
  }
  for(; k < end; ++k)
  {
    for(unsigned int i = 0; i < n_vectors; ++i)
    {
      if(combine_two)
      {
        const double v_i_k = vec_ptrs[2 * i][k] + compensate_factor[i] * vec_ptrs[2 * i + 1][k];
        for(unsigned int j = 0; j < i; ++j)
        {
          local_sums[i][j][k - regular_size] +=
            v_i_k * (vec_ptrs[2 * j][k] + compensate_factor[j] * vec_ptrs[2 * j + 1][k]);
        }
        local_sums[i][i][k - regular_size] += v_i_k * v_i_k;
        local_sums[i][i + 1][k - regular_size] += v_i_k * rhs_ptr[k];
      }
      else
      {
        for(unsigned int j = 0; j <= i; ++j)
          local_sums[i][j][k - regular_size] += vec_ptrs[i][k] * vec_ptrs[j][k];
        local_sums[i][i + 1][k - regular_size] += vec_ptrs[i][k] * rhs_ptr[k];
      }
    }
    local_sums[0][5][k - regular_size] += rhs_ptr[k] * rhs_ptr[k];
  }
}



// Compute a least squares fit and return the norm of the right-hand side as
// well as the achieved residual
inline std::pair<double, double>
do_compute_least_squares_fit(
  const MPI_Comm                                                 communicator,
  const dealii::ndarray<dealii::VectorizedArray<double>, 5, 6> & local_sums,
  std::vector<double> &                                          small_vector)
{
  const unsigned int n_vectors = small_vector.size();
  AssertThrow(n_vectors <= 5,
              dealii::ExcNotImplemented("Cannot handle " + std::to_string(n_vectors) +
                                        " vectors in least squares fit yet!"));

  dealii::ndarray<double, 5, 5> matrix;
  std::array<double, 21>        scalar_sums;
  unsigned int                  count = 0;
  for(unsigned int i = 0; i < n_vectors; ++i)
    for(unsigned int j = 0; j < i + 2; ++j, ++count)
      scalar_sums[count] = local_sums[i][j].sum();
  scalar_sums[count] = local_sums[0][5].sum();
  AssertThrow(count < 21, dealii::ExcInternalError());

  dealii::Utilities::MPI::sum(dealii::ArrayView<double const>(scalar_sums.data(), count + 1),
                              communicator,
                              dealii::ArrayView<double>(scalar_sums.data(), count + 1));

  for(unsigned int c = 0, i = 0; i < n_vectors; ++i, ++c)
    for(unsigned int j = 0; j <= i; ++j, ++c)
      matrix[i][j] = scalar_sums[c];

  // This algorithm performs a Cholesky (LDLT) factorization of
  // which eventually gives the linear combination sum (alpha_i x_i)
  // minimizing the residual among the given search vectors
  unsigned int i = 0;
  for(unsigned int c = 0; i < n_vectors; ++i, ++c)
  {
    for(unsigned int j = 0; j <= i; ++j, ++c)
      matrix[i][j] = scalar_sums[c];

    // update row in Cholesky factorization associated to matrix of normal
    // equations using the diagonal entry D
    for(unsigned int j = 0; j < i; ++j)
    {
      double const inv_entry = matrix[i][j] / matrix[j][j];
      for(unsigned int k = j + 1; k <= i; ++k)
        matrix[i][k] -= matrix[k][j] * inv_entry;
    }
    if(matrix[i][i] < 1e-12 * matrix[0][0] or matrix[0][0] < 1e-30)
      break;

    // update for the right hand side (forward substitution)
    small_vector[i] = scalar_sums[c];
    for(unsigned int j = 0; j < i; ++j)
      small_vector[i] -= matrix[i][j] / matrix[j][j] * small_vector[j];
  }

  // backward substitution of Cholesky factorization
  for(unsigned int s = i; s < small_vector.size(); ++s)
    small_vector[s] = 0.;
  for(int s = i - 1; s >= 0; --s)
  {
    double sum = small_vector[s];
    for(unsigned int j = s + 1; j < i; ++j)
      sum -= small_vector[j] * matrix[j][s];
    small_vector[s] = sum / matrix[s][s];
  }

  // compute residual norm of resulting minimization problem
  double residual_norm_sqr = scalar_sums[count];
  for(unsigned int i = 0, c = 0; i < n_vectors; ++i, c += 2)
  {
    for(unsigned int j = 0; j < i; ++j, ++c)
      residual_norm_sqr += 2. * scalar_sums[c] * small_vector[i] * small_vector[j];
    residual_norm_sqr += scalar_sums[c] * small_vector[i] * small_vector[i];
    residual_norm_sqr -= 2 * scalar_sums[c + 1] * small_vector[i];
  }
  return std::make_pair(std::sqrt(scalar_sums[count]), std::sqrt(std::abs(residual_norm_sqr)));
}



// Compute a least squares fit and return the norm of the right-hand side as
// well as the achieved residual
template<typename VectorType1, typename VectorType2, bool combine_two = false>
std::pair<double, double>
compute_least_squares_fit(std::vector<VectorType1> const & vectors_matvec,
                          VectorType2 const &              rhs,
                          std::vector<VectorType1> const & vectors,
                          VectorType2 &                    result,
                          std::vector<double> const &      factor_second = {})
{
  AssertDimension((combine_two ? 2 : 1) * vectors.size(), vectors_matvec.size());
  if(combine_two)
    AssertDimension(factor_second.size(), vectors.size() + 1);
  const unsigned int  n_vectors = vectors.size();
  std::vector<double> small_vector(n_vectors);

  // Solve the normal equations for the minimization problem
  // min_{alpha_i} | sum(alpha_i A x_i) - b |
  // for which we compute the matrix (A x_i)^T (A x_j) and rhs (A x_i)^T b
  AssertThrow(vectors.size() <= 5, dealii::ExcNotImplemented());
  std::array<const typename VectorType1::value_type *, (combine_two ? 10 : 5)> vec_ptrs = {};
  for(unsigned int j = 0; j < (combine_two ? 2 * n_vectors : n_vectors); ++j)
    vec_ptrs[j] = vectors_matvec[j].begin();
  std::array<double, 6> factors_for_combination;
  if(combine_two)
    for(unsigned int i = 0; i < factor_second.size() + 1; ++i)
      factors_for_combination[i] = factor_second[i];

  // compute inner products in normal equations (all at once)
  dealii::ndarray<dealii::VectorizedArray<double>, 5, 6> local_sums = {};

  do_compute_inner_products_range<combine_two>(0,
                                               result.locally_owned_size(),
                                               n_vectors,
                                               factors_for_combination,
                                               vec_ptrs,
                                               rhs.begin(),
                                               local_sums);

  const auto residuals =
    do_compute_least_squares_fit(result.get_mpi_communicator(), local_sums, small_vector);

  extrapolate_vectors(small_vector, vectors, result);

  return residuals;
}
