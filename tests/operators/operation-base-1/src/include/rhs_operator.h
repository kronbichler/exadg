#ifndef OPERATOR_BASE_RHS_OPERATOR
#define OPERATOR_BASE_RHS_OPERATOR

template <int dim, int fe_degree, typename value_type> class RHSOperator {
public:
  typedef RHSOperator<dim, fe_degree, value_type> This;

  RHSOperator(MatrixFree<dim, value_type> const &mf_data) : data(&mf_data) {}

  // apply matrix vector multiplication
  void evaluate(parallel::distributed::Vector<value_type> &dst) const {
    dst = 0;
    evaluate_add(dst);
  }

  void evaluate_add(parallel::distributed::Vector<value_type> &dst) const {
    parallel::distributed::Vector<value_type> src;
    data->cell_loop(&This::cell_loop, this, dst, src);
  }

private:
  template <typename FEEvaluation>
  inline void do_cell_integral(FEEvaluation &fe_eval) const {
    for (unsigned int q = 0; q < fe_eval.n_q_points; ++q) {
      VectorizedArray<value_type> rhs = make_vectorized_array<value_type>(0.0);
      rhs = 1.0;
      fe_eval.submit_value(rhs, q);
    }
    fe_eval.integrate(true, false);
  }

  void
  cell_loop(MatrixFree<dim, value_type> const &data,
            parallel::distributed::Vector<value_type> &dst,
            parallel::distributed::Vector<value_type> const &,
            std::pair<unsigned int, unsigned int> const &cell_range) const {
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, value_type> fe_eval(data);

    for (unsigned int cell = cell_range.first; cell < cell_range.second;
         ++cell) {
      fe_eval.reinit(cell);

      do_cell_integral(fe_eval);

      fe_eval.distribute_local_to_global(dst);
    }
  }

  MatrixFree<dim, value_type> const *data;
};

#endif