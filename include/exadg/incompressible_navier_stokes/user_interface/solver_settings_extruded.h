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

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/parameter_handler.h>

#include <iomanip>
#include <sstream>
#include <string>

namespace ExaDG
{
namespace IncNS
{
/**
 * Settings of the solvers of the time integrators on extruded meshes
 * (BDFConsistentSplittingExtruded, and partly BDFDualSplittingExtruded):
 * algorithmic variants, parameters of the smoothers of the pressure
 * multigrid, and options for the communication, diagnostics and checks.
 * They are read from the subsection "SolverSettingsExtruded" of the input
 * file. The defaults are the recommended settings.
 */
struct SolverSettingsExtruded
{
  // --- momentum step ---

  // compute the right-hand side and the residual of the momentum step in one
  // loop
  bool fused_momentum_rhs = true;

  // compare the fused right-hand side and residual with the separate
  // computation in each time step (for debugging)
  bool check_fused_momentum_rhs = false;

  // interval of time steps between computations of the norm of the
  // right-hand side of the momentum step for the statistics
  unsigned int momentum_rhs_norm_interval = 16;

  // interval of time steps between diagnostics of the initial guess of the
  // momentum step, 0 to disable
  unsigned int initial_guess_diagnostic_interval = 0;

  // exchange the face data of the Raviart-Thomas operator via shared memory
  // within a node
  bool rt_shared_memory = true;

  // --- Chebyshev smoothers of the pressure multigrid ---

  // kind of the Chebyshev polynomials on the DG level and the FE levels:
  // "first", "fourth" or "opt_fourth"
  std::string chebyshev_kind_dg = "first";
  std::string chebyshev_kind_fe = "first";

  // degree on the DG level; 0 selects the default (4 with point Jacobi, 2
  // with block Jacobi up to pressure degree 5 and 3 above)
  unsigned int chebyshev_degree_dg = 0;

  // degree on the FE levels with degree > 1 and point Jacobi
  unsigned int chebyshev_degree_fe = 5;

  // degree on the FE levels of degree 1
  unsigned int chebyshev_degree_linear = 3;

  // degree on the FE levels with block Jacobi
  unsigned int chebyshev_degree_fe_block = 3;

  // smoothing range on the DG level; 0 selects the default (20 with point
  // Jacobi, 6 with block Jacobi)
  double smoothing_range_dg = 0.;

  // smoothing range on the FE levels
  double smoothing_range_fe = 20.;

  // safety factor on the estimate of the largest eigenvalue
  double eigenvalue_factor = 1.05;

  // correction steps of the block-Jacobi inverse on the DG level with the
  // actual cell block
  unsigned int dg_block_iterations = 0;

  // preconditioner of the Chebyshev smoother on the FE levels of degree > 1
  // without hanging nodes: "PointJacobi", "BlockJacobiBasic" (single cells,
  // applied on whole vectors), "PatchBlockJacobiBasic" (patches of 2^dim
  // cells, applied on whole vectors), "BlockJacobi" (single cells,
  // vectorized and embedded in the cell loop)
  std::string fe_smoother_preconditioner = "PointJacobi";

  // with "BlockJacobi" on the FE levels: compute the residual of the blocks
  // of the cell loop from the result of the operator in the scratch array
  // instead of the stored residual
  bool fe_block_from_scratch = false;

  // --- communication of the FE levels ---

  // fraction of the interior cell batches run before the cell batches
  // touching ghost entries, to overlap the ghost exchange and the compress
  // with computation; negative for the original order without overlap
  double fe_overlap_split = 0.5;

  // interval of cell batches between calls to MPI_Iprobe to let messages
  // progress during the interior cell batches, 0 to disable
  unsigned int fe_mpi_progress_interval = 16;

  // number the DoFs of the FE levels in the order of the cell loop instead
  // of dealii::DoFRenumbering::matrix_free_data_locality()
  bool fe_loop_order_numbering = false;

  // exchange of the FE levels within the node through shared memory instead
  // of MPI messages; the vectors of the FE levels are then allocated in
  // shared memory. The node can be split into groups of
  // 'transfer_ranks_per_node' processes, as for the transfers.
  bool fe_shared_memory = false;

  // --- transfers of the pressure multigrid ---

  // compute the residual within the restriction on the FE levels
  bool fused_residual_restriction = true;

  // compare the fused residual and restriction with the separate
  // computation at setup (for debugging)
  bool check_fused_residual_restriction = false;

  // prolongate to the DG level within the first product of the DG smoother
  bool fused_dg_prolongation = true;

  // compare the fused DG prolongation with the separate computation at
  // setup (for debugging)
  bool check_fused_dg_prolongation = false;

  // fused DG prolongation per cell batch and lane instead of batches of
  // arbitrary cells ordered by the consumers
  bool dg_prolongation_lane_wise = false;

  // look-ahead in cell batches when filling the batches of the fused DG
  // prolongation
  unsigned int dg_prolongation_lookahead = 2;

  // repetitions of a benchmark of the DG prolongation kernels at setup, 0 to
  // disable
  unsigned int benchmark_dg_prolongation = 0;

  // exchange the data of the anisotropic transfers via shared memory within
  // a node
  bool transfer_shared_memory = true;

  // barrier before each anisotropic transfer, to separate the waiting for
  // late processes in the timings (for profiling)
  bool transfer_profile_barrier = false;

  // emulate nodes of this many processes for the shared memory of the
  // transfers (for testing on a single machine), 0 for the actual nodes
  unsigned int transfer_ranks_per_node = 0;

  void
  add_parameters(dealii::ParameterHandler & prm,
                 const std::string &        subsection_name = "SolverSettingsExtruded")
  {
    prm.enter_subsection(subsection_name);
    {
      prm.add_parameter("FusedMomentumRhs",
                        fused_momentum_rhs,
                        "Compute right-hand side and residual of the momentum step in one loop");
      prm.add_parameter("CheckFusedMomentumRhs",
                        check_fused_momentum_rhs,
                        "Compare the fused momentum right-hand side with the separate one");
      prm.add_parameter("MomentumRhsNormInterval",
                        momentum_rhs_norm_interval,
                        "Time steps between norms of the momentum right-hand side",
                        dealii::Patterns::Integer(1));
      prm.add_parameter("InitialGuessDiagnosticInterval",
                        initial_guess_diagnostic_interval,
                        "Time steps between diagnostics of the momentum initial guess, 0 = off");
      prm.add_parameter("RTSharedMemory",
                        rt_shared_memory,
                        "Exchange RT face data via shared memory within a node");

      prm.add_parameter("ChebyshevKindDG",
                        chebyshev_kind_dg,
                        "Chebyshev polynomial kind on the DG level",
                        dealii::Patterns::Selection("first|fourth|opt_fourth"));
      prm.add_parameter("ChebyshevKindFE",
                        chebyshev_kind_fe,
                        "Chebyshev polynomial kind on the FE levels",
                        dealii::Patterns::Selection("first|fourth|opt_fourth"));
      prm.add_parameter("ChebyshevDegreeDG",
                        chebyshev_degree_dg,
                        "Chebyshev degree on the DG level, 0 = default for the preconditioner");
      prm.add_parameter("ChebyshevDegreeFE",
                        chebyshev_degree_fe,
                        "Chebyshev degree on FE levels with degree > 1 and point Jacobi",
                        dealii::Patterns::Integer(1));
      prm.add_parameter("ChebyshevDegreeLinear",
                        chebyshev_degree_linear,
                        "Chebyshev degree on FE levels of degree 1",
                        dealii::Patterns::Integer(1));
      prm.add_parameter("ChebyshevDegreeFEBlock",
                        chebyshev_degree_fe_block,
                        "Chebyshev degree on FE levels with block Jacobi",
                        dealii::Patterns::Integer(1));
      prm.add_parameter("SmoothingRangeDG",
                        smoothing_range_dg,
                        "Smoothing range on the DG level, 0 = default for the preconditioner",
                        dealii::Patterns::Double(0.));
      prm.add_parameter("SmoothingRangeFE",
                        smoothing_range_fe,
                        "Smoothing range on the FE levels",
                        dealii::Patterns::Double(1.));
      prm.add_parameter("EigenvalueFactor",
                        eigenvalue_factor,
                        "Safety factor on the largest eigenvalue estimate",
                        dealii::Patterns::Double(0.));
      prm.add_parameter("DGBlockIterations",
                        dg_block_iterations,
                        "Correction steps of the DG block-Jacobi inverse with the actual block");
      prm.add_parameter(
        "FESmootherPreconditioner",
        fe_smoother_preconditioner,
        "Preconditioner of the smoother on FE levels of degree > 1 without hanging nodes",
        dealii::Patterns::Selection(
          "PointJacobi|BlockJacobiBasic|PatchBlockJacobiBasic|BlockJacobi"));
      prm.add_parameter("FEBlockFromScratch",
                        fe_block_from_scratch,
                        "FE block Jacobi: residual of the blocks from the scratch array");

      prm.add_parameter(
        "FEOverlapSplit",
        fe_overlap_split,
        "Fraction of interior cells before the boundary cells, negative = no overlap",
        dealii::Patterns::Double());
      prm.add_parameter("FEMPIProgressInterval",
                        fe_mpi_progress_interval,
                        "Cell batches between MPI_Iprobe calls in the FE products, 0 = off");
      prm.add_parameter("FELoopOrderNumbering",
                        fe_loop_order_numbering,
                        "Number the DoFs of the FE levels in the order of the cell loop");

      prm.add_parameter("FESharedMemory",
                        fe_shared_memory,
                        "Exchange of the FE levels within the node through shared memory");

      prm.add_parameter("FusedResidualRestriction",
                        fused_residual_restriction,
                        "Compute the residual within the restriction on the FE levels");
      prm.add_parameter("CheckFusedResidualRestriction",
                        check_fused_residual_restriction,
                        "Compare the fused residual and restriction with the separate one");
      prm.add_parameter("FusedDGProlongation",
                        fused_dg_prolongation,
                        "Prolongate to the DG level within the first DG smoother product");
      prm.add_parameter("CheckFusedDGProlongation",
                        check_fused_dg_prolongation,
                        "Compare the fused DG prolongation with the separate one");
      prm.add_parameter("DGProlongationLaneWise",
                        dg_prolongation_lane_wise,
                        "Fused DG prolongation per cell batch and lane");
      prm.add_parameter("DGProlongationLookahead",
                        dg_prolongation_lookahead,
                        "Look-ahead in cell batches for the fused DG prolongation batches");
      prm.add_parameter("BenchmarkDGProlongation",
                        benchmark_dg_prolongation,
                        "Repetitions of a benchmark of the DG prolongation kernels, 0 = off");
      prm.add_parameter("TransferSharedMemory",
                        transfer_shared_memory,
                        "Exchange the data of the anisotropic transfers via shared memory");
      prm.add_parameter("TransferProfileBarrier",
                        transfer_profile_barrier,
                        "Barrier before each anisotropic transfer (profiling)");
      prm.add_parameter("TransferRanksPerNode",
                        transfer_ranks_per_node,
                        "Emulated processes per node for the transfers and the shared-memory "
                        "exchange of the FE levels, 0 = actual nodes");
    }
    prm.leave_subsection();
  }

  void
  print(dealii::ConditionalOStream const & pcout) const
  {
    const auto str = [](const auto & value) {
      std::ostringstream stream;
      stream << std::boolalpha << value;
      return stream.str();
    };
    const auto line = [&](const std::string & name, const std::string & value) {
      pcout << "  " << std::left << std::setw(46) << name << value << std::endl;
    };
    pcout << std::endl << "Solver settings for extruded meshes:" << std::endl;
    line("Fused momentum rhs:", str(fused_momentum_rhs));
    line("Check fused momentum rhs:", str(check_fused_momentum_rhs));
    line("Momentum rhs norm interval:", str(momentum_rhs_norm_interval));
    line("Initial guess diagnostic interval:", str(initial_guess_diagnostic_interval));
    line("RT shared memory:", str(rt_shared_memory));
    line("Chebyshev kind DG / FE:", chebyshev_kind_dg + " / " + chebyshev_kind_fe);
    line("Chebyshev degree DG (0 = default):", str(chebyshev_degree_dg));
    line("Chebyshev degree FE / linear / FE block:",
         str(chebyshev_degree_fe) + " / " + str(chebyshev_degree_linear) + " / " +
           str(chebyshev_degree_fe_block));
    line("Smoothing range DG (0 = default) / FE:",
         str(smoothing_range_dg) + " / " + str(smoothing_range_fe));
    line("Eigenvalue factor:", str(eigenvalue_factor));
    line("DG block iterations:", str(dg_block_iterations));
    line("FE smoother preconditioner:", fe_smoother_preconditioner);
    line("FE block from scratch:", str(fe_block_from_scratch));
    line("FE overlap split:", str(fe_overlap_split));
    line("FE MPI progress interval:", str(fe_mpi_progress_interval));
    line("FE loop order numbering:", str(fe_loop_order_numbering));
    line("FE shared memory:", str(fe_shared_memory));
    line("Fused residual restriction / check:",
         str(fused_residual_restriction) + " / " + str(check_fused_residual_restriction));
    line("Fused DG prolongation / check:",
         str(fused_dg_prolongation) + " / " + str(check_fused_dg_prolongation));
    line("DG prolongation lane-wise / look-ahead:",
         str(dg_prolongation_lane_wise) + " / " + str(dg_prolongation_lookahead));
    line("Benchmark DG prolongation:", str(benchmark_dg_prolongation));
    line("Transfer shared memory / barrier / ranks:",
         str(transfer_shared_memory) + " / " + str(transfer_profile_barrier) + " / " +
           str(transfer_ranks_per_node));
  }
};

} // namespace IncNS
} // namespace ExaDG
