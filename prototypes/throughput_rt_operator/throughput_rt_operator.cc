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

// Throughput of RTOperator::RaviartThomasOperator on the mesh of the
// periodic hill flow (box with a curved bottom wall, periodic in x and z
// direction, no-slip walls in y direction, extruded in z direction), in
// FP64 and FP32, for different orders of the cells in the loop:
// - vmult_scratch() with an 'after_loop' copying the result (operator
//   alone) and with the update of a Chebyshev iteration,
// - vmult() with 'before_loop' and 'after_loop' as done by the conjugate
//   gradient method, the pattern of the passes with 'pre' functions,
// - evaluate_convective_term().
// Also prints statistics of the data locality of the different orders, see
// RaviartThomasOperator::print_data_locality_statistics().

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q_cache.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/laplace_operator_extruded.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/momentum_operator_rt.h>

#include "cell_orderings.h"

#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace
{
using namespace dealii;

constexpr unsigned int dim = 3;

struct Parameters
{
  unsigned int              degree        = 5;
  std::vector<unsigned int> subdivisions  = {4, 3, 2};
  unsigned int              n_refinements = 3;
  std::vector<std::string>  orderings     = {"current"};
  std::vector<std::string>  numbers       = {"double", "float"};
  std::vector<std::string>  benchmarks    = {"scratch", "chebyshev", "cg", "convective"};
  // 'mf': DoF numbering of matrix_free_data_locality() with the default
  // order of MatrixFree as in the spatial operator of ExaDG, 'loop': with
  // the order of the cells of the RT operator
  std::string renumbering = "mf";
  // benchmarks of the pressure: the DG Laplace operator alone ('p_scratch'),
  // in the Chebyshev smoother ('p_chebyshev') and the conjugate gradient
  // method ('p_cg'), and the coupling terms of the RT operator with the
  // pressure ('conv_div': convective term with the divergence terms of the
  // pressure Poisson equation, 'momentum_rhs': right-hand side of the
  // momentum equation with the pressure gradient)
  std::vector<std::string> pressure_benchmarks = {};
  // 'default': numbering of distribute_dofs(), 'loop': in the order of the
  // cell loop with Helper::renumber_dg_dofs_in_cell_loop_order()
  std::string  pressure_renumbering = "default";
  double       min_time             = 0.5;
  unsigned int n_samples            = 5;
  bool         statistics           = true;
  bool         shared_memory        = true;
  bool         timings              = false;
};

template<typename T>
std::vector<T>
split_list(const std::string & text)
{
  std::vector<T>     result;
  std::istringstream stream(text);
  std::string        item;
  while(std::getline(stream, item, ','))
  {
    std::istringstream item_stream(item);
    T                  value;
    item_stream >> value;
    result.push_back(value);
  }
  return result;
}

void
print_usage()
{
  std::cout
    << "Usage: throughput_rt_operator [options]\n"
    << "  --degree k              Raviart-Thomas degree (normal degree k), default 5\n"
    << "  --subdivisions nx,ny,nz coarse mesh, default 4,3,2 (periodic hill)\n"
    << "  --refinements r         global refinements, default 3\n"
    << "  --orderings list        orders of the cells, comma-separated from\n"
    << "                          " << CellOrderings::list_names() << "\n"
    << "                          default current\n"
    << "  --numbers list          double,float (default both)\n"
    << "  --benchmarks list       scratch,chebyshev,cg,convective (default), and the\n"
    << "                          references cg_vectors,cg_unfused, and cheb_rev,\n"
    << "                          cheb_alt with the reverse order of cells\n"
    << "  --renumbering name      DoF numbering: mf (default, as the spatial operator),\n"
    << "                          loop (mf with the order of the RT operator), first or\n"
    << "                          last (first read or last write in the loop of the RT\n"
    << "                          operator), surflast (last, cells next to remote cells\n"
    << "                          first), exadg (Helper::renumber_dofs_in_cell_loop_order,\n"
    << "                          as surflast)\n"
    << "  --min-time s            duration of each measurement, default 0.5\n"
    << "  --samples n             number of measurements, default 5\n"
    << "  --no-statistics         skip the data locality statistics\n"
    << "  --no-shared-memory      exchange via MPI messages only\n"
    << "  --timings               print the timings collected by the operator\n"
    << "  --pressure-benchmarks l p_scratch,p_chebyshev,p_cg (DG Laplace operator of\n"
    << "                          the pressure), conv_div,momentum_rhs (coupling terms\n"
    << "                          of the RT operator with the pressure); default none\n"
    << "  --pressure-renumbering  default (distribute_dofs) or loop (cell loop order)\n";
}

Parameters
parse_parameters(int argc, char ** argv)
{
  Parameters prm;
  for(int i = 1; i < argc; ++i)
  {
    const std::string arg  = argv[i];
    const auto        next = [&]() -> std::string {
      AssertThrow(i + 1 < argc, ExcMessage("Missing value for " + arg));
      return argv[++i];
    };
    if(arg == "--degree")
      prm.degree = std::stoi(next());
    else if(arg == "--subdivisions")
      prm.subdivisions = split_list<unsigned int>(next());
    else if(arg == "--refinements")
      prm.n_refinements = std::stoi(next());
    else if(arg == "--orderings")
      prm.orderings = split_list<std::string>(next());
    else if(arg == "--numbers")
      prm.numbers = split_list<std::string>(next());
    else if(arg == "--benchmarks")
      prm.benchmarks = split_list<std::string>(next());
    else if(arg == "--renumbering")
      prm.renumbering = next();
    else if(arg == "--min-time")
      prm.min_time = std::stod(next());
    else if(arg == "--samples")
      prm.n_samples = std::stoi(next());
    else if(arg == "--no-statistics")
      prm.statistics = false;
    else if(arg == "--no-shared-memory")
      prm.shared_memory = false;
    else if(arg == "--pressure-benchmarks")
      prm.pressure_benchmarks = split_list<std::string>(next());
    else if(arg == "--pressure-renumbering")
      prm.pressure_renumbering = next();
    else if(arg == "--timings")
      prm.timings = true;
    else
    {
      if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        print_usage();
      AssertThrow(arg == "--help", ExcMessage("Unknown option " + arg));
      std::exit(0);
    }
  }
  AssertThrow(prm.subdivisions.size() == dim, ExcMessage("Need 3 subdivisions"));
  return prm;
}



// Geometry of the periodic hill: hill height H, the domain spans
// [0, 9H] x [H, 3.036H] x [-W/2, W/2] before the deformation, with the
// width W as in the application of ExaDG
constexpr double H      = 0.028;
constexpr double LENGTH = 9.0 * H;
constexpr double HEIGHT = 2.036 * H;

// Shape of the hill, see applications/incompressible_navier_stokes/
// periodic_hill/include/manifold.h
double
hill_shape(double x_m)
{
  if(x_m > LENGTH / 2.0)
    x_m = LENGTH - x_m;
  const double x = 1000. * x_m;
  const double h = 1000. * H;
  double       y = 0;
  if(x <= 9.0)
    y = -h + std::min(h, h + 6.775070969851e-3 * x * x - 2.124527775800e-3 * x * x * x);
  else if(x <= 14.0)
    y = -h + 2.507355893131e1 + 9.754803562315e-1 * x - 1.016116352781e-1 * x * x +
        1.889794677828e-3 * x * x * x;
  else if(x <= 20.0)
    y = -h + 2.579601052357e1 + 8.206693007457e-1 * x - 9.055370274339e-2 * x * x +
        1.626510569859e-3 * x * x * x;
  else if(x <= 30.0)
    y = -h + 4.046435022819e1 - 1.379581654948 * x + 1.945884504128e-2 * x * x -
        2.070318932190e-4 * x * x * x;
  else if(x <= 40.0)
    y = -h + 1.792461334664e1 + 8.743920332081e-1 * x - 5.567361123058e-2 * x * x +
        6.277731764683e-4 * x * x * x;
  else if(x <= 54.0)
    y = -h + std::max(0.0,
                      5.639011190988e1 - 2.010520359035 * x + 1.644919857549e-2 * x * x +
                        2.674976141766e-5 * x * x * x);
  else
    y = -h;
  return 0.001 * y;
}

// Map a point of the box to the hill geometry with a tanh grading of the
// cells towards the walls
Point<dim>
push_forward(const Point<dim> & p)
{
  const double gamma  = 2.0;
  const double y      = 2.0 * (p[1] - H) / HEIGHT - 1.0;
  const double t      = 0.5 * std::tanh(gamma * y) / std::tanh(gamma) + 0.5;
  Point<dim>   bottom = p, top = p;
  bottom[1] = H + hill_shape(p[0]);
  top[1]    = H + HEIGHT;
  return Point<dim>(top * t + bottom * (1. - t));
}



struct Setup
{
  Setup(const Parameters & prm)
    : tria(MPI_COMM_WORLD),
      fe(prm.degree - 1),
      dof_handler(tria),
      fe_p(FE_DGQ<dim>(prm.degree - 1), 1),
      dof_handler_p(tria)
  {
    const double width =
      LENGTH * (0.5 + 0.5 * (double(prm.subdivisions[2]) / double(prm.subdivisions[0]) - 0.5));
    GridGenerator::subdivided_hyper_rectangle(tria,
                                              prm.subdivisions,
                                              Point<dim>(0, H, -width / 2),
                                              Point<dim>(LENGTH, H + HEIGHT, width / 2));
    for(const auto & cell : tria.cell_iterators())
      for(const unsigned int f : {0U, 1U, 4U, 5U})
        if(cell->at_boundary(f))
          cell->face(f)->set_all_boundary_ids(10 + (f < 2 ? f : f - 2));
    GridTools::collect_periodic_faces(tria, 10, 11, 0, periodic_faces);
    GridTools::collect_periodic_faces(tria, 12, 13, 2, periodic_faces);
    tria.add_periodicity(periodic_faces);
    tria.refine_global(prm.n_refinements);

    auto mapping_q_cache = std::make_shared<MappingQCache<dim>>(prm.degree);
    mapping_q_cache->initialize(
      MappingQ<dim>(1),
      tria,
      [](const typename Triangulation<dim>::cell_iterator &, const Point<dim> & p) {
        return push_forward(p);
      },
      false);
    mapping = mapping_q_cache;

    dof_handler.distribute_dofs(fe);
  }

  // Constraints and DoF numbering as in the spatial operator of ExaDG for
  // the HDIV discretization; for 'loop', the matrix_free_data_locality()
  // numbering is computed with the cell order given by 'categories'
  void
  setup_dofs(const std::string & renumbering, const std::vector<unsigned int> & categories)
  {
    const auto make_constraints = [&]() {
      constraints.clear();
      constraints.reinit(dof_handler.locally_owned_dofs(),
                         DoFTools::extract_locally_relevant_dofs(dof_handler));
      DoFTools::make_periodicity_constraints(dof_handler, 10, 11, 0, constraints);
      DoFTools::make_periodicity_constraints(dof_handler, 12, 13, 2, constraints);
      VectorTools::project_boundary_values_div_conforming(
        dof_handler, 0, Functions::ZeroFunction<dim>(dim), 0, constraints, *mapping);
      constraints.close();
    };
    dof_handler.distribute_dofs(fe);
    make_constraints();
    typename MatrixFree<dim, double>::AdditionalData mf_data;
    mf_data.overlap_communication_computation = false;
    if(renumbering == "loop")
    {
      mf_data.cell_vectorization_category          = categories;
      mf_data.cell_vectorization_categories_strict = false;
    }
    else
      AssertThrow(renumbering == "mf" || renumbering == "first" || renumbering == "last" ||
                    renumbering == "surflast" || renumbering == "exadg",
                  ExcMessage("Unknown renumbering " + renumbering));
    DoFRenumbering::matrix_free_data_locality(dof_handler, constraints, mf_data);
    make_constraints();
    if(renumbering == "first" || renumbering == "last" || renumbering == "surflast")
    {
      renumber_in_loop_order(renumbering != "first", renumbering == "surflast", categories);
      make_constraints();
    }
    else if(renumbering == "exadg")
    {
      AssertThrow(!categories.empty(), ExcMessage("Need categories for 'exadg'"));
      Helper::renumber_dofs_in_cell_loop_order(dof_handler, constraints, categories);
      make_constraints();
    }
  }

  // Number the locally owned DoFs by the entities (faces, cell interiors)
  // of the RT element in the order of the cell loop of the RT operator,
  // which is the one of MatrixFree with the given categories: by the first
  // read (the entities of the cell and of its neighbors across the right
  // faces, which the face integrals read) or by the last write. The DoFs
  // on faces to cells of other processes come after the others, followed
  // by the constrained DoFs, as in DoFRenumbering::matrix_free_data_locality().
  // With 'surface_first', the entities of the cells with a face to a cell
  // of another process, which pass through 'before_loop' ahead of the loop,
  // come first.
  void
  renumber_in_loop_order(const bool                        by_last_write,
                         const bool                        surface_first,
                         const std::vector<unsigned int> & categories)
  {
    typename MatrixFree<dim, double>::AdditionalData mf_data;
    mf_data.cell_vectorization_category          = categories;
    mf_data.cell_vectorization_categories_strict = false;
    mf_data.overlap_communication_computation    = false;
    mf_data.initialize_mapping                   = false;
    MatrixFree<dim, double> matrix_free;
    matrix_free.reinit(MappingQ1<dim>(), dof_handler, constraints, QGauss<1>(2), mf_data);

    const IndexSet &   owned         = dof_handler.locally_owned_dofs();
    const unsigned int dofs_per_face = fe.n_dofs_per_face();
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int invalid       = numbers::invalid_unsigned_int;

    // first DoF of each entity after resolving the periodicity constraints,
    // and the number of DoFs of the entity, of all 2*dim+1 entities
    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);
    const auto get_entities = [&](const typename DoFHandler<dim>::cell_iterator & cell) {
      cell->get_dof_indices(dof_indices);
      std::array<std::pair<types::global_dof_index, unsigned int>, 2 * dim + 1> entities;
      for(unsigned int e = 0; e < 2 * dim + 1; ++e)
      {
        types::global_dof_index first = dof_indices[e * dofs_per_face];
        const auto              line  = constraints.get_constraint_entries(first);
        if(line != nullptr && line->size() == 1)
          first = (*line)[0].first;
        entities[e] = {first,
                       e < 2 * dim ? dofs_per_face : dofs_per_cell - 2 * dim * dofs_per_face};
      }
      return entities;
    };
    const auto get_neighbor = [](const typename DoFHandler<dim>::cell_iterator & cell,
                                 const unsigned int                              f) {
      if(cell->at_boundary(f) && cell->has_periodic_neighbor(f))
        return cell->periodic_neighbor(f);
      else if(!cell->at_boundary(f))
        return cell->neighbor(f);
      else
        return typename DoFHandler<dim>::cell_iterator();
    };

    // DoFs on faces to remote cells are numbered separately
    std::vector<unsigned char> is_remote(owned.n_elements(), 0);
    for(const auto & cell : dof_handler.active_cell_iterators())
      if(cell->is_locally_owned())
      {
        const auto entities = get_entities(cell);
        for(unsigned int f = 0; f < 2 * dim; ++f)
        {
          const auto neighbor = get_neighbor(cell, f);
          if(neighbor.state() == IteratorState::valid && !neighbor->is_locally_owned() &&
             owned.is_element(entities[f].first))
            for(unsigned int i = 0; i < entities[f].second; ++i)
              is_remote[owned.index_within_set(entities[f].first + i)] = 1;
        }
      }

    // entities of cells with a face to a remote cell
    std::set<types::global_dof_index> surface_entities;
    if(surface_first)
      for(const auto & cell : dof_handler.active_cell_iterators())
        if(cell->is_locally_owned())
        {
          bool is_surface = false;
          for(unsigned int f = 0; f < 2 * dim; ++f)
          {
            const auto neighbor = get_neighbor(cell, f);
            if(neighbor.state() == IteratorState::valid && !neighbor->is_locally_owned())
              is_surface = true;
          }
          if(is_surface)
            for(const auto & entity : get_entities(cell))
              surface_entities.insert(entity.first);
        }

    // number of cells writing into each entity for the last write
    std::map<types::global_dof_index, unsigned int> n_writers;
    if(by_last_write)
      for(const auto & cell : dof_handler.active_cell_iterators())
        if(cell->is_locally_owned())
          for(const auto & entity : get_entities(cell))
            ++n_writers[entity.first];

    std::vector<unsigned int> new_numbers(owned.n_elements(), invalid);
    std::vector<unsigned int> remote_order, interior_order;
    unsigned int              counter = 0;
    const auto assign = [&](const std::pair<types::global_dof_index, unsigned int> & entity) {
      if(!owned.is_element(entity.first) || constraints.is_constrained(entity.first))
        return;
      const unsigned int local = owned.index_within_set(entity.first);
      if(new_numbers[local] != invalid)
        return;
      for(unsigned int i = 0; i < entity.second; ++i)
      {
        if(is_remote[local + i])
        {
          remote_order.push_back(local + i);
          new_numbers[local + i] = invalid - 1;
        }
        else if(surface_first && surface_entities.count(entity.first) == 0)
        {
          interior_order.push_back(local + i);
          new_numbers[local + i] = invalid - 1;
        }
        else
          new_numbers[local + i] = counter++;
      }
    };

    for(unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
      for(unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(cell); ++v)
      {
        const auto dcell = matrix_free.get_cell_iterator(cell, v);
        for(const auto & entity : get_entities(dcell))
          if(!by_last_write || --n_writers[entity.first] == 0)
            assign(entity);
        if(!by_last_write)
          for(unsigned int f = 1; f < 2 * dim; f += 2)
          {
            const auto neighbor = get_neighbor(dcell, f);
            if(neighbor.state() == IteratorState::valid && neighbor->is_locally_owned())
              for(const auto & entity : get_entities(neighbor))
                assign(entity);
          }
      }
    for(const unsigned int i : interior_order)
      new_numbers[i] = counter++;
    for(const unsigned int i : remote_order)
      new_numbers[i] = counter++;
    for(unsigned int & n : new_numbers)
      if(n == invalid)
        n = counter++;
    AssertThrow(counter == owned.n_elements(), ExcInternalError());

    std::vector<types::global_dof_index> new_global(owned.n_elements());
    for(unsigned int i = 0; i < owned.n_elements(); ++i)
      new_global[i] = owned.nth_index_in_set(new_numbers[i]);
    dof_handler.renumber_dofs(new_global);
  }

  parallel::distributed::Triangulation<dim> tria;
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
                             periodic_faces;
  FE_RaviartThomasNodal<dim> fe;
  FESystem<dim>              fe_p;
  DoFHandler<dim>            dof_handler_p;
  AffineConstraints<double>  constraints_p;

  // DoFs of the pressure as in the spatial operator of ExaDG, optionally
  // renumbered in the order of the cell loop
  void
  setup_pressure_dofs(const std::string & renumbering, const std::vector<unsigned int> & categories)
  {
    dof_handler_p.distribute_dofs(fe_p);
    if(renumbering == "loop")
      Helper::renumber_dg_dofs_in_cell_loop_order(dof_handler_p, categories);
    else
      AssertThrow(renumbering == "default", ExcMessage("Unknown renumbering " + renumbering));
    constraints_p.clear();
    constraints_p.close();
  }

  DoFHandler<dim>               dof_handler;
  std::shared_ptr<Mapping<dim>> mapping;
  AffineConstraints<double>     constraints;
};



// Run 'function' repeatedly and return the best and average time per call
// over the samples, using the maximum over the processes for each sample
template<typename Function>
std::pair<double, double>
measure(const Parameters & prm, const Function & function)
{
  const MPI_Comm comm = MPI_COMM_WORLD;
  for(unsigned int i = 0; i < 3; ++i)
    function();

  // number of calls per sample from a short calibration
  Timer        timer;
  unsigned int n_calls = 0;
  MPI_Barrier(comm);
  timer.restart();
  while(Utilities::MPI::max(timer.wall_time(), comm) < 0.05)
  {
    function();
    ++n_calls;
  }
  const double time_calibration = Utilities::MPI::max(timer.wall_time(), comm) / n_calls;
  n_calls =
    std::max(1U, static_cast<unsigned int>(prm.min_time / prm.n_samples / time_calibration));

  double best = std::numeric_limits<double>::max(), sum = 0;
  for(unsigned int s = 0; s < prm.n_samples; ++s)
  {
    MPI_Barrier(comm);
    timer.restart();
    for(unsigned int i = 0; i < n_calls; ++i)
      function();
    const double time = Utilities::MPI::max(timer.wall_time(), comm) / n_calls;
    best              = std::min(best, time);
    sum += time;
  }
  return {best, sum / prm.n_samples};
}



template<typename Number>
void
run(const Parameters &                prm,
    Setup &                           setup,
    const std::string &               ordering,
    const std::vector<unsigned int> & categories,
    const MPI_Comm                    comm_sm)
{
  using VectorType        = LinearAlgebra::distributed::Vector<Number>;
  const MPI_Comm     comm = MPI_COMM_WORLD;
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0);

  RTOperator::RaviartThomasOperator<dim, Number> op;
  op.reinit(*setup.mapping,
            setup.dof_handler,
            setup.constraints,
            categories,
            QGauss<1>(prm.degree + 1),
            !prm.timings,
            comm_sm);
  op.set_penalty_parameters(1.);
  op.set_parameters(1., 1e-2);

  if(prm.statistics)
    op.print_data_locality_statistics();

  VectorType x, x_old, dst, rhs, inverse_diagonal, r;
  op.initialize_dof_vector(x);
  op.initialize_dof_vector(x_old);
  op.initialize_dof_vector(dst);
  op.initialize_dof_vector(rhs);
  op.initialize_dof_vector(r);
  op.initialize_dof_vector(inverse_diagonal);
  op.compute_diagonal(inverse_diagonal);
  for(Number & a : inverse_diagonal)
    a = Number(1.) / a;
  x   = Number(1.);
  rhs = Number(1.);
  r   = Number(0.5);

  std::unique_ptr<RTOperator::RaviartThomasOperator<dim, Number>> op_reverse;

  const double n_dofs = x.size();
  const auto   print  = [&](const std::string & name, const std::pair<double, double> & times) {
    pcout << "  " << std::left << std::setw(12) << ordering << std::setw(7)
          << (std::is_same_v<Number, double> ? "double" : "float") << std::setw(12) << name
          << std::right << std::setw(9) << std::fixed << std::setprecision(3)
          << n_dofs * 1e-9 / times.first << " GDoF/s (best), " << std::setw(9)
          << n_dofs * 1e-9 / times.second << " GDoF/s (avg), " << std::setw(10) << std::defaultfloat
          << std::setprecision(4) << times.first * 1e3 << " ms" << std::endl;
  };

  // check of the result: norm of the product with a vector of ones, which
  // does not depend on the order of the cells or the numbering of the DoFs
  op.vmult(dst, x);
  const double norm_vmult = dst.l2_norm();
  op.evaluate_convective_term(x, Number(0.5), dst);
  const double norm_convective = dst.l2_norm();
  pcout << "  " << std::left << std::setw(12) << ordering << std::setw(7)
        << (std::is_same_v<Number, double> ? "double" : "float")
        << "check: |A 1| = " << std::setprecision(10) << norm_vmult
        << ", |C(1)| = " << norm_convective << std::setprecision(6) << std::endl;

  for(const std::string & benchmark : prm.benchmarks)
  {
    if(benchmark == "scratch")
    {
      // operator alone, copying the result into a global vector
      print(benchmark, measure(prm, [&]() {
              op.vmult_scratch(
                x, [&](const unsigned int begin, const unsigned int end, const Number * values) {
                  Number * d = dst.begin();
                  DEAL_II_OPENMP_SIMD_PRAGMA
                  for(unsigned int i = begin; i < end; ++i)
                    d[i] = values[i - begin];
                });
            }));
    }
    else if(benchmark == "chebyshev" || benchmark == "cheb_rev" || benchmark == "cheb_alt")
    {
      // a step of a Chebyshev iteration with the result held in the scratch
      // array, writing x^{n+1} over x^{n-1} as in
      // PreconditionChebyshevScratch, with the iterate kept bounded. The
      // variants 'cheb_rev' and 'cheb_alt' use an operator with the reverse
      // order of the cells, always or in every second step, to assess the
      // reuse of cached data between consecutive passes over the vectors
      if(benchmark != "chebyshev" && op_reverse == nullptr)
      {
        AssertThrow(!categories.empty(), ExcMessage("Need categories to reverse"));
        std::vector<unsigned int> reverse_categories = categories;
        unsigned int              max_category       = 0;
        for(const unsigned int c : categories)
          if(c != numbers::invalid_unsigned_int)
            max_category = std::max(max_category, c);
        for(unsigned int & c : reverse_categories)
          if(c != numbers::invalid_unsigned_int)
            c = max_category - c;
        op_reverse = std::make_unique<RTOperator::RaviartThomasOperator<dim, Number>>();
        op_reverse->reinit(*setup.mapping,
                           setup.dof_handler,
                           setup.constraints,
                           reverse_categories,
                           QGauss<1>(prm.degree + 1),
                           true,
                           comm_sm);
        op_reverse->set_penalty_parameters(1.);
        op_reverse->set_parameters(1., 1e-2);
      }
      const Number factor_1 = 0.2, factor_2 = 0.4;
      x                  = Number(1.);
      x_old              = Number(1.);
      unsigned int count = 0;
      print(benchmark, measure(prm, [&]() {
              const auto & op_now =
                (benchmark == "cheb_rev" || (benchmark == "cheb_alt" && count % 2 == 1)) ?
                  *op_reverse :
                  op;
              ++count;
              op_now.vmult_scratch(
                x, [&](const unsigned int begin, const unsigned int end, const Number * values) {
                  const Number * xn   = x.begin();
                  Number *       xo   = x_old.begin();
                  const Number * diag = inverse_diagonal.begin();
                  const Number * b    = rhs.begin();
                  DEAL_II_OPENMP_SIMD_PRAGMA
                  for(unsigned int i = begin; i < end; ++i)
                    xo[i] = xn[i] + factor_1 * (xn[i] - xo[i]) +
                            factor_2 * diag[i] * (b[i] - values[i - begin]);
                });
              x.swap(x_old);
            }));
    }
    else if(benchmark == "cg")
    {
      // product of the conjugate gradient method of deal.II with the
      // operations on the vectors fused into the loop as in
      // SolverCG::IterationWorker::do_matrix_vector(): updates of the
      // solution, residual and search direction before the first access,
      // where x_old plays the role of the solution, r the residual, x the
      // search direction p and dst the product v = A p, and the inner
      // products after the last write; the factors keep the vectors bounded
      x                  = Number(1.);
      x_old              = Number(0.);
      r                  = Number(0.5);
      const Number alpha = 0., beta = 0.25;
      double       sum = 0;
      print(benchmark, measure(prm, [&]() {
              std::array<VectorizedArray<Number>, 4> sums = {};
              op.vmult(
                dst,
                x,
                [&](const unsigned int begin, const unsigned int end) {
                  Number *       sol  = x_old.begin();
                  Number *       res  = r.begin();
                  Number *       p    = x.begin();
                  Number *       v    = dst.begin();
                  const Number * diag = inverse_diagonal.begin();
                  DEAL_II_OPENMP_SIMD_PRAGMA
                  for(unsigned int i = begin; i < end; ++i)
                  {
                    sol[i] += alpha * p[i];
                    res[i] -= alpha * v[i];
                    p[i] = diag[i] * res[i] + beta * p[i];
                    v[i] = 0;
                  }
                },
                [&](const unsigned int begin, const unsigned int end) {
                  const Number *         p           = x.begin();
                  const Number *         v           = dst.begin();
                  const Number *         res         = r.begin();
                  const Number *         diag        = inverse_diagonal.begin();
                  constexpr unsigned int n_lanes     = VectorizedArray<Number>::size();
                  const unsigned int     end_regular = begin + (end - begin) / n_lanes * n_lanes;
                  for(unsigned int i = begin; i < end_regular; i += n_lanes)
                  {
                    VectorizedArray<Number> pi, vi, ri, di;
                    pi.load(p + i);
                    vi.load(v + i);
                    ri.load(res + i);
                    di.load(diag + i);
                    sums[0] += pi * vi;
                    sums[1] += vi * vi;
                    sums[2] += ri * vi;
                    sums[3] += di * ri * vi;
                  }
                  for(unsigned int i = end_regular; i < end; ++i)
                  {
                    sums[0][0] += p[i] * v[i];
                    sums[1][0] += v[i] * v[i];
                    sums[2][0] += res[i] * v[i];
                    sums[3][0] += diag[i] * res[i] * v[i];
                  }
                });
              std::array<double, 4> scalar_sums;
              for(unsigned int i = 0; i < 4; ++i)
                scalar_sums[i] = sums[i].sum();
              Utilities::MPI::sum(ArrayView<const double>(scalar_sums.data(), 4),
                                  comm,
                                  ArrayView<double>(scalar_sums.data(), 4));
              sum += scalar_sums[0];
            }));
      if(sum == 12345.)
        pcout << "";
    }
    else if(benchmark == "cg_vectors" || benchmark == "cg_unfused")
    {
      // references for the conjugate gradient method: the same vector
      // operations as in 'cg' in separate sweeps over the vectors, alone
      // ('cg_vectors', the memory traffic without the operator) or together
      // with vmult() without fused operations ('cg_unfused')
      x                        = Number(1.);
      x_old                    = Number(0.);
      r                        = Number(0.5);
      const Number       alpha = 0., beta = 0.25;
      double             sum     = 0;
      const unsigned int n_owned = x.locally_owned_size();
      print(benchmark, measure(prm, [&]() {
              Number *       sol  = x_old.begin();
              Number *       res  = r.begin();
              Number *       p    = x.begin();
              Number *       v    = dst.begin();
              const Number * diag = inverse_diagonal.begin();
              DEAL_II_OPENMP_SIMD_PRAGMA
              for(unsigned int i = 0; i < n_owned; ++i)
              {
                sol[i] += alpha * p[i];
                res[i] -= alpha * v[i];
                p[i] = diag[i] * res[i] + beta * p[i];
              }
              if(benchmark == "cg_unfused")
                op.vmult(dst, x);
              else
                std::memset(v, 0, n_owned * sizeof(Number));
              std::array<VectorizedArray<Number>, 4> sums        = {};
              constexpr unsigned int                 n_lanes     = VectorizedArray<Number>::size();
              const unsigned int                     end_regular = n_owned / n_lanes * n_lanes;
              for(unsigned int i = 0; i < end_regular; i += n_lanes)
              {
                VectorizedArray<Number> pi, vi, ri, di;
                pi.load(p + i);
                vi.load(v + i);
                ri.load(res + i);
                di.load(diag + i);
                sums[0] += pi * vi;
                sums[1] += vi * vi;
                sums[2] += ri * vi;
                sums[3] += di * ri * vi;
              }
              for(unsigned int i = end_regular; i < n_owned; ++i)
                sums[0][0] += p[i] * v[i];
              std::array<double, 4> scalar_sums;
              for(unsigned int i = 0; i < 4; ++i)
                scalar_sums[i] = sums[i].sum();
              Utilities::MPI::sum(ArrayView<const double>(scalar_sums.data(), 4),
                                  comm,
                                  ArrayView<double>(scalar_sums.data(), 4));
              sum += scalar_sums[0];
            }));
      if(sum == 12345.)
        pcout << "";
    }
    else if(benchmark == "convective")
    {
      x = Number(1.);
      print(benchmark, measure(prm, [&]() { op.evaluate_convective_term(x, Number(0.5), dst); }));
    }
    else
      AssertThrow(false, ExcMessage("Unknown benchmark " + benchmark));
  }
  pcout << std::endl;
}



// Benchmarks of the pressure: the DG Laplace operator and the coupling terms
// of the RT operator with the pressure, see Parameters::pressure_benchmarks
template<typename Number>
void
run_pressure(const Parameters &                prm,
             Setup &                           setup,
             const std::string &               ordering,
             const std::vector<unsigned int> & categories,
             const MPI_Comm                    comm_sm)
{
  using VectorType        = LinearAlgebra::distributed::Vector<Number>;
  const MPI_Comm     comm = MPI_COMM_WORLD;
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0);
  const std::string  number_name = std::is_same_v<Number, double> ? "double" : "float";

  LaplaceOperator::LaplaceOperatorDG<dim, Number> laplace;
  laplace.reinit(*setup.mapping,
                 setup.dof_handler_p,
                 setup.constraints_p,
                 categories,
                 QGauss<1>(setup.fe_p.degree + 1),
                 true);
  laplace.set_penalty_parameters(1.);
  laplace.set_parameters(0., 1.);
  if(prm.statistics)
    laplace.print_data_locality_statistics();

  RTOperator::RaviartThomasOperator<dim, Number> op;
  op.reinit(*setup.mapping,
            setup.dof_handler,
            setup.constraints,
            categories,
            QGauss<1>(prm.degree + 1),
            true,
            comm_sm);
  op.set_penalty_parameters(1.);
  op.set_parameters(1., 1e-2);
  op.verify_other_cell_level_index(laplace.get_cell_level_index());
  op.initialize_coupling_pressure(setup.fe_p, laplace.get_dof_indices());

  VectorType p, p_old, p_dst, p_rhs, p_res, p_diag, p_div;
  for(VectorType * v : {&p, &p_old, &p_dst, &p_rhs, &p_res, &p_diag, &p_div})
    laplace.initialize_dof_vector(*v);
  laplace.compute_inverse_diagonal(p_diag);
  VectorType u, u_dst;
  op.initialize_dof_vector(u);
  op.initialize_dof_vector(u_dst);
  u = Number(1.);

  const auto print =
    [&](const std::string & name, const double n_dofs, const std::pair<double, double> & times) {
      pcout << "  " << std::left << std::setw(12) << ordering << std::setw(7) << number_name
            << std::setw(16) << name << std::right << std::setw(9) << std::fixed
            << std::setprecision(3) << n_dofs * 1e-9 / times.first << " GDoF/s (best), "
            << std::setw(9) << n_dofs * 1e-9 / times.second << " GDoF/s (avg), " << std::setw(10)
            << std::defaultfloat << std::setprecision(4) << times.first * 1e3 << " ms" << std::endl;
    };

  // check: the norms do not depend on the order of the cells or the
  // numbering of the DoFs
  op.evaluate_convective_and_divergence_term(u, u_dst, p_dst, p_div);
  const double norm_div = p_div.l2_norm(), norm_conv_div = p_dst.l2_norm();
  op.evaluate_momentum_rhs(p_div, u, Number(1.), u_dst);
  const double norm_rhs = u_dst.l2_norm();
  laplace.vmult(p_dst, p_div);
  pcout << "  " << std::left << std::setw(12) << ordering << std::setw(7) << number_name
        << "check: |div| = " << std::setprecision(10) << norm_div
        << ", |conv div| = " << norm_conv_div << ", |rhs(div)| = " << norm_rhs
        << ", |L div| = " << p_dst.l2_norm() << std::setprecision(6) << std::endl;

  const double n_dofs_p = p.size(), n_dofs_u = u.size();
  for(const std::string & benchmark : prm.pressure_benchmarks)
  {
    p     = Number(1.);
    p_old = Number(1.);
    p_rhs = Number(1.);
    p_res = Number(0.5);
    if(benchmark == "p_scratch")
      print(benchmark, n_dofs_p, measure(prm, [&]() {
              laplace.vmult_scratch(
                p, [&](const unsigned int begin, const unsigned int end, const Number * values) {
                  Number * d = p_dst.begin();
                  DEAL_II_OPENMP_SIMD_PRAGMA
                  for(unsigned int i = begin; i < end; ++i)
                    d[i] = values[i - begin];
                });
            }));
    else if(benchmark == "p_chebyshev")
    {
      const Number factor_1 = 0.2, factor_2 = 0.4;
      print(benchmark, n_dofs_p, measure(prm, [&]() {
              laplace.vmult_scratch(
                p, [&](const unsigned int begin, const unsigned int end, const Number * values) {
                  const Number * xn   = p.begin();
                  Number *       xo   = p_old.begin();
                  const Number * diag = p_diag.begin();
                  const Number * b    = p_rhs.begin();
                  DEAL_II_OPENMP_SIMD_PRAGMA
                  for(unsigned int i = begin; i < end; ++i)
                    xo[i] = xn[i] + factor_1 * (xn[i] - xo[i]) +
                            factor_2 * diag[i] * (b[i] - values[i - begin]);
                });
              p.swap(p_old);
            }));
    }
    else if(benchmark == "p_cg")
    {
      const Number alpha = 0., beta = 0.25;
      double       sum = 0;
      print(benchmark, n_dofs_p, measure(prm, [&]() {
              std::array<VectorizedArray<Number>, 4> sums = {};
              laplace.vmult(
                p_dst,
                p,
                [&](const unsigned int begin, const unsigned int end) {
                  Number *       sol  = p_old.begin();
                  Number *       res  = p_res.begin();
                  Number *       pp   = p.begin();
                  Number *       v    = p_dst.begin();
                  const Number * diag = p_diag.begin();
                  DEAL_II_OPENMP_SIMD_PRAGMA
                  for(unsigned int i = begin; i < end; ++i)
                  {
                    sol[i] += alpha * pp[i];
                    res[i] -= alpha * v[i];
                    pp[i] = diag[i] * res[i] + beta * pp[i];
                    v[i]  = 0;
                  }
                },
                [&](const unsigned int begin, const unsigned int end) {
                  const Number * pp   = p.begin();
                  const Number * v    = p_dst.begin();
                  const Number * res  = p_res.begin();
                  const Number * diag = p_diag.begin();
                  for(unsigned int i = begin; i < end; ++i)
                  {
                    sums[0][0] += pp[i] * v[i];
                    sums[1][0] += v[i] * v[i];
                    sums[2][0] += res[i] * v[i];
                    sums[3][0] += diag[i] * res[i] * v[i];
                  }
                });
              std::array<double, 4> scalar_sums;
              for(unsigned int i = 0; i < 4; ++i)
                scalar_sums[i] = sums[i].sum();
              Utilities::MPI::sum(ArrayView<const double>(scalar_sums.data(), 4),
                                  comm,
                                  ArrayView<double>(scalar_sums.data(), 4));
              sum += scalar_sums[0];
            }));
      if(sum == 12345.)
        pcout << "";
    }
    else if(benchmark == "conv_div")
      print(benchmark + " [u]", n_dofs_u, measure(prm, [&]() {
              op.evaluate_convective_and_divergence_term(u, u_dst, p_dst, p_div);
            }));
    else if(benchmark == "momentum_rhs")
      print(benchmark + " [u]", n_dofs_u, measure(prm, [&]() {
              op.evaluate_momentum_rhs(p, u, Number(1.), u_dst);
            }));
    else
      AssertThrow(false, ExcMessage("Unknown pressure benchmark " + benchmark));
  }
  pcout << std::endl;
}

} // namespace



int
main(int argc, char ** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  const MPI_Comm                   comm = MPI_COMM_WORLD;
  ConditionalOStream               pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0);

  const Parameters prm = parse_parameters(argc, argv);

  MPI_Comm comm_sm = MPI_COMM_SELF;
  if(prm.shared_memory)
  {
    const int ierr = MPI_Comm_split_type(
      comm, MPI_COMM_TYPE_SHARED, Utilities::MPI::this_mpi_process(comm), MPI_INFO_NULL, &comm_sm);
    AssertThrowMPI(ierr);
  }

  Setup setup(prm);
  pcout << "Periodic hill mesh " << prm.subdivisions[0] << "x" << prm.subdivisions[1] << "x"
        << prm.subdivisions[2] << " refined " << prm.n_refinements
        << " times: " << setup.tria.n_global_active_cells() << " cells, RT degree " << prm.degree
        << ", " << Utilities::MPI::n_mpi_processes(comm) << " processes, SIMD width double "
        << VectorizedArray<double>::size() << ", DoF numbering '" << prm.renumbering << "'"
        << std::endl
        << std::endl;

  for(const std::string & ordering : prm.orderings)
  {
    const std::vector<unsigned int> categories =
      CellOrderings::compute_categories(ordering, setup.tria);
    setup.setup_dofs(prm.renumbering, categories);
    setup.setup_pressure_dofs(prm.pressure_renumbering, categories);
    pcout << "Order of cells '" << ordering << "', " << setup.dof_handler.n_dofs() << " DoFs"
          << std::endl;
    for(const std::string & number : prm.numbers)
      if(number == "double")
        run<double>(prm, setup, ordering, categories, comm_sm);
      else if(number == "float")
        run<float>(prm, setup, ordering, categories, comm_sm);
      else
        AssertThrow(false, ExcMessage("Unknown number type " + number));
    if(!prm.pressure_benchmarks.empty())
    {
      pcout << "Pressure with DoF numbering '" << prm.pressure_renumbering << "', "
            << setup.dof_handler_p.n_dofs() << " DoFs" << std::endl;
      for(const std::string & number : prm.numbers)
        if(number == "double")
          run_pressure<double>(prm, setup, ordering, categories, comm_sm);
        else
          run_pressure<float>(prm, setup, ordering, categories, comm_sm);
    }
  }

  if(comm_sm != MPI_COMM_SELF)
    MPI_Comm_free(&comm_sm);
}
