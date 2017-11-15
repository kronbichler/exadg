/*
 * evaluate_solution_in_given_point.h
 *
 *  Created on: Mar 15, 2017
 *      Author: fehn
 */

#ifndef INCLUDE_POSTPROCESSOR_EVALUATE_SOLUTION_IN_GIVEN_POINT_H_
#define INCLUDE_POSTPROCESSOR_EVALUATE_SOLUTION_IN_GIVEN_POINT_H_

#include <deal.II/lac/parallel_vector.h>

template<int dim, typename Number>
void my_point_value(Mapping<dim> const                                   &mapping,
                    DoFHandler<dim> const                                &dof_handler,
                    parallel::distributed::Vector<Number> const          &solution,
                    typename DoFHandler<dim>::active_cell_iterator const &cell,
                    Point<dim>  const                                    &point_in_ref_coord,
                    Vector<Number>                                       &value)
{
  Assert(GeometryInfo<dim>::distance_to_unit_cell(point_in_ref_coord) < 1e-10,ExcInternalError());

  const FiniteElement<dim> &fe = dof_handler.get_fe();

  const Quadrature<dim> quadrature (GeometryInfo<dim>::project_to_unit_cell(point_in_ref_coord));

  FEValues<dim> fe_values(mapping, fe, quadrature, update_values);
  fe_values.reinit(cell);

  // then use this to get the values of the given fe_function at this point
  std::vector<Vector<Number> > solution_value(1, Vector<Number> (fe.n_components()));
  fe_values.get_function_values(solution, solution_value);
  value = solution_value[0];
}

//template<int dim, typename Number>
//void evaluate_solution_in_point(DoFHandler<dim> const                       &dof_handler,
//                                Mapping<dim> const                          &mapping,
//                                parallel::distributed::Vector<Number> const &numerical_solution,
//                                Point<dim> const                            &point,
//                                Number                                      &solution_value)
//{
//  // processor local variables: initialize with zeros since we add values to these variables
//  unsigned int counter = 0;
//  solution_value = 0.0;
//
//  // find adjacent cells to specified point by calculating the closest vertex and the cells
//  // surrounding this vertex, make sure that at least one cell is found
//  unsigned int vertex_id = GridTools::find_closest_vertex(dof_handler, point);
//  std::vector<typename DoFHandler<dim>::active_cell_iterator> adjacent_cells_tmp
//    = GridTools::find_cells_adjacent_to_vertex(dof_handler,vertex_id);
//  Assert(adjacent_cells_tmp.size()>0, ExcMessage("No adjacent cells found for given point."));
//
//  // copy adjacent cells into a set
//  std::set<typename DoFHandler<dim>::active_cell_iterator> adjacent_cells (adjacent_cells_tmp.begin(),adjacent_cells_tmp.end());
//
//  // loop over all adjacent cells
//  for (typename std::set<typename DoFHandler<dim>::active_cell_iterator>::iterator cell = adjacent_cells.begin(); cell != adjacent_cells.end(); ++cell)
//  {
//    // go on only if cell is owned by the processor
//    if((*cell)->is_locally_owned())
//    {
//      // this is a safety factor and might be insufficient for strongly distorted elements
//      Number const factor = 1.1;
//      Point<dim> point_in_ref_coord;
//      // This if() is needed because the function transform_real_to_unit_cell() throws exception
//      // if the point is too far away from the cell.
//      // Hence, we make sure that the cell is only considered if the point is close to the cell.
//      if((*cell)->center().distance(point) < factor * (*cell)->center().distance(dof_handler.get_triangulation().get_vertices()[vertex_id]))
//      {
//        try
//        {
//          point_in_ref_coord = mapping.transform_real_to_unit_cell(*cell, point);
//        }
//        catch(...)
//        {
//          std::cerr << std::endl
//                    << "Could not transform point from real to unit cell. "
//                       "Probably, the specified point is too far away from the cell."
//                    << std::endl;
//        }
//
//        const Number distance = GeometryInfo<dim>::distance_to_unit_cell(point_in_ref_coord);
//
//        // if point lies on the current cell
//        if(distance < 1.0e-10)
//        {
//          Vector<Number> value(1);
//          my_point_value(mapping,
//                         dof_handler,
//                         numerical_solution,
//                         *cell,
//                         point_in_ref_coord,
//                         value);
//
//          solution_value += value(0);
//          ++counter;
//        }
//      }
//    }
//  }
//
//  // parallel computations: add results of all processors and calculate mean value
//  counter = Utilities::MPI::sum(counter,MPI_COMM_WORLD);
//  Assert(counter>0,ExcMessage("No points found."));
//
//  solution_value = Utilities::MPI::sum(solution_value,MPI_COMM_WORLD);
//  solution_value /= (Number)counter;
//}

/*
 *  Find all active cells around a point given in physical space.
 *  The return value is a std::vector of std::pair<cell,point_in_ref_coordinates>.
 *  A <cell,point_in_ref_coordinates> pair is inserted in this vector only if the
 *  distance of the point to the cell (in reference space) is smaller than a tolerance.
 */
template<int dim, template<int,int> class MeshType, int spacedim = dim>
std::vector<std::pair<typename MeshType<dim, spacedim>::active_cell_iterator, Point<dim> > >
find_all_active_cells_around_point(Mapping<dim> const           &mapping,
                                   MeshType<dim,spacedim> const &mesh,
                                   Point<dim> const             &p)
{
  std::vector<std::pair<typename MeshType<dim, spacedim>::active_cell_iterator, Point<dim> > > cells;

  // find adjacent cells to specified point by calculating the closest vertex and the cells
  // surrounding this vertex, make sure that at least one cell is found
  unsigned int vertex_id = GridTools::find_closest_vertex(mesh, p);
  std::vector<typename MeshType<dim,spacedim>::active_cell_iterator>
      adjacent_cells_tmp = GridTools::find_cells_adjacent_to_vertex(mesh,vertex_id);
  Assert(adjacent_cells_tmp.size()>0, ExcMessage("No adjacent cells found for given point."));

  // copy adjacent cells into a set
  std::set<typename MeshType<dim,spacedim>::active_cell_iterator>
    adjacent_cells (adjacent_cells_tmp.begin(),adjacent_cells_tmp.end());

  // loop over all adjacent cells
  typename std::set<typename MeshType<dim,spacedim>::active_cell_iterator>::iterator
    cell = adjacent_cells.begin(), endc = adjacent_cells.end();

  for (; cell != endc; ++cell)
  {
    Point<dim> point_in_ref_coord;

    try
    {
      point_in_ref_coord = mapping.transform_real_to_unit_cell(*cell, p);
    }
    catch(...)
    {
      // A point that does not lie on the reference cell.
      point_in_ref_coord[0] = 2.0;

//      std::cerr << std::endl
//                << "Could not transform point from real to unit cell. "
//                   "Probably, the specified point is too far away from the cell."
//                << std::endl;
    }

    const double distance = GeometryInfo<dim>::distance_to_unit_cell(point_in_ref_coord);

    // insert cell into vector if point lies on the current cell
    double const tol = 1.0e-10;
    if(distance < tol)
    {
      cells.push_back(std::make_pair(*cell,point_in_ref_coord));
    }
  }

  return cells;
}

template<int dim, typename Number>
void evaluate_scalar_quantity_in_point(DoFHandler<dim> const                       &dof_handler,
                                       Mapping<dim> const                          &mapping,
                                       parallel::distributed::Vector<double> const &numerical_solution,
                                       Point<dim> const                            &point,
                                       Number                                      &solution_value)
{
  // processor local variables: initialize with zeros since we add values to these variables
  unsigned int counter = 0;
  solution_value = 0.0;

  typedef std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim> > MY_PAIR;
  std::vector<MY_PAIR> adjacent_cells = find_all_active_cells_around_point(mapping,dof_handler,point);

  // loop over all adjacent cells
  for (typename std::vector<MY_PAIR>::iterator cell = adjacent_cells.begin(); cell != adjacent_cells.end(); ++cell)
  {
    // go on only if cell is owned by the processor
    if(cell->first->is_locally_owned())
    {
        Vector<Number> value(1);
        my_point_value(mapping,
                       dof_handler,
                       numerical_solution,
                       cell->first,
                       cell->second,
                       value);

        solution_value += value(0);
        ++counter;
    }
  }

  // parallel computations: add results of all processors and calculate mean value
  counter = Utilities::MPI::sum(counter,MPI_COMM_WORLD);
  Assert(counter>0,ExcMessage("No points found."));

  solution_value = Utilities::MPI::sum(solution_value,MPI_COMM_WORLD);
  solution_value /= (double)counter;
}

template<int dim, typename Number>
void evaluate_vectorial_quantity_in_point(DoFHandler<dim> const                       &dof_handler,
                                          Mapping<dim> const                          &mapping,
                                          parallel::distributed::Vector<double> const &numerical_solution,
                                          Point<dim> const                            &point,
                                          Tensor<1,dim,Number>                        &solution_value)
{
  // processor local variables: initialize with zeros since we add values to these variables
  unsigned int counter = 0;
  solution_value = 0.0;

  typedef std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim> > MY_PAIR;
  std::vector<MY_PAIR> adjacent_cells = find_all_active_cells_around_point(mapping,dof_handler,point);

  // loop over all adjacent cells
  for (typename std::vector<MY_PAIR>::iterator cell = adjacent_cells.begin(); cell != adjacent_cells.end(); ++cell)
  {
    // go on only if cell is owned by the processor
    if(cell->first->is_locally_owned())
    {
      Vector<Number> value(dim);
      my_point_value(mapping,
                     dof_handler,
                     numerical_solution,
                     cell->first,
                     cell->second,
                     value);

      for(unsigned int d=0; d<dim; ++d)
        solution_value[d] += value(d);

      ++counter;
    }
  }

  // parallel computations: add results of all processors and calculate mean value
  counter = Utilities::MPI::sum(counter,MPI_COMM_WORLD);
  Assert(counter>0,ExcMessage("No points found."));

  for(unsigned int d=0; d<dim; ++d)
    solution_value[d] = Utilities::MPI::sum(solution_value[d],MPI_COMM_WORLD);
  solution_value /= (double)counter;
}

/*
 *  For a given point in physical space, find all adjacent cells and store the global dof index
 *  as well as the shape function values (to be used for interpolation of the solution in the given point afterwards).
 *  (global_dof_index, shape_values) are stored in a vector where each entry corresponds to one adjacent, locally-owned cell.
 */
template<int dim, typename Number>
void get_global_dof_index_and_shape_values(DoFHandler<dim> const                                      &dof_handler,
                                           Mapping<dim> const                                         &mapping,
                                           parallel::distributed::Vector<double> const                &numerical_solution,
                                           Point<dim> const                                           &point,
                                           std::vector<std::pair<unsigned int,std::vector<Number> > > &global_dof_index_and_shape_values)
{
  typedef std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim> > MY_PAIR;
  std::vector<MY_PAIR> adjacent_cells = find_all_active_cells_around_point(mapping,dof_handler,point);

  // loop over all adjacent cells
  for (typename std::vector<MY_PAIR>::iterator cell = adjacent_cells.begin(); cell != adjacent_cells.end(); ++cell)
  {
    // go on only if cell is owned by the processor
    if(cell->first->is_locally_owned())
    {
      Assert(GeometryInfo<dim>::distance_to_unit_cell(cell->second) < 1e-10,ExcInternalError());

      const FiniteElement<dim> &fe = dof_handler.get_fe();
      const Quadrature<dim> quadrature (GeometryInfo<dim>::project_to_unit_cell(cell->second));
      FEValues<dim> fe_values(mapping, fe, quadrature, update_values);
      fe_values.reinit(cell->first);
      std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
      cell->first->get_dof_indices(dof_indices);
      unsigned int global_dof_index = numerical_solution.get_partitioner()->global_to_local(dof_indices[0]);
      std::vector<Number> fe_shape_values(fe.dofs_per_cell);
      for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
        fe_shape_values[i] = fe_values.shape_value(i,0);

      global_dof_index_and_shape_values.push_back(std::pair<unsigned int,std::vector<Number> >(global_dof_index,fe_shape_values));
    }
  }
}

/*
 *  Interpolate solution in point by using precomputed shape functions values (for efficiency!)
 *  Noet that we assume that we are dealing in discontinuous finite elements.
 */
template<int dim, typename Number>
void interpolate_value(DoFHandler<dim> const                                &dof_handler,
                       parallel::distributed::Vector<Number> const          &solution,
                       unsigned int const                                   &global_dof_index,
                       std::vector<Number> const                            &fe_shape_values,
                       Tensor<1,dim,Number>                                 &result)
{
  const FiniteElement<dim> &fe = dof_handler.get_fe();
  Number const * sol_ptr = solution.begin() + global_dof_index;
  for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
    result[fe.system_to_component_index(i).first] += sol_ptr[i] * fe_shape_values[i];
}


#endif /* INCLUDE_POSTPROCESSOR_EVALUATE_SOLUTION_IN_GIVEN_POINT_H_ */