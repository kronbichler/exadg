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
#include <deal.II/base/ndarray.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <boost/serialization/vector.hpp>

#include <atomic>
#include <functional>
#include <map>
#include <ostream>
#include <vector>

namespace RTOperator
{
// Setup data for a vmult of RaviartThomasOperator with a single MPI
// communication round. A naive implementation of vmult first runs
// update_ghost_values() to obtain all data needed for the finite element
// solution expansion on locally owned cells and then packs the face data
// (values and normal derivatives) of cells adjacent to remote processes. The
// data structures here allow to pack face data from locally owned DoFs
// only. The message from one process to a neighbor consists of three
// sections:
//
//  - G: raw ghost section, exactly the data update_ghost_values() would
//    send, placed in the receiver's ghost order,
//  - F: face packs computed with the regular indices of the sending cell;
//    the format is the one of the existing face exchange, i.e., values and
//    normal derivatives of the tangential components and only the normal
//    derivative of the normal component, without the contribution of the DoFs
//    on the shared face,
//  - S: supplements, i.e., the contribution of DoFs owned by this process
//    on a face of a cell of a third process to the face data the third
//    process sends to the receiver (at subdomain edges or for partitions only
//    one cell thick).
//
// The receiver uses section F from the receive buffer and completes it with
// the 'local supplements' for those entities of the remote cell it owns
// itself and by the received section S. The value of the normal component and
// the contribution of the DoFs on the shared face to its normal derivative
// are not part of the face data, as the operator takes them from the
// receiving cell itself.
//
// The linear maps from the DoFs on one face entity of a cell to the face
// data are obtained at setup by evaluating the face data of unit vectors with
// the same function the operator uses for packing (passed in as
// FaceEvaluator), which keeps the index arithmetic in one place.
namespace SingleExchange
{
using namespace dealii;

// A contribution the receiver of a face pack adds itself, because the
// receiver rather than the sender owns the DoFs on face 'sender_face' of the
// remote cell
struct LocalSupplement
{
  unsigned int  slot;
  unsigned char sender_face;
  unsigned int  local_index;
};

// A contribution from an entity owned by a third process, collected during
// setup before being turned into the lists of section S
struct RemoteSupplement
{
  unsigned int            slot;
  unsigned char           sender_face;
  types::global_dof_index global_index;
  unsigned int            owner;
};

// Entry of section S on the process owning the DoFs: the DoFs starting at
// 'local_index' sit on face 'entity' of the remote cell, whose face data is
// evaluated on face 'face'
struct SupplementSend
{
  unsigned int  local_index;
  unsigned char face;
  unsigned char entity;
};

// Entry of section S on the receiving process
struct SupplementRecv
{
  unsigned int  slot;
  unsigned char entity;
};

// All data exchanged with one neighboring process. Faces are numbered
// consecutively over all partners, with the same index for the send face
// (index into send_data_cell_index) and the receive slot (as stored in
// mpi_exchange_data_on_faces). Partners that only exchange supplements have
// n_faces == 0.
struct Partner
{
  unsigned int rank       = numbers::invalid_unsigned_int;
  unsigned int first_face = 0;
  unsigned int n_faces    = 0;

  // section G: ranges of locally owned indices to send, and position in the
  // ghost range of the vector (relative to locally_owned_size()) to receive
  std::vector<std::pair<unsigned int, unsigned int>> ghost_send_ranges;
  unsigned int                                       n_ghost_send     = 0;
  unsigned int                                       ghost_recv_start = 0;
  unsigned int                                       n_ghost_recv     = 0;

  // section S
  std::vector<SupplementSend> supplements_send;
  std::vector<SupplementRecv> supplements_recv;
  unsigned int                n_supplement_send_entries = 0;
  unsigned int                n_supplement_recv_entries = 0;

  // position of the message in the send and receive buffers, counted in
  // entries of type Number: sections G, F, S in this order
  unsigned int send_offset = 0;
  unsigned int send_size   = 0;
  unsigned int recv_offset = 0;
  unsigned int recv_size   = 0;

  // for partners in the same shared-memory domain: rank in the
  // shared-memory communicator, else numbers::invalid_unsigned_int, and the
  // start of the message to this process in the send buffer of the partner
  unsigned int sm_rank            = numbers::invalid_unsigned_int;
  unsigned int remote_send_offset = 0;

  // whether the partner reads all cells of section F directly in shared
  // memory, in which case the face data is not computed
  bool faces_read_directly = false;
};

template<int dim, typename Number>
class Data
{
public:
  static constexpr unsigned int n_lanes = VectorizedArray<Number>::size();

  void
  reinit(
    const DoFHandler<dim> &                                                  dof_handler,
    const internal::MatrixFreeFunctions::ShapeInfo<Number> &                 shape_info,
    const Utilities::MPI::Partitioner &                                      partitioner,
    const std::vector<dealii::ndarray<unsigned int, 2 * dim + 1, n_lanes>> & dof_indices,
    const std::vector<dealii::ndarray<unsigned int, n_lanes, 2>> &           cell_level_index,
    const std::vector<dealii::ndarray<unsigned int, 2 * dim, n_lanes>> & mpi_exchange_data_on_faces,
    const std::vector<std::pair<unsigned int, unsigned int>> &           send_data_process,
    const std::vector<unsigned int> &                                    send_data_cell_index,
    const std::vector<unsigned char> &                                   send_data_face_index)
  {
    const MPI_Comm     comm         = dof_handler.get_mpi_communicator();
    const unsigned int my_rank      = Utilities::MPI::this_mpi_process(comm);
    const unsigned int n_owned      = partitioner.locally_owned_size();
    const unsigned int n_send_faces = send_data_cell_index.size();
    n_locally_owned                 = n_owned;
    AssertDimension(n_send_faces, send_data_face_index.size());

    const Triangulation<dim> & tria     = dof_handler.get_triangulation();
    const auto                 get_cell = [&](const unsigned int cell, const unsigned int lane) {
      return typename Triangulation<dim>::cell_iterator(&tria,
                                                        cell_level_index[cell][lane][0],
                                                        cell_level_index[cell][lane][1]);
    };

    const unsigned int n_t = shape_info.data[0].fe_degree;
    AssertThrow(shape_info.data[1].fe_degree + 1 == n_t, ExcInternalError());
    const unsigned int n_n                = n_t + 1;
    dofs_per_face                         = Utilities::pow(n_t, dim - 1);
    const unsigned int tangential_on_face = n_n * Utilities::pow(n_t, dim - 2);
    data_per_face                         = dofs_per_face + 2 * (dim - 1) * tangential_on_face;

    setup_tensor_product_kernels(shape_info);

    // Set up partners in the order of send_data_process, which is the order
    // of the faces in the send and import buffers of the existing exchange
    partners.clear();
    std::map<unsigned int, unsigned int> partner_index;
    {
      unsigned int face_offset = 0;
      for(const auto & [rank, n_faces] : send_data_process)
      {
        Partner partner;
        partner.rank       = rank;
        partner.first_face = face_offset;
        partner.n_faces    = n_faces;
        face_offset += n_faces;
        partner_index[rank] = partners.size();
        partners.push_back(partner);
      }
      AssertThrow(face_offset == n_send_faces, ExcInternalError());
    }
    const auto get_or_add_partner = [&](const unsigned int rank) -> Partner & {
      const auto it = partner_index.find(rank);
      if(it != partner_index.end())
        return partners[it->second];
      Partner partner;
      partner.rank        = rank;
      partner.first_face  = n_send_faces;
      partner_index[rank] = partners.size();
      partners.push_back(partner);
      return partners.back();
    };

    // Section G, receive side: the ghost range of the vector is contiguous
    // per owning process in the order of ghost_targets()
    {
      unsigned int ghost_offset = 0;
      for(const auto & [rank, n_indices] : partitioner.ghost_targets())
      {
        const auto it = partner_index.find(rank);
        AssertThrow(it != partner_index.end(),
                    ExcMessage("Process " + std::to_string(my_rank) + " has ghost DoFs owned by " +
                               std::to_string(rank) +
                               ", which is not a face neighbor. This is not expected for "
                               "Raviart-Thomas DoFs without hanging nodes."));
        partners[it->second].ghost_recv_start = ghost_offset;
        partners[it->second].n_ghost_recv     = n_indices;
        ghost_offset += n_indices;
      }
      AssertThrow(ghost_offset == partitioner.n_ghost_indices(), ExcInternalError());
    }

    // Section G, send side: split the import index ranges by target process
    {
      const auto & import_ranges = partitioner.import_indices();
      auto         range         = import_ranges.begin();
      unsigned int consumed      = 0;
      for(const auto & [rank, n_indices] : partitioner.import_targets())
      {
        const auto it = partner_index.find(rank);
        AssertThrow(it != partner_index.end(),
                    ExcMessage("Process " + std::to_string(my_rank) + " exports DoFs to " +
                               std::to_string(rank) + ", which is not a face neighbor."));
        Partner & partner      = partners[it->second];
        partner.n_ghost_send   = n_indices;
        unsigned int remaining = n_indices;
        while(remaining > 0)
        {
          AssertThrow(range != import_ranges.end(), ExcInternalError());
          const unsigned int start  = range->first + consumed;
          const unsigned int length = std::min(remaining, range->second - start);
          partner.ghost_send_ranges.emplace_back(start, start + length);
          remaining -= length;
          consumed += length;
          if(range->first + consumed == range->second)
          {
            ++range;
            consumed = 0;
          }
        }
      }
      AssertThrow(range == import_ranges.end(), ExcInternalError());
    }

    const auto ghost_owner = [&](const unsigned int local_index) {
      unsigned int start = n_owned;
      for(const auto & [rank, n_indices] : partitioner.ghost_targets())
      {
        if(local_index < start + n_indices)
          return rank;
        start += n_indices;
      }
      AssertThrow(false, ExcInternalError());
      return numbers::invalid_unsigned_int;
    };

    // Send side: collect, per face, the information the receiver needs to
    // complete the face data. The message contains per face: global cell
    // index of the sending cell, global cell index of the receiving cell,
    // face number on the sending side, global index of the first DoF on the
    // shared face, number of ghost entities of the sending cell other than
    // the shared face, and for each such entity (face number on the sending
    // cell, global index of the first DoF, owner).
    std::map<unsigned int, std::vector<types::global_dof_index>> setup_send;
    send_cell_entities.resize(n_send_faces);
    first_owned_index = partitioner.local_range().first;
    for(const Partner & partner : partners)
    {
      std::vector<types::global_dof_index> & message = setup_send[partner.rank];
      for(unsigned int c = partner.first_face; c < partner.first_face + partner.n_faces; ++c)
      {
        const unsigned int cell              = send_data_cell_index[c] / n_lanes;
        const unsigned int lane              = send_data_cell_index[c] % n_lanes;
        const unsigned int face              = send_data_face_index[c];
        const auto         dcell             = get_cell(cell, lane);
        const auto [neighbor, neighbor_face] = get_neighbor(dcell, face);
        check_face_pair(tria, dcell, face, neighbor, neighbor_face, partner.rank);

        const auto & indices = dof_indices[cell];
        AssertThrow(indices[2 * dim][lane] < n_owned,
                    ExcMessage("Cell-interior DoFs must always be locally owned"));
        AssertThrow(indices[face][lane] != numbers::invalid_unsigned_int,
                    ExcMessage("DoFs on face to remote process must not be constrained"));

        // all entities of the sending cell, for reading them directly in
        // shared memory, see setup_shared_memory()
        for(unsigned int e = 0; e < 2 * dim + 1; ++e)
        {
          const unsigned int index = indices[e][lane];
          if(index == numbers::invalid_unsigned_int)
            send_cell_entities[c][e] = {
              {numbers::invalid_dof_index, numbers::invalid_unsigned_int}};
          else
            send_cell_entities[c][e] = {
              {partitioner.local_to_global(index), index < n_owned ? my_rank : ghost_owner(index)}};
        }

        message.push_back(dcell->global_active_cell_index());
        message.push_back(neighbor->global_active_cell_index());
        message.push_back(face);
        message.push_back(partitioner.local_to_global(indices[face][lane]));

        const std::size_t count_position = message.size();
        message.push_back(0);
        for(unsigned int g = 0; g < 2 * dim; ++g)
          if(g != face && indices[g][lane] != numbers::invalid_unsigned_int &&
             indices[g][lane] >= n_owned)
          {
            const unsigned int owner = ghost_owner(indices[g][lane]);
            AssertThrow(owner != my_rank, ExcInternalError());
            message.push_back(g);
            message.push_back(partitioner.local_to_global(indices[g][lane]));
            message.push_back(owner);
            ++message[count_position];
          }
      }
    }

    // Receive side: inverse map from slot to (cell batch, face, lane)
    std::vector<std::array<unsigned int, 3>> slot_info(
      n_send_faces, {{numbers::invalid_unsigned_int, numbers::invalid_unsigned_int, 0}});
    for(unsigned int cell = 0; cell < mpi_exchange_data_on_faces.size(); ++cell)
      for(unsigned int f = 0; f < 2 * dim; ++f)
        for(unsigned int v = 0; v < n_lanes; ++v)
        {
          const unsigned int slot = mpi_exchange_data_on_faces[cell][f][v];
          if(slot != numbers::invalid_unsigned_int)
          {
            AssertThrow(slot < n_send_faces && slot_info[slot][0] == numbers::invalid_unsigned_int,
                        ExcInternalError());
            slot_info[slot] = {{cell, f, v}};
          }
        }
    n_cell_batches = mpi_exchange_data_on_faces.size();
    slot_location  = slot_info;

    const std::map<unsigned int, std::vector<types::global_dof_index>> setup_received =
      Utilities::MPI::some_to_some(comm, setup_send);

    recv_sender_face.assign(n_send_faces, static_cast<unsigned char>(255));
    local_supplements.clear();
    std::vector<RemoteSupplement> remote_supplements;
    for(const Partner & partner : partners)
    {
      if(partner.n_faces == 0)
        continue;
      const auto it = setup_received.find(partner.rank);
      AssertThrow(it != setup_received.end(),
                  ExcMessage("No setup data received from face neighbor " +
                             std::to_string(partner.rank)));
      const std::vector<types::global_dof_index> & message  = it->second;
      std::size_t                                  position = 0;
      for(unsigned int slot = partner.first_face; slot < partner.first_face + partner.n_faces;
          ++slot)
      {
        const auto [cell, f, v] = slot_info[slot];
        AssertThrow(cell != numbers::invalid_unsigned_int, ExcInternalError());
        AssertThrow(position + 5 <= message.size(), ExcInternalError());
        const types::global_dof_index sender_cell_index   = message[position++];
        const types::global_dof_index receiver_cell_index = message[position++];
        const unsigned int            sender_face         = message[position++];
        const types::global_dof_index face_dof_index      = message[position++];
        const unsigned int            n_entries           = message[position++];

        // Check that the order of faces matches between the two processes
        // and that both sides refer to the same DoFs on the shared face,
        // including the remapping of periodic DoFs
        const auto dcell = get_cell(cell, v);
        AssertThrow(dcell->global_active_cell_index() == receiver_cell_index &&
                      get_neighbor(dcell, f).first->global_active_cell_index() ==
                        sender_cell_index &&
                      f == (sender_face ^ 1),
                    ExcMessage("Mismatch in the order of faces exchanged between processes " +
                               std::to_string(my_rank) + " and " + std::to_string(partner.rank)));
        const unsigned int own_index = dof_indices[cell][f][v];
        AssertThrow(own_index != numbers::invalid_unsigned_int &&
                      partitioner.local_to_global(own_index) == face_dof_index,
                    ExcMessage("DoFs on face shared between processes " + std::to_string(my_rank) +
                               " and " + std::to_string(partner.rank) + " do not match"));
        recv_sender_face[slot] = sender_face;

        for(unsigned int e = 0; e < n_entries; ++e)
        {
          AssertThrow(position + 3 <= message.size(), ExcInternalError());
          const unsigned int            g            = message[position++];
          const types::global_dof_index global_index = message[position++];
          const unsigned int            owner        = message[position++];
          AssertThrow(g < 2 * dim && g != sender_face && owner != partner.rank, ExcInternalError());
          if(owner == my_rank)
          {
            AssertThrow(partitioner.in_local_range(global_index), ExcInternalError());
            local_supplements.push_back(LocalSupplement{
              slot,
              static_cast<unsigned char>(g),
              static_cast<unsigned int>(partitioner.global_to_local(global_index))});
          }
          else
            remote_supplements.push_back(
              RemoteSupplement{slot, static_cast<unsigned char>(g), global_index, owner});
        }
      }
      AssertThrow(position == message.size(), ExcInternalError());
    }

    // Section S: ask the owners of the remote supplements to send their
    // contribution; the request contains the global index of the first DoF
    // of the entity, the face the data is evaluated on, and the entity
    // number on the remote cell. Both lists are in the order of the requests.
    {
      std::map<unsigned int, std::vector<types::global_dof_index>> requests;
      for(const RemoteSupplement & s : remote_supplements)
      {
        std::vector<types::global_dof_index> & request = requests[s.owner];
        request.push_back(s.global_index);
        request.push_back(recv_sender_face[s.slot]);
        request.push_back(s.sender_face);
        Partner & partner = get_or_add_partner(s.owner);
        partner.supplements_recv.push_back(SupplementRecv{s.slot, s.sender_face});
        partner.n_supplement_recv_entries +=
          supplement_size(n_t, recv_sender_face[s.slot], s.sender_face);
      }

      const std::map<unsigned int, std::vector<types::global_dof_index>> requests_received =
        Utilities::MPI::some_to_some(comm, requests);
      for(const auto & [rank, request] : requests_received)
      {
        AssertThrow(request.size() % 3 == 0, ExcInternalError());
        Partner & partner = get_or_add_partner(rank);
        for(unsigned int i = 0; i < request.size(); i += 3)
        {
          AssertThrow(partitioner.in_local_range(request[i]),
                      ExcMessage("Process " + std::to_string(rank) +
                                 " requested a supplement for DoFs not owned by " +
                                 std::to_string(my_rank)));
          const unsigned int face   = request[i + 1];
          const unsigned int entity = request[i + 2];
          AssertThrow(face < 2 * dim && entity < 2 * dim && face != entity, ExcInternalError());
          partner.supplements_send.push_back(
            SupplementSend{static_cast<unsigned int>(partitioner.global_to_local(request[i])),
                           static_cast<unsigned char>(face),
                           static_cast<unsigned char>(entity)});
          partner.n_supplement_send_entries += supplement_size(n_t, face, entity);
        }
      }
    }

    // Buffer layout: sections G, F, S for each partner
    {
      unsigned int send_offset = 0, recv_offset = 0;
      for(Partner & partner : partners)
      {
        partner.send_offset = send_offset;
        partner.send_size   = partner.n_ghost_send + partner.n_faces * data_per_face +
                            partner.n_supplement_send_entries;
        partner.recv_offset = recv_offset;
        partner.recv_size   = partner.n_ghost_recv + partner.n_faces * data_per_face +
                            partner.n_supplement_recv_entries;
        send_offset += partner.send_size;
        recv_offset += partner.recv_size;
      }
      send_buffer_size = send_offset;
      recv_buffer_size = recv_offset;

      recv_face_offset.resize(n_send_faces);
      for(const Partner & partner : partners)
        for(unsigned int slot = partner.first_face; slot < partner.first_face + partner.n_faces;
            ++slot)
          recv_face_offset[slot] = partner.recv_offset + partner.n_ghost_recv +
                                   (slot - partner.first_face) * data_per_face;
    }

    compute_statistics(partitioner, remote_supplements, comm);
  }

  // Copy section G of the message to 'partner' from the locally owned
  // entries of the vector 'src' into the send buffer
  void
  pack_ghost_section(const Partner & partner, const Number * src, Number * send_buffer) const
  {
    Number * out = send_buffer + partner.send_offset;
    for(const auto & [begin, end] : partner.ghost_send_ranges)
    {
      std::copy(src + begin, src + end, out);
      out += end - begin;
    }
  }

  // Start of section F of the message to 'partner' in the send buffer,
  // filled by the caller with data_per_face entries per face
  Number *
  face_section(const Partner & partner, Number * send_buffer) const
  {
    return send_buffer + partner.send_offset + partner.n_ghost_send;
  }

  // Compute section S of the message to 'partner' from locally owned
  // entries of the vector 'src'
  void
  pack_supplement_section(const Partner & partner, const Number * src, Number * send_buffer) const
  {
    dispatch_degree([&](auto n_t_constant) {
      pack_supplement_section_tensor<decltype(n_t_constant)::value>(partner, src, send_buffer);
    });
  }

  // Post the receives for all partners into requests[0, partners.size())
  void
  start_receives(Number * recv_buffer, const MPI_Comm comm, MPI_Request * requests) const
  {
    exchange_pending = true;
    for(unsigned int p = 0; p < partners.size(); ++p)
      if(partners[p].recv_size > 0)
      {
        const bool shared = partners[p].sm_rank != numbers::invalid_unsigned_int;
        const int  ierr   = MPI_Irecv(recv_buffer + partners[p].recv_offset,
                                   shared ? 0 : partners[p].recv_size * sizeof(Number),
                                   MPI_BYTE,
                                   partners[p].rank,
                                   mpi_tag,
                                   comm,
                                   &requests[p]);
        AssertThrowMPI(ierr);
      }
      else
        requests[p] = MPI_REQUEST_NULL;
  }

  // Send the message to partner p, with the request stored in requests[p]
  void
  start_send(const unsigned int p,
             const Number *     send_buffer,
             const MPI_Comm     comm,
             MPI_Request *      requests) const
  {
    if(partners[p].send_size > 0)
    {
      const bool shared = partners[p].sm_rank != numbers::invalid_unsigned_int;
      if(shared)
        std::atomic_thread_fence(std::memory_order_release);
      const int ierr = MPI_Isend(send_buffer + partners[p].send_offset,
                                 shared ? 0 : partners[p].send_size * sizeof(Number),
                                 MPI_BYTE,
                                 partners[p].rank,
                                 mpi_tag,
                                 comm,
                                 &requests[p]);
      AssertThrowMPI(ierr);
    }
    else
      requests[p] = MPI_REQUEST_NULL;
  }

  // After all messages have arrived: write section G into the ghost range
  // of the vector starting at 'vector_begin' and add section S and the
  // local supplements to the face data of section F, which stays in place in
  // the receive buffer (see recv_face_offset)
  void
  unpack_and_complete(Number * recv_buffer, Number * vector_begin) const
  {
    dispatch_degree([&](auto n_t_constant) {
      unpack_and_complete_tensor<decltype(n_t_constant)::value>(recv_buffer, vector_begin);
    });
  }

  // Identify the partners in the same shared-memory domain as given by
  // 'comm_sm' and look up the position of their messages to this process in
  // their send buffers. For these partners, the messages are not sent but
  // read directly from the send buffer of the partner, which must be
  // allocated in shared memory and passed to set_buffers(). The MPI
  // messages to these partners are empty and only signal that the send
  // buffer is ready. Must be called by all processes of 'comm'.
  void
  setup_shared_memory(const MPI_Comm comm, const MPI_Comm comm_sm)
  {
    free_communication();
    for(Partner & partner : partners)
    {
      partner.sm_rank             = numbers::invalid_unsigned_int;
      partner.faces_read_directly = false;
    }
    slot_in_shared_memory.clear();
    remote_cell_entities.clear();
    remote_cell_global_indices.clear();
    if(comm_sm == MPI_COMM_SELF)
      return;

    MPI_Group group, group_sm;
    int       ierr = MPI_Comm_group(comm, &group);
    AssertThrowMPI(ierr);
    ierr = MPI_Comm_group(comm_sm, &group_sm);
    AssertThrowMPI(ierr);
    std::vector<int> ranks(partners.size()), ranks_sm(partners.size());
    for(unsigned int p = 0; p < partners.size(); ++p)
      ranks[p] = partners[p].rank;
    ierr =
      MPI_Group_translate_ranks(group, partners.size(), ranks.data(), group_sm, ranks_sm.data());
    AssertThrowMPI(ierr);
    MPI_Group_free(&group);
    MPI_Group_free(&group_sm);
    for(unsigned int p = 0; p < partners.size(); ++p)
      if(ranks_sm[p] != MPI_UNDEFINED)
        partners[p].sm_rank = ranks_sm[p];

    // exchange the offsets of the messages in the send buffers
    std::vector<MPI_Request> requests;
    requests.reserve(2 * partners.size());
    for(Partner & partner : partners)
      if(partner.sm_rank != numbers::invalid_unsigned_int)
      {
        if(partner.recv_size > 0)
        {
          requests.emplace_back();
          ierr = MPI_Irecv(&partner.remote_send_offset,
                           1,
                           MPI_UNSIGNED,
                           partner.rank,
                           mpi_tag + 1,
                           comm,
                           &requests.back());
          AssertThrowMPI(ierr);
        }
        if(partner.send_size > 0)
        {
          requests.emplace_back();
          ierr = MPI_Isend(&partner.send_offset,
                           1,
                           MPI_UNSIGNED,
                           partner.rank,
                           mpi_tag + 1,
                           comm,
                           &requests.back());
          AssertThrowMPI(ierr);
        }
      }
    ierr = MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    AssertThrowMPI(ierr);

    setup_remote_cell_entities(comm, comm_sm);

    // separate communicator for the signals that the messages have been
    // read, which are still in flight when the operator returns and must not
    // be confused with messages of other operators
    ierr = MPI_Comm_dup(comm, &comm_done);
    AssertThrowMPI(ierr);
    done_requests.assign(2 * partners.size(), MPI_REQUEST_NULL);
  }

  // Set the location of the received data for all partners: the receive
  // buffer for partners communicating via MPI messages and the send buffers
  // of the partners in shared memory, given by the views into the send
  // buffers of all processes in the shared-memory communicator. Face data
  // from shared memory that must be completed by supplements is copied
  // into the receive buffer in unpack_and_complete(). Fills
  // remote_face_data, so the buffers must not be reallocated afterwards.
  void
  set_buffers(Number *                                     recv_buffer,
              const std::vector<ArrayView<const Number>> & shared_send_buffers)
  {
    std::vector<bool> needs_completion(recv_face_offset.size(), false);
    for(const Partner & partner : partners)
      for(const SupplementRecv & s : partner.supplements_recv)
        needs_completion[s.slot] = true;
    for(const LocalSupplement & s : local_supplements)
      needs_completion[s.slot] = true;

    recv_data.resize(partners.size());
    {
      dealii::ndarray<const Number *, 2 * dim, n_lanes> no_remote_data;
      for(unsigned int f = 0; f < 2 * dim; ++f)
        for(unsigned int v = 0; v < n_lanes; ++v)
          no_remote_data[f][v] = nullptr;
      remote_face_data.assign(n_cell_batches, no_remote_data);
      dealii::ndarray<const std::array<EntityLocation, 2 * dim + 1> *, 2 * dim, n_lanes>
        no_remote_cell;
      for(unsigned int f = 0; f < 2 * dim; ++f)
        for(unsigned int v = 0; v < n_lanes; ++v)
          no_remote_cell[f][v] = nullptr;
      remote_cell_locations.assign(n_cell_batches, no_remote_cell);
    }
    copy_to_recv_buffer.clear();
    for(unsigned int p = 0; p < partners.size(); ++p)
    {
      const Partner & partner = partners[p];
      if(partner.sm_rank == numbers::invalid_unsigned_int)
        recv_data[p] = recv_buffer + partner.recv_offset;
      else
      {
        AssertIndexRange(partner.sm_rank, shared_send_buffers.size());
        AssertIndexRange(partner.remote_send_offset + partner.recv_size,
                         shared_send_buffers[partner.sm_rank].size() + 1);
        recv_data[p] = shared_send_buffers[partner.sm_rank].data() + partner.remote_send_offset;
      }
      for(unsigned int slot = partner.first_face; slot < partner.first_face + partner.n_faces;
          ++slot)
      {
        const Number * data =
          recv_data[p] + partner.n_ghost_recv + (slot - partner.first_face) * data_per_face;
        const auto [cell, f, v] = slot_location[slot];
        if(slot < slot_in_shared_memory.size() && slot_in_shared_memory[slot])
        {
          // the remote cell is read directly in shared memory
          remote_cell_locations[cell][f][v] = &remote_cell_entities[slot];
          continue;
        }
        if(partner.sm_rank != numbers::invalid_unsigned_int && needs_completion[slot])
        {
          copy_to_recv_buffer.emplace_back(data, recv_face_offset[slot]);
          remote_face_data[cell][f][v] = recv_buffer + recv_face_offset[slot];
        }
        else
          remote_face_data[cell][f][v] = data;
      }
    }
  }

  // Wait until all partners in shared memory have signaled that they
  // finished reading the messages in the send buffer and the entries of the
  // source vector of the last exchange, which must be done before either is
  // modified, i.e., before returning from an operation using the exchange
  void
  wait_until_partners_done_reading() const
  {
    if(!done_requests.empty())
    {
      const int ierr = MPI_Waitall(done_requests.size(), done_requests.data(), MPI_STATUSES_IGNORE);
      AssertThrowMPI(ierr);
      std::atomic_thread_fence(std::memory_order_acquire);
    }
  }

  // After the last access to the received face data of an exchange: signal
  // the partners in shared memory that their send buffer may be reused, and
  // post the receives of the corresponding signals of the partners reading
  // from this process. Does nothing if no exchange is pending.
  void
  signal_done_reading() const
  {
    if(!exchange_pending)
      return;
    exchange_pending = false;
    if(done_requests.empty())
      return;
    std::atomic_thread_fence(std::memory_order_release);
    const unsigned int n_partners = partners.size();
    for(unsigned int p = 0; p < n_partners; ++p)
      if(partners[p].sm_rank != numbers::invalid_unsigned_int)
      {
        int ierr;
        if(partners[p].recv_size > 0)
        {
          ierr = MPI_Isend(
            nullptr, 0, MPI_BYTE, partners[p].rank, mpi_tag, comm_done, &done_requests[p]);
          AssertThrowMPI(ierr);
        }
        if(partners[p].send_size > 0)
        {
          ierr = MPI_Irecv(nullptr,
                           0,
                           MPI_BYTE,
                           partners[p].rank,
                           mpi_tag,
                           comm_done,
                           &done_requests[n_partners + p]);
          AssertThrowMPI(ierr);
        }
      }
  }

  // Location of an entity (face or cell interior) of a cell on another
  // process in the same shared-memory domain: rank in the shared-memory
  // communicator of the owner of the entity, and index of its first DoF in
  // the locally owned range of the owner; numbers::invalid_unsigned_int for
  // constrained entities
  struct EntityLocation
  {
    unsigned int sm_rank = numbers::invalid_unsigned_int;
    unsigned int offset  = numbers::invalid_unsigned_int;
  };

  // per receive slot: whether all entities of the remote cell are owned by
  // processes in the same shared-memory domain, and their locations in the
  // order of the entities of dof_indices; see setup_shared_memory()
  std::vector<bool>                                    slot_in_shared_memory;
  std::vector<std::array<EntityLocation, 2 * dim + 1>> remote_cell_entities;

  // Check the tables of remote_cell_entities with a vector in shared memory
  // whose locally owned entries hold their global index: return the number
  // of entities whose first entry, read through the table, does not match
  // the global index sent by the process owning the cell
  template<typename VectorType2>
  unsigned int
  count_wrong_remote_cell_entities(const VectorType2 & global_indices) const
  {
    const auto & shared_data = global_indices.shared_vector_data();
    unsigned int n_wrong     = 0;
    for(unsigned int slot = 0; slot < remote_cell_entities.size(); ++slot)
      if(slot_in_shared_memory[slot])
        for(unsigned int e = 0; e < 2 * dim + 1; ++e)
        {
          const EntityLocation & location = remote_cell_entities[slot][e];
          if(location.sm_rank == numbers::invalid_unsigned_int)
          {
            if(remote_cell_global_indices[slot][e] != numbers::invalid_dof_index)
              ++n_wrong;
          }
          else if(location.sm_rank >= shared_data.size() ||
                  location.offset >= shared_data[location.sm_rank].size() ||
                  static_cast<types::global_dof_index>(
                    shared_data[location.sm_rank][location.offset]) !=
                    remote_cell_global_indices[slot][e])
            ++n_wrong;
        }
    return n_wrong;
  }

  // Number of receive slots whose remote cell can be read in shared memory
  unsigned int
  n_slots_in_shared_memory() const
  {
    unsigned int count = 0;
    for(const bool in_shared_memory : slot_in_shared_memory)
      if(in_shared_memory)
        ++count;
    return count;
  }

  // Number of partners in the same shared-memory domain
  unsigned int
  n_shared_memory_partners() const
  {
    unsigned int count = 0;
    for(const Partner & partner : partners)
      if(partner.sm_rank != numbers::invalid_unsigned_int)
        ++count;
    return count;
  }

  ~Data()
  {
    free_communication();
  }

  // Global counts, summed over all processes
  struct Statistics
  {
    unsigned long long n_faces                          = 0;
    unsigned long long n_messages_old                   = 0;
    unsigned long long n_messages_new                   = 0;
    unsigned long long n_entries_old                    = 0;
    unsigned long long n_entries_new                    = 0;
    unsigned long long n_entries_supplements            = 0;
    unsigned long long local_supplements_opposite       = 0;
    unsigned long long local_supplements_perpendicular  = 0;
    unsigned long long remote_supplements_opposite      = 0;
    unsigned long long remote_supplements_perpendicular = 0;
  };

  void
  print_statistics(std::ostream & out) const
  {
    out << "Single-exchange setup for RT operator:" << std::endl
        << "  remote faces (each side counted):   " << statistics.n_faces << std::endl
        << "  entries per face:                   " << data_per_face << std::endl
        << "  messages sent old | new:            " << statistics.n_messages_old << " | "
        << statistics.n_messages_new << std::endl
        << "  entries sent old | new (G+F+S):     " << statistics.n_entries_old << " | "
        << statistics.n_entries_new << " (S: " << statistics.n_entries_supplements << ")"
        << std::endl
        << "  local supplements opposite | perp:  " << statistics.local_supplements_opposite
        << " | " << statistics.local_supplements_perpendicular << std::endl
        << "  remote supplements opposite | perp: " << statistics.remote_supplements_opposite
        << " | " << statistics.remote_supplements_perpendicular << std::endl;
  }

  // one message per pair of processes and direction in each vmult, so a
  // fixed tag is sufficient
  static constexpr int mpi_tag = 103;

  unsigned int n_locally_owned = 0;

  // layout of the face data, the same as in the existing face exchange: per
  // component the values followed by the normal derivatives, except for the
  // normal component, of which only the normal derivative is stored, starting
  // at entry normal_derivative_offset[face_direction]
  unsigned int                  data_per_face = 0;
  unsigned int                  dofs_per_face = 0;
  std::array<unsigned int, dim> normal_derivative_offset;

  std::vector<Partner> partners;
  unsigned int         send_buffer_size = 0;
  unsigned int         recv_buffer_size = 0;

  // per receive slot: position of the face data in the receive buffer, and
  // the face number of the shared face on the sending cell
  std::vector<unsigned int>  recv_face_offset;
  std::vector<unsigned char> recv_sender_face;

  std::vector<LocalSupplement> local_supplements;

  // per cell batch, face and lane: pointer to the face data of the
  // neighbor on a remote process in the layout described above, or nullptr
  // if the neighbor is local, see set_buffers()
  std::vector<dealii::ndarray<const Number *, 2 * dim, n_lanes>> remote_face_data;

  // per cell batch, face and lane: locations of the entities of the
  // neighbor on a remote process in the same shared-memory domain, which is
  // read directly from the source vector, or nullptr; a lane has either
  // remote_face_data or remote_cell_locations set if the neighbor is remote
  std::vector<dealii::ndarray<const std::array<EntityLocation, 2 * dim + 1> *, 2 * dim, n_lanes>>
    remote_cell_locations;

  Statistics statistics;

private:
  // per send face: global index of the first DoF and owner of the entities
  // of the sending cell, and the first locally owned global index
  std::vector<std::array<std::array<types::global_dof_index, 2>, 2 * dim + 1>> send_cell_entities;
  types::global_dof_index first_owned_index = 0;

  // per receive slot: global indices of the entities of the remote cell, for
  // count_wrong_remote_cell_entities()
  std::vector<std::array<types::global_dof_index, 2 * dim + 1>> remote_cell_global_indices;

  // Send the entities of the cells next to partners in the same
  // shared-memory domain to those partners, and translate the received
  // entities to locations in the shared-memory domain
  void
  setup_remote_cell_entities(const MPI_Comm comm, const MPI_Comm comm_sm)
  {
    const unsigned int n_slots = recv_face_offset.size();
    slot_in_shared_memory.assign(n_slots, false);
    remote_cell_entities.assign(n_slots, std::array<EntityLocation, 2 * dim + 1>());
    remote_cell_global_indices.assign(n_slots, std::array<types::global_dof_index, 2 * dim + 1>());

    // ranks of all processes in the shared-memory communicator, and the
    // locally owned ranges of the processes in the shared-memory domain
    const unsigned int n_procs = Utilities::MPI::n_mpi_processes(comm);
    std::vector<int>   ranks(n_procs), ranks_sm(n_procs);
    for(unsigned int p = 0; p < n_procs; ++p)
      ranks[p] = p;
    {
      MPI_Group group, group_sm;
      int       ierr = MPI_Comm_group(comm, &group);
      AssertThrowMPI(ierr);
      ierr = MPI_Comm_group(comm_sm, &group_sm);
      AssertThrowMPI(ierr);
      ierr = MPI_Group_translate_ranks(group, n_procs, ranks.data(), group_sm, ranks_sm.data());
      AssertThrowMPI(ierr);
      MPI_Group_free(&group);
      MPI_Group_free(&group_sm);
    }
    const std::vector<types::global_dof_index> first_owned_sm =
      Utilities::MPI::all_gather(comm_sm, first_owned_index);
    const std::vector<unsigned int> n_owned_sm =
      Utilities::MPI::all_gather(comm_sm, n_locally_owned);

    // per face: global index and owner of each entity
    constexpr unsigned int   entries_per_face = 2 * (2 * dim + 1);
    std::vector<MPI_Request> requests;
    requests.reserve(2 * partners.size());
    std::vector<std::vector<types::global_dof_index>> send_data(partners.size()),
      recv_data_entities(partners.size());
    for(unsigned int p = 0; p < partners.size(); ++p)
    {
      const Partner & partner = partners[p];
      if(partner.sm_rank == numbers::invalid_unsigned_int || partner.n_faces == 0)
        continue;
      for(unsigned int c = partner.first_face; c < partner.first_face + partner.n_faces; ++c)
        for(const auto & [index, owner] : send_cell_entities[c])
        {
          send_data[p].push_back(index);
          send_data[p].push_back(owner);
        }
      recv_data_entities[p].resize(partner.n_faces * entries_per_face);
      requests.emplace_back();
      int ierr = MPI_Irecv(recv_data_entities[p].data(),
                           recv_data_entities[p].size(),
                           Utilities::MPI::mpi_type_id_for_type<types::global_dof_index>,
                           partner.rank,
                           mpi_tag + 2,
                           comm,
                           &requests.back());
      AssertThrowMPI(ierr);
      requests.emplace_back();
      ierr = MPI_Isend(send_data[p].data(),
                       send_data[p].size(),
                       Utilities::MPI::mpi_type_id_for_type<types::global_dof_index>,
                       partner.rank,
                       mpi_tag + 2,
                       comm,
                       &requests.back());
      AssertThrowMPI(ierr);
    }
    const int ierr = MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    AssertThrowMPI(ierr);

    // Send side: the partner reads a cell directly if all its entities are
    // owned by processes in the shared-memory domain, the same criterion as
    // used on the receive side below
    for(Partner & partner : partners)
    {
      if(partner.sm_rank == numbers::invalid_unsigned_int || partner.n_faces == 0)
        continue;
      partner.faces_read_directly = true;
      for(unsigned int c = partner.first_face; c < partner.first_face + partner.n_faces; ++c)
        for(const auto & [index, owner] : send_cell_entities[c])
          if(index != numbers::invalid_dof_index && ranks_sm[owner] == MPI_UNDEFINED)
            partner.faces_read_directly = false;
    }

    for(unsigned int p = 0; p < partners.size(); ++p)
    {
      const Partner & partner = partners[p];
      if(partner.sm_rank == numbers::invalid_unsigned_int)
        continue;
      for(unsigned int f = 0; f < partner.n_faces; ++f)
      {
        const unsigned int slot     = partner.first_face + f;
        bool               on_node  = true;
        const auto *       entities = recv_data_entities[p].data() + f * entries_per_face;
        for(unsigned int e = 0; e < 2 * dim + 1; ++e)
        {
          const types::global_dof_index index = entities[2 * e];
          remote_cell_global_indices[slot][e] = index;
          if(index == numbers::invalid_dof_index)
            continue;
          const int owner_sm = ranks_sm[entities[2 * e + 1]];
          if(owner_sm == MPI_UNDEFINED)
          {
            on_node = false;
            break;
          }
          AssertThrow(index >= first_owned_sm[owner_sm] &&
                        index < first_owned_sm[owner_sm] + n_owned_sm[owner_sm],
                      ExcMessage("Entity not in the locally owned range of its owner"));
          remote_cell_entities[slot][e] =
            EntityLocation{static_cast<unsigned int>(owner_sm),
                           static_cast<unsigned int>(index - first_owned_sm[owner_sm])};
        }
        slot_in_shared_memory[slot] = on_node;
        if(!on_node)
          remote_cell_entities[slot] = std::array<EntityLocation, 2 * dim + 1>();
      }
    }
  }

  // per receive slot: cell batch, face and lane of the receiving cell
  unsigned int                             n_cell_batches = 0;
  std::vector<std::array<unsigned int, 3>> slot_location;

  // per partner: start of the received message, in the receive buffer or
  // in the send buffer of the partner in shared memory
  std::vector<const Number *> recv_data;

  // face data in shared memory to be copied to the given offset in the
  // receive buffer before adding supplements
  std::vector<std::pair<const Number *, unsigned int>> copy_to_recv_buffer;

  // signals that messages in shared memory have been read, see
  // signal_done_reading(): the first half holds the sends to the partners
  // this process reads from, the second half the receives from the partners
  // reading from this process
  MPI_Comm                         comm_done = MPI_COMM_NULL;
  mutable std::vector<MPI_Request> done_requests;
  mutable bool                     exchange_pending = false;

  void
  free_communication()
  {
    if(!done_requests.empty())
    {
      const int ierr = MPI_Waitall(done_requests.size(), done_requests.data(), MPI_STATUSES_IGNORE);
      AssertThrowMPI(ierr);
      done_requests.clear();
    }
    if(comm_done != MPI_COMM_NULL)
    {
      const int ierr = MPI_Comm_free(&comm_done);
      AssertThrowMPI(ierr);
      comm_done = MPI_COMM_NULL;
    }
  }

  // 1D shape functions of the normal (index 0, n_t + 1 polynomials) and
  // tangential (index 1, n_t polynomials) direction, evaluated on the two
  // sides of the reference interval, copied from
  // shape_info.data[0/1].shape_data_on_face: values of all polynomials,
  // followed by the derivatives
  unsigned int                                        degree_normal = 0;
  std::array<std::array<AlignedVector<Number>, 2>, 2> shape_data_on_face;

  // Placement of the contribution of the DoFs on face 'entity' to the face
  // data evaluated on face 'face', for entity perpendicular to face: the
  // DoFs are interpolated along the normal direction of 'face' (stride
  // 'stride_in' in the DoFs of 'entity') to the edge shared by the two faces
  // and written to the endpoint of the component normal to 'entity'; the
  // remaining direction along the edge has stride 'stride_free_in' in the
  // DoFs and 'stride_free_out' in the face data
  struct EdgeLayout
  {
    unsigned int stride_in        = 0;
    unsigned int stride_free_in   = 0;
    unsigned int out_offset       = 0;
    unsigned int stride_free_out  = 0;
    unsigned int derivative_shift = 0;
  };
  std::array<std::array<EdgeLayout, 2 * dim>, 2 * dim> edge_layouts;

  template<typename Function>
  DEAL_II_ALWAYS_INLINE void
  dispatch_degree(const Function & function) const
  {
    switch(degree_normal)
    {
      case 2:
        function(std::integral_constant<int, 2>());
        break;
      case 3:
        function(std::integral_constant<int, 3>());
        break;
      case 4:
        function(std::integral_constant<int, 4>());
        break;
      case 5:
        function(std::integral_constant<int, 5>());
        break;
      case 6:
        function(std::integral_constant<int, 6>());
        break;
      case 7:
        function(std::integral_constant<int, 7>());
        break;
      case 8:
        function(std::integral_constant<int, 8>());
        break;
      case 9:
        function(std::integral_constant<int, 9>());
        break;
      default:
        AssertThrow(false, ExcNotImplemented());
    }
  }

  // Add the contribution of the DoFs 'u' on the face 'entity' opposite to
  // 'face' to the normal derivative of the normal component in 'face_data'
  template<int n_t>
  void
  add_opposite_face_derivative(const unsigned int face,
                               const unsigned int entity,
                               const Number *     u,
                               Number *           face_data) const
  {
    constexpr unsigned int n_n           = n_t + 1;
    constexpr unsigned int dofs_per_face = Utilities::pow(n_t, dim - 1);
    const Number           deriv_factor = shape_data_on_face[0][face % 2][n_n + (entity % 2) * n_t];
    Number *               derivatives  = face_data + normal_derivative_offset[face / 2];
    for(unsigned int i = 0; i < dofs_per_face; ++i)
      derivatives[i] += deriv_factor * u[i];
  }

  // Interpolate the DoFs 'u' on face 'entity' perpendicular to 'face' to the
  // shared edge: values in trace[0, n_free), normal derivatives with respect
  // to 'face' in trace[n_free, 2 n_free)
  template<int n_t>
  void
  compute_edge_trace(const unsigned int face,
                     const unsigned int entity,
                     const Number *     u,
                     Number *           trace) const
  {
    constexpr unsigned int          n_free  = Utilities::pow(n_t, dim - 2);
    const EdgeLayout &              layout  = edge_layouts[face][entity];
    const Number * DEAL_II_RESTRICT shape_t = shape_data_on_face[1][face % 2].data();
    const std::size_t               offset  = layout.stride_free_in;
    const std::size_t               stride  = layout.stride_in;
    for(unsigned int i2 = 0; i2 < n_free; ++i2)
    {
      const Number * u_line     = u + i2 * offset;
      Number         value      = shape_t[0] * u_line[0];
      Number         derivative = shape_t[n_t] * u_line[0];
      for(unsigned int i = 1; i < n_t; ++i)
      {
        value += shape_t[i] * u_line[i * stride];
        derivative += shape_t[n_t + i] * u_line[i * stride];
      }
      trace[i2]          = value;
      trace[n_free + i2] = derivative;
    }
  }

  // Add an edge trace computed by compute_edge_trace() to the face data
  template<int n_t>
  void
  add_edge_trace(const unsigned int face,
                 const unsigned int entity,
                 const Number *     trace,
                 Number *           face_data) const
  {
    constexpr unsigned int n_free  = Utilities::pow(n_t, dim - 2);
    const EdgeLayout &     layout  = edge_layouts[face][entity];
    Number *               out     = face_data + layout.out_offset;
    Number *               out_der = face_data + layout.out_offset + layout.derivative_shift;
    const std::size_t      stride  = layout.stride_free_out;
    for(unsigned int i2 = 0; i2 < n_free; ++i2)
      out[i2 * stride] += trace[i2];
    for(unsigned int i2 = 0; i2 < n_free; ++i2)
      out_der[i2 * stride] += trace[n_free + i2];
  }

  // Contribution of the DoFs 'u' on face 'entity' to the face data on 'face'
  // in section S
  template<int n_t>
  void
  compute_supplement(const unsigned int face,
                     const unsigned int entity,
                     const Number *     u,
                     Number *           out) const
  {
    if(entity / 2 == face / 2)
    {
      // only the normal derivative is nonzero for Gauss-Lobatto points
      constexpr unsigned int n_n           = n_t + 1;
      constexpr unsigned int dofs_per_face = Utilities::pow(n_t, dim - 1);
      const Number deriv_factor = shape_data_on_face[0][face % 2][n_n + (entity % 2) * n_t];
      for(unsigned int i = 0; i < dofs_per_face; ++i)
        out[i] = deriv_factor * u[i];
    }
    else
      compute_edge_trace<n_t>(face, entity, u, out);
  }

  // Add a supplement in the format of compute_supplement() to the face data
  template<int n_t>
  void
  add_supplement(const unsigned int face,
                 const unsigned int entity,
                 const Number *     in,
                 Number *           face_data) const
  {
    if(entity / 2 == face / 2)
    {
      constexpr unsigned int dofs_per_face = Utilities::pow(n_t, dim - 1);
      Number *               derivatives   = face_data + normal_derivative_offset[face / 2];
      for(unsigned int i = 0; i < dofs_per_face; ++i)
        derivatives[i] += in[i];
    }
    else
      add_edge_trace<n_t>(face, entity, in, face_data);
  }

  template<int n_t>
  static constexpr unsigned int
  supplement_size(const unsigned int face, const unsigned int entity)
  {
    return (entity / 2 == face / 2) ? Utilities::pow(n_t, dim - 1) :
                                      2 * Utilities::pow(n_t, dim - 2);
  }

  unsigned int
  supplement_size(const unsigned int n_t, const unsigned int face, const unsigned int entity)
  {
    return (entity / 2 == face / 2) ? Utilities::pow(n_t, dim - 1) :
                                      2 * Utilities::pow(n_t, dim - 2);
  }

  template<int n_t>
  void
  pack_supplement_section_tensor(const Partner & partner,
                                 const Number *  src,
                                 Number *        send_buffer) const
  {
    Number * out = face_section(partner, send_buffer) + partner.n_faces * data_per_face;
    for(const SupplementSend & s : partner.supplements_send)
    {
      compute_supplement<n_t>(s.face, s.entity, src + s.local_index, out);
      out += supplement_size<n_t>(s.face, s.entity);
    }
    AssertDimension(out - send_buffer, partner.send_offset + partner.send_size);
  }

  template<int n_t>
  void
  unpack_and_complete_tensor(Number * recv_buffer, Number * vector_begin) const
  {
    if(!copy_to_recv_buffer.empty() || !done_requests.empty())
      std::atomic_thread_fence(std::memory_order_acquire);
    unpack_ghost_section(vector_begin);

    for(const auto & [data, offset] : copy_to_recv_buffer)
      std::copy(data, data + data_per_face, recv_buffer + offset);

    for(unsigned int p = 0; p < partners.size(); ++p)
    {
      const Partner & partner = partners[p];
      const Number *  in = recv_data[p] + partner.n_ghost_recv + partner.n_faces * data_per_face;
      for(const SupplementRecv & s : partner.supplements_recv)
      {
        const unsigned int face = recv_sender_face[s.slot];
        add_supplement<n_t>(face, s.entity, in, recv_buffer + recv_face_offset[s.slot]);
        in += supplement_size<n_t>(face, s.entity);
      }
      AssertDimension(in - recv_data[p], partner.recv_size);
    }

    constexpr unsigned int n_free = Utilities::pow(n_t, dim - 2);
    for(const LocalSupplement & s : local_supplements)
    {
      const unsigned int face = recv_sender_face[s.slot];
      Number *           data = recv_buffer + recv_face_offset[s.slot];
      if(s.sender_face / 2 == face / 2)
        add_opposite_face_derivative<n_t>(face, s.sender_face, vector_begin + s.local_index, data);
      else
      {
        Number trace[2 * n_free];
        compute_edge_trace<n_t>(face, s.sender_face, vector_begin + s.local_index, trace);
        add_edge_trace<n_t>(face, s.sender_face, trace, data);
      }
    }
  }

  // Write section G into the ghost range of the vector
  void
  unpack_ghost_section(Number * vector_begin) const
  {
    for(unsigned int p = 0; p < partners.size(); ++p)
    {
      const Partner & partner = partners[p];
      const Number *  in      = recv_data[p];
      std::copy(in,
                in + partner.n_ghost_recv,
                vector_begin + n_locally_owned + partner.ghost_recv_start);
    }
  }

  // Copy the 1D shape data, compute the edge layouts and check the
  // tensor-product kernels against the face matrices for random input
  void
  setup_tensor_product_kernels(const internal::MatrixFreeFunctions::ShapeInfo<Number> & shape_info)
  {
    degree_normal = shape_info.data[0].fe_degree;
    for(unsigned int d = 0; d < 2; ++d)
      for(unsigned int side = 0; side < 2; ++side)
        shape_data_on_face[d][side] = shape_info.data[d].shape_data_on_face[side];

    const unsigned int n_t    = degree_normal;
    const unsigned int n_n    = n_t + 1;
    const unsigned int n_free = Utilities::pow(n_t, dim - 2);
    for(unsigned int face = 0; face < 2 * dim; ++face)
      for(unsigned int entity = 0; entity < 2 * dim; ++entity)
      {
        const unsigned int d0 = face / 2, de = entity / 2;
        if(d0 == de)
          continue;
        // direction along the shared edge (in 2D, there is none, so put it
        // last such that the strides of the other directions are 1)
        const unsigned int d_free = dim == 3 ? 3 - d0 - de : dim;
        EdgeLayout &       layout = edge_layouts[face][entity];
        layout.stride_in          = d0 < d_free ? 1 : n_t;
        layout.stride_free_in     = d0 < d_free ? n_t : 1;
        unsigned int block_offset = 0;
        for(unsigned int c = 0; c < de; ++c)
          block_offset += c == d0 ? dofs_per_face : 2 * n_n * n_free;
        layout.out_offset       = block_offset + (entity % 2) * (de < d_free ? n_t : n_t * n_free);
        layout.stride_free_out  = de < d_free ? n_n : 1;
        layout.derivative_shift = n_n * n_free;
      }
  }

  static std::pair<typename Triangulation<dim>::cell_iterator, unsigned int>
  get_neighbor(const typename Triangulation<dim>::cell_iterator & cell, const unsigned int face)
  {
    if(cell->at_boundary(face))
    {
      AssertThrow(cell->has_periodic_neighbor(face), ExcInternalError());
      return {cell->periodic_neighbor(face), cell->periodic_neighbor_face_no(face)};
    }
    else
      return {cell->neighbor(face), cell->neighbor_face_no(face)};
  }

  // The face data is packed and unpacked in the coordinate system of the
  // cells, so we need conforming neighbors with standard orientation
  static void
  check_face_pair(const Triangulation<dim> &                         tria,
                  const typename Triangulation<dim>::cell_iterator & cell,
                  const unsigned int                                 face,
                  const typename Triangulation<dim>::cell_iterator & neighbor,
                  const unsigned int                                 neighbor_face,
                  const unsigned int                                 neighbor_rank)
  {
    AssertThrow(!neighbor->has_children() && neighbor->level() == cell->level(),
                ExcMessage("Hanging nodes are not supported by the RT operator"));
    AssertThrow(neighbor->subdomain_id() == neighbor_rank, ExcInternalError());
    AssertThrow(neighbor_face == (face ^ 1),
                ExcMessage("Expected neighbor across face " + std::to_string(face) +
                           " to have face number " + std::to_string(face ^ 1) + ", got " +
                           std::to_string(neighbor_face)));
    if(cell->at_boundary(face))
    {
      const auto & periodic_face_map = tria.get_periodic_face_map();
      const auto   it                = periodic_face_map.find({cell, face});
      AssertThrow(it != periodic_face_map.end(), ExcInternalError());
      AssertThrow(it->second.second == numbers::default_geometric_orientation,
                  ExcMessage("Periodic faces need to be in standard orientation"));
    }
    else
      AssertThrow(cell->combined_face_orientation(face) == numbers::default_geometric_orientation &&
                    neighbor->combined_face_orientation(neighbor_face) ==
                      numbers::default_geometric_orientation,
                  ExcMessage("Faces need to be in standard orientation"));
  }

  void
  compute_statistics(const Utilities::MPI::Partitioner &   partitioner,
                     const std::vector<RemoteSupplement> & remote_supplements,
                     const MPI_Comm                        comm)
  {
    std::array<unsigned long long, 10> local = {};
    local[0]                                 = recv_face_offset.size();
    local[1]                                 = partitioner.import_targets().size();
    for(const Partner & partner : partners)
    {
      local[1] += partner.n_faces > 0 ? 1 : 0;
      local[2] += partner.send_size > 0 ? 1 : 0;
      local[5] += partner.n_supplement_send_entries;
    }
    local[3] = partitioner.n_import_indices() +
               static_cast<unsigned long long>(recv_face_offset.size()) * data_per_face;
    local[4] = send_buffer_size;
    for(const LocalSupplement & s : local_supplements)
      ++local[s.sender_face / 2 == recv_sender_face[s.slot] / 2 ? 6 : 7];
    for(const RemoteSupplement & s : remote_supplements)
      ++local[s.sender_face / 2 == recv_sender_face[s.slot] / 2 ? 8 : 9];

    std::array<unsigned long long, 10> global;
    const int                          ierr = MPI_Allreduce(
      local.data(), global.data(), local.size(), MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
    AssertThrowMPI(ierr);

    statistics.n_faces                          = global[0];
    statistics.n_messages_old                   = global[1];
    statistics.n_messages_new                   = global[2];
    statistics.n_entries_old                    = global[3];
    statistics.n_entries_new                    = global[4];
    statistics.n_entries_supplements            = global[5];
    statistics.local_supplements_opposite       = global[6];
    statistics.local_supplements_perpendicular  = global[7];
    statistics.remote_supplements_opposite      = global[8];
    statistics.remote_supplements_perpendicular = global[9];
  }
};

} // namespace SingleExchange
} // namespace RTOperator
