#pragma once

#include "MeshAlgorithms_decl.hh"

namespace Impl {

template<typename Mesh_type>
Kokkos::View<Coordinate*, typename Space<Mesh_type::mem>::execution_space>
computeCellCentroids(Mesh_type& m) {
  Kokkos::View<Coordinate*, typename Space<Mesh_type::mem>::execution_space> ret;
  Kokkos::resize(ret, m.getNumCells());
  if constexpr(Mesh_type::mem == device) {
    Kokkos::parallel_for(
      "",m.getNumCells(),KOKKOS_LAMBDA(const int c){
      auto nnodes = m.getCellNumNodes(c);
      ret[c] = {0,0,0};
      for (int i=0; i!=nnodes; ++i) {
        auto nc = m.getNodeCoordinate(m.getCellNode(c,i));
        ret[c][0] += nc[0];
        ret[c][1] += nc[1];
        ret[c][2] += nc[2];
      }
      ret[c][0] /= nnodes;
      ret[c][1] /= nnodes;
      ret[c][2] /= nnodes;
    });
  } else {
    for (Entity_ID c=0; c!=m.getNumCells(); ++c) {
      auto nnodes = m.getCellNumNodes(c);
      ret[c] = {0,0,0};
      for (int i=0; i!=nnodes; ++i) {
        auto nc = m.getNodeCoordinate(m.getCellNode(c,i));
        ret[c][0] += nc[0];
        ret[c][1] += nc[1];
        ret[c][2] += nc[2];
      }
      ret[c][0] /= nnodes;
      ret[c][1] /= nnodes;
      ret[c][2] /= nnodes;
    }
  }
  return ret;
};


template<typename Mesh_type>
Kokkos::View<double*, typename Space<Mesh_type::mem>::execution_space>
computeCellVolumes(Mesh_type& m) {
  Kokkos::View<double*, typename Space<Mesh_type::mem>::execution_space> ret;
  Kokkos::resize(ret, m.getNumCells());

  if constexpr(Mesh_type::mem == device) {
    Kokkos::parallel_for(
      "",m.getNumCells(),KOKKOS_LAMBDA(const int c){
      auto nnodes = m.getCellNumNodes(c);
      assert(nnodes == 2);
      ret[c] = m.getNodeCoordinate(m.getCellNode(c,1))[0] - m.getNodeCoordinate(m.getCellNode(c,0))[0];
    });
  } else {
    for (Entity_ID c=0; c!=m.getNumCells(); ++c) {
      auto nnodes = m.getCellNumNodes(c);
      assert(nnodes == 2);
      ret[c] = m.getNodeCoordinate(m.getCellNode(c,1))[0] - m.getNodeCoordinate(m.getCellNode(c,0))[0];
    }
  }
  return ret;
};

} // namespace Impl
