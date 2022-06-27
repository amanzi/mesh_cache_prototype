#pragma once

#include "MeshAlgorithms_decl.hh"
#include "MeshAlgorithmsDefault_decl.hh"
#include "MeshAlgorithms_impl.hh"
#include "MeshCache_decl.hh"

// MeshAlgorithms derived class
template<memory MEM>
Kokkos::View<double*, typename Space<MEM>::execution_space>
MeshAlgorithmsDefault<MEM>::computeCellVolumes(const MeshCache<MEM>& m) const {
  return Impl::computeCellVolumes(m);
}

template<memory MEM>
Kokkos::View<double*, typename Space<MEM>::execution_space>
MeshAlgorithmsDefault<MEM>::computeCellVolumes(const MeshFramework& m) const {
  return Impl::computeCellVolumes(m);
}

template<memory MEM>
Kokkos::View<Coordinate*, typename Space<MEM>::execution_space>
MeshAlgorithmsDefault<MEM>::computeCellCentroids(const MeshCache<MEM>& m) const {
  return Impl::computeCellCentroids(m);
}

template<memory MEM>
Kokkos::View<Coordinate*, typename Space<MEM>::execution_space>
MeshAlgorithmsDefault<MEM>::computeCellCentroids(const MeshFramework& m) const {
  return Impl::computeCellCentroids(m);
}

template<memory MEM>
MeshAlgorithms<device>*
MeshAlgorithmsDefault<MEM>::create_device() {
  return new MeshAlgorithmsDefault<device>();
}

template<memory MEM>
MeshAlgorithms<host>*
MeshAlgorithmsDefault<MEM>::create_host() {
  return new MeshAlgorithmsDefault<host>();
}
