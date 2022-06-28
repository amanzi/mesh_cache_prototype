#pragma once

#include "MeshDefs.hh"

namespace Impl {

template<typename Mesh_type>
Kokkos::View<Coordinate*, typename Space<Mesh_type::mem>::execution_space>
computeCellCentroids(Mesh_type& m);


template<typename Mesh_type>
Kokkos::View<double*, typename Space<Mesh_type::mem>::execution_space>
computeCellVolumes(Mesh_type& m);


} // namespace Impl

template<memory MEM> struct MeshCache;
template<memory MEM> struct MeshCacheDumb;
struct MeshFramework;



// MeshAlgorithm generic interface
template<memory MEM>
struct MeshAlgorithms {
  virtual ~MeshAlgorithms() = default;
  virtual Kokkos::View<double*, typename Space<MEM>::execution_space>
    computeCellVolumes(const MeshCache<MEM>& m) const = 0;
  virtual Kokkos::View<double*, typename Space<MEM>::execution_space>
    computeCellVolumes(const MeshCacheDumb<MEM>& m) const = 0;
  virtual Kokkos::View<double*, typename Space<MEM>::execution_space>
    computeCellVolumes(const MeshFramework& m) const = 0;
  virtual Kokkos::View<Coordinate*, typename Space<MEM>::execution_space>
    computeCellCentroids(const MeshCache<MEM>& m) const = 0;
  virtual Kokkos::View<Coordinate*, typename Space<MEM>::execution_space>
    computeCellCentroids(const MeshCacheDumb<MEM>& m) const = 0;
  virtual Kokkos::View<Coordinate*, typename Space<MEM>::execution_space>
    computeCellCentroids(const MeshFramework& m) const = 0;
  virtual MeshAlgorithms<device>* create_device() = 0;
  virtual MeshAlgorithms<host>* create_host() = 0;

  template<memory M>
  MeshAlgorithms<M>* create_on() {
    if constexpr (M == MEM) return this;
    else if constexpr (M == host) return create_host();
    else return create_device();
  }
};

