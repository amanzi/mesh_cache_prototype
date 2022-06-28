#pragma once


#include "MeshAlgorithms_decl.hh"


// MeshAlgorithms derived class
template<memory MEM>
struct MeshAlgorithmsDefault: MeshAlgorithms<MEM>  {
  virtual Kokkos::View<double*, typename Space<MEM>::execution_space>
  computeCellVolumes(const MeshCache<MEM>& m) const override;

  virtual Kokkos::View<double*, typename Space<MEM>::execution_space>
  computeCellVolumes(const MeshCacheDumb<MEM>& m) const override;

  virtual Kokkos::View<double*, typename Space<MEM>::execution_space>
  computeCellVolumes(const MeshFramework& m) const override;

  virtual Kokkos::View<Coordinate*, typename Space<MEM>::execution_space>
  computeCellCentroids(const MeshCache<MEM>& m) const override;

  virtual Kokkos::View<Coordinate*, typename Space<MEM>::execution_space>
  computeCellCentroids(const MeshCacheDumb<MEM>& m) const override;

  virtual Kokkos::View<Coordinate*, typename Space<MEM>::execution_space>
  computeCellCentroids(const MeshFramework& m) const override;

  virtual MeshAlgorithms<device>* create_device() override;
  virtual MeshAlgorithms<host>* create_host() override;
};

