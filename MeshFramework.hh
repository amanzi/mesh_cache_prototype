#pragma once

#include "MeshAlgorithms_decl.hh"

// Base class for mesh frameworks
// the host only, slow class
struct MeshFramework {
  static constexpr memory mem = host;

  MeshFramework() {}

  virtual std::size_t getNumCells() const = 0;
  virtual std::size_t getNumNodes() const = 0;

  virtual Coordinate getNodeCoordinate(Entity_ID n) const = 0;
  virtual std::size_t getCellNumNodes(Entity_ID c) const = 0;
  virtual Kokkos::View<const Entity_ID*,Kokkos::HostSpace> getCellNodes(const Entity_ID id) const = 0;
  virtual Entity_ID getCellNode(Entity_ID c, std::size_t i) const = 0;

  Kokkos::View<double*, Kokkos::HostSpace>
  getCellVolumes() const { return algorithms_->computeCellVolumes(*this); }

  Kokkos::View<Coordinate*, Kokkos::HostSpace>
  getCellCentroids() const { return algorithms_->computeCellCentroids(*this); }

  MeshAlgorithms<host>* algorithms_;
};
