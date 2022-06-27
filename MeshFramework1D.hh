#pragma once
#include "MeshFramework.hh"
#include "MeshAlgorithmsDefault_decl.hh"
#include "MeshAlgorithmsDefault_impl.hh"

struct MeshFramework1D : public MeshFramework {

  MeshFramework1D(std::size_t size)
    : MeshFramework(), mesh_size_(size) {
    // Init singleton
    algorithms_ = new MeshAlgorithmsDefault<host>();
  }

  std::size_t getNumNodes() const override { return mesh_size_; }
  std::size_t getNumCells() const override { return mesh_size_ - 1; }

  // cell-to-node connections
  std::size_t getCellNumNodes(Entity_ID c) const override { return 2; }
  Kokkos::View<const Entity_ID*,Kokkos::HostSpace> getCellNodes(const Entity_ID id) const override {
    // 2 Elements per entries
    Kokkos::View<Entity_ID*, Kokkos::HostSpace> tmp("",2);
    for(int i = 0 ; i < 2; ++i) tmp(i) = id + i;
    return tmp;
  }
  Entity_ID getCellNode(Entity_ID c, std::size_t i) const override { return c + i; }

  // node coordinates -- must be cached
  Coordinate getNodeCoordinate(Entity_ID n) const override {
    double x = 0;
    for (int i = 1; i != n+1; ++i) x += i;
    return {x,0,0};
  }

 public:
  int mesh_size_;
};










