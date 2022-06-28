#pragma once
#include "MeshDefs.hh"
#include "MeshAlgorithms_decl.hh"


// MeshCache templated on the memory type
// Defaulted to the device memory space
template<memory MEM = device>
struct MeshCacheDumb {
  static constexpr memory mem = MEM;

  // Building a Mesh Cache on ExecutionSpace memory space.
  // Retrieve/compute data on the Host from the Mesh Framework then copy to device
  // NOTE: limit this to only on DEVICE?  Or only on HOST? Or does it not matter? --etc
  MeshCacheDumb(MeshFramework* mf);


  Entity_ID getNumCells() const { return num_cells; }
  Entity_ID getNumNodes() const { return num_nodes; }


  Coordinate& getCellCentroid(const Entity_ID c) const {
    return cell_centroids[c];
  }

  auto getCellCentroids() const {
    return cell_centroids;
  }

  std::size_t getCellNumNodes(const Entity_ID c) const {
    return cell_nodes[c].size();
  }

  // This function can just be overloaded with the Kokkos::View type.
  auto getCellNodes(Entity_ID id) const {
    return cell_nodes[id];
  }

  Entity_ID getCellNode(Entity_ID c, int i) const {
    return cell_nodes[c][i];
  }


  Coordinate getNodeCoordinate(const Entity_ID n) const {
    return node_coordinates[n];
  }


  double getCellVolume(const Entity_ID c) const {
   return cell_volumes[c];
  }


  MeshAlgorithms<MEM>* algorithms_;
  MeshFramework* framework_mesh_;
  int num_cells, num_nodes;

  std::vector<double> cell_volumes;
  std::vector<Coordinate> node_coordinates;
  std::vector<Coordinate> cell_centroids;
  std::vector<std::vector<Entity_ID>> cell_nodes;
};

