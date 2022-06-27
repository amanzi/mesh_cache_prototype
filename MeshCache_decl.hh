#pragma once
#include "MeshDefs.hh"
#include "MeshAlgorithms_decl.hh"

// utility class for data
struct MeshCacheData {
  Kokkos::DualView<Coordinate*,Kokkos::DefaultExecutionSpace> node_coordinates;
  DualCrs<Entity_ID*> cell_nodes;
  Kokkos::DualView<Coordinate*,Kokkos::DefaultExecutionSpace> cell_centroids;
  Kokkos::DualView<double*,Kokkos::DefaultExecutionSpace> cell_volumes;
};



// MeshCache templated on the memory type
// Defaulted to the device memory space
template<memory MEM = device>
struct MeshCache {
  static constexpr memory mem = MEM;

  // Building a Mesh Cache on ExecutionSpace or Host Memory from another cache.
  template<memory M>
  MeshCache(const MeshCache<M>& mc);

  // Building a Mesh Cache on ExecutionSpace memory space.
  // Retrieve/compute data on the Host from the Mesh Framework then copy to device
  // NOTE: limit this to only on DEVICE?  Or only on HOST? Or does it not matter? --etc
  MeshCache(MeshFramework* mf);


  Entity_ID getNumCells() const { return num_cells; }
  Entity_ID getNumNodes() const { return num_nodes; }


  KOKKOS_INLINE_FUNCTION Coordinate& getCellCentroid(const Entity_ID c) const {
    return view<MEM>(mcd->cell_centroids)(c);
  }

  KOKKOS_INLINE_FUNCTION auto getCellCentroids() const {
    return view<MEM>(mcd->cell_centroids);
  }

  KOKKOS_INLINE_FUNCTION std::size_t getCellNumNodes(const Entity_ID c) const {
    return view<MEM>(mcd->cell_nodes.row_map)(c+1) - view<MEM>(mcd->cell_nodes.row_map)(c);
  }

  // This function can just be overloaded with the Kokkos::View type.
  KOKKOS_INLINE_FUNCTION auto getCellNodes(Entity_ID id) const {
    auto begin = view<MEM>(mcd->cell_nodes.row_map)(id);
    auto end = view<MEM>(mcd->cell_nodes.row_map)(id+1);
    return Kokkos::subview(view<MEM>(mcd->cell_nodes.entries),
                           Kokkos::make_pair(begin, end));
  }

  KOKKOS_INLINE_FUNCTION Entity_ID getCellNode(Entity_ID c, int i) const {
   return view<MEM>(mcd->cell_nodes.entries)(view<MEM>(mcd->cell_nodes.row_map)(c)+i);
  }


  KOKKOS_INLINE_FUNCTION Coordinate getNodeCoordinate(const Entity_ID n) const {
   return view<MEM>(mcd->node_coordinates)(n);
  }


  KOKKOS_INLINE_FUNCTION double getCellVolume(const Entity_ID c) const {
   return view<MEM>(mcd->cell_volumes)(c);
  }


  MeshAlgorithms<MEM>* algorithms_;
  std::shared_ptr<MeshCacheData> mcd;
  MeshFramework* framework_mesh_;
  int num_cells, num_nodes;
};

