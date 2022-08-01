template<typename Data, typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
struct CSR;


namespace Algorithms {

// The algorithms namespace provides some basic algorithms.  Note that both the
// MeshFramework and MeshCache classes may use these algorithms.
template<
  typename Mesh_type,
  typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
void computeCellGeometry(Mesh_type& m) {
  if constexpr(ExecutionSpace == HostSpace) {
    for (Entity_ID c=0; c!=m.num_cells; ++c) {
      auto nnodes = m.getCellNumNodes(c);
      m.cell_centroids[c] = {0,0,0};
      for (int i=0; i!=nnodes; ++i) {
        auto nc = m.getNodeCoordinate(m.getCellNodes(c,i));
        m.cell_centroids[c][0] += nc[0];
        m.cell_centroids[c][1] += nc[1];
        m.cell_centroids[c][2] += nc[2];
      }
      m.cell_centroids[c][0] /= nnodes;
      m.cell_centroids[c][1] /= nnodes;
      m.cell_centroids[c][2] /= nnodes;
    }
  } else {
    // parallel for....
  }
}

}


// the MeshAlgorithms struct includes virtual methods for doing the work.  Note
// these must be virtual, because different MeshFrameworks will supply
// different implementations of these.  The default just uses the above
// algorithm, but other MeshFramework instances use other algorithms.
//
// Note, I hope this works -- I've only compiled a version of this without an
// ExecutionSpace.
template<typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
struct MeshAlgorithms {
  virtual void computeCellGeometry(MeshCache<ExecutionSpace>& m) = 0;
  virtual void computeCellGeometry(MeshFramework& m) = 0;
}

template<typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
struct MeshAlgorithmsDefault : MeshAlgorithms<ExecutionSpace> {
  virtual void computeCellGeometry(MeshCache<ExecutionSpace> m) {
    Algorithms::computeCellGeometry(m);
  }
  virtual void computeCellGeometry(MeshFramework& m) {
    Algorithms::computeCellGeometry<MeshFramework, Kokkos::HostExecutionSpace>(m);
  }
}


//
// Note, this API is consistent with amanzi/ecoon/mesh_refactor src/mesh/MeshCache.hh API.
//
// If it helped, these could be dual views?
//
template<typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
struct MeshCache {
  using Coordinate = std::array<double,3>;
  using Coordinate_View = Kokkos::View<Coordinate*, ExecutionSpace>;
  using Double_View = Kokkos::View<double*, ExecutionSpace>; // ?????
  using cDouble_View = Kokkos::View<const double*, ExecutionSpace>; // ?????
  using Entity_ID_View = Kokkos::View<Entity_ID*, ExecutionSpace>;
  using cEntity_ID_View = Kokkos::View<const Entity_ID*, ExecutionSpace>;

  MeshCache(Teuchos::RCP<MeshFramework>& mf) {
    // do whatever!
    // make sure to keep the algorithms struct for future use!
    algorithms_ = mf->getAlgorithms();
  }

  // typical adjacency information
  std::size_t getCellNumNodes(const Entity_ID c) const;
  cEntity_ID_View getCellNodes(const Entity_ID c) const;
  Entity_ID getCellNode(Entity_ID c, std::size_t i) const;
  CRS<Entity_ID, ExecutionSpace> cell_nodes;

  // typical coordinate information
  Coordinate getNodeCoordinate(const Entity_ID n) const;
  Coordinate_View node_coordinates;

  // cell centroids may be cached on the fly, then deleted
  // cached version
  bool cell_geometry_cached;
  void cacheCellGeometry() {
    if (cell_geometry_cached) return;
    cell_centroids.resize(ncells_all);
    algorithms_->computeCellGeometry(*this);
    cell_geometry_cached = true;
  }
  void clearCellGeometry() {
    cell_centroids.clear(); // not sure this is valid...
    cell_geometry_cached = false;
  }

  Coordinate getCellCentroid(const Entity_ID c) const;
  Coordinate_View cell_centroids;
}
