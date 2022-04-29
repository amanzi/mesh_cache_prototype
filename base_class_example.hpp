template<typename Data, typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
struct CSR;


//
// Note, this API is consistent with amanzi/ecoon/mesh_refactor src/mesh/MeshCache.hh API.
//
// If it helped, these could be dual views?
//
template<typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
struct MeshCache {

  using Double_View = Kokkos::View<double*, ExecutionSpace>; // ?????
  using cDouble_View = Kokkos::View<const double*, ExecutionSpace>; // ?????
  using Entity_ID_View = Kokkos::View<Entity_ID*, ExecutionSpace>;
  using cEntity_ID_View = Kokkos::View<const Entity_ID*, ExecutionSpace>;

  MeshCache(Teuchos::RCP<MeshFramework>& mf); // note, feel free to make shared_ptr or raw ptr for now...

  double getCellVolume(const Entity_ID c) const;
  Double_View cell_volumes; // note, this is public and can be used inside kernels directly
  // is there a const problem with exposing this directly?  Specifically,
  // assume you have a const MeshCache.  Can you still change
  // mc.cell_volumes(3) = 1?  Should this be cDoubleView?

  std::size_t getCellNumFaces(const Entity_ID c) const;
  cEntity_ID_View getCellFaces(const Entity_ID c) const;
  Entity_ID getCellFace(Entity_ID c, std::size_t i) const;
  CRS<Entity_ID, ExecutionSpace> cell_faces;
}
