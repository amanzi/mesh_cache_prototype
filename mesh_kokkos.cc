#include "Kokkos_Core.hpp"
#include "Kokkos_DualView.hpp"

//
// Note, this API is consistent with amanzi/ecoon/mesh_refactor src/mesh/MeshFramework.hh API
//
struct MeshFramework {
  MeshFramework(const int size): mesh_size_(size) {}

  // In MeshFramework this function "compute" the result on Host
  int getCellVolume(const int i) const {
    return i;
  }

  // In MeshFramework this function "compute" the result on Host
  void getCellFaces(const int id, Kokkos::View<const int*,Kokkos::HostSpace>& ret) const {
    // 2 Elements per entries
    Kokkos::View<int*, Kokkos::HostSpace> tmp("",2);
    for(int i = 0 ; i < 2; ++i){
      tmp(i) = value*i;
    }
    ret = tmp;
  }

  int meshSize() const { return mesh_size_; }

private:
  const int value = 10;
  int mesh_size_;
};


// NOTE: need this...
template<typename Data, typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
struct CSR;


using Entity_ID = int;

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




// MeshCache templated on the memory type
// Defaulted to the device memory space
template<typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
struct MeshCache {

  // Type to get CRS on DualViews
  template<typename T, typename ExecutionSpace>
  struct DualCrs{
    Kokkos::DualView<int*, ExecutionSpace> row_map;
    Kokkos::DualView<T, ExecutionSpace> entries;
  };

  // Building a Mesh Cache on ExecutionSpace memory space.
  // Retrieve/compute data on the Host from the Mesh Framework then copy to device
  MeshCache(MeshFramework* mf): framework_mesh_(mf), mesh_size_(mf->meshSize()) {
    // Init data on Host
    Kokkos::resize(value_1d_,mesh_size_);
    Kokkos::resize(value_2d_.row_map,mesh_size_+1);
    Kokkos::resize(value_2d_.entries,mesh_size_*2);
    for(int i = 0 ; i < mesh_size_ ; ++i){
      value_1d_.view_host()(i) = mf->getCellVolume(i);
    }
    for(int i = 0 ; i < mesh_size_ ; ++i){
      value_2d_.row_map.view_host()(i) = i*2;
      Kokkos::View<const int*, Kokkos::HostSpace> tmp;
      mf->getCellFaces(i,tmp);
      for(int j = 0 ; j < 2; ++j){
        value_2d_.entries.view_host()(i*2+j) = tmp(j);
      }
    }
    value_2d_.row_map.view_host()(mesh_size_) = mesh_size_*2;
    // Copy to device memory
    Kokkos::deep_copy(value_1d_.view_device(),value_1d_.view_host());
    Kokkos::deep_copy(value_2d_.row_map.view_device(),value_2d_.row_map.view_host());
    Kokkos::deep_copy(value_2d_.entries.view_device(),value_2d_.entries.view_host());
  }

  // This function can just be overloaded with the Kokkos::View type.
  KOKKOS_INLINE_FUNCTION void getCellFaces(int id, Kokkos::View<const int*,ExecutionSpace>& ret) const {
      ret = Kokkos::subview(value_2d_.entries.view_device(),
        Kokkos::make_pair(value_2d_.row_map.view_device()(id),
        value_2d_.row_map.view_device()(id + 1)));
  }
  void getCellFaces(int id, Kokkos::View<const int*,Kokkos::HostSpace>& ret) const {
      Kokkos::subview(value_2d_.entries.view_host(),
        Kokkos::make_pair(value_2d_.row_map.view_host()(id),
        value_2d_.row_map.view_host()(id + 1)));
  }

  // The following function cannot be overloaded with parameters
  // Default: only use Device function
  KOKKOS_INLINE_FUNCTION int getCellVolume(int i) const {
    return value_1d_.view_device()(i);
  }

  // Duplicate function with host decoration
  int getCellVolume_host(int i) const {
    return value_1d_.view_host()(i);
  }

  // Using C++ 17 with constexpr
  #if V2
  template<typename FExecutionSpace = Kokkos::HostSpace>
  int getCellVolume(int i) const {
    if constexpr(std::is_same<Kokkos::HostSpace,FExecutionSpace>::value) {
      return value_1d_.view_host()(i);
    } else if (std::is_same<Kokkos::CudaSpace,FExecutionSpace>::value) {
      return value_1d_.view_device()(i);
    }
  }
  #endif

private:
  Kokkos::DualView<int*,ExecutionSpace> value_1d_;
  DualCrs<int*,ExecutionSpace> value_2d_;
  MeshFramework* framework_mesh_;
  int mesh_size_;
};




void test_mesh()  {
    using DeviceSpace = Kokkos::CudaSpace;
    const int ncells = 10;
    MeshFramework m(ncells);
    MeshCache<DeviceSpace> mc(&m);

    // ------- Mesh Framework -------
    // Only accessible on Host
    for(int i = 0 ; i < ncells ; ++i){
      // 1d
      assert(m.getCellVolume(i) == i);
      // 2d
      Kokkos::View<const int*, Kokkos::HostSpace> v_h;
      m.getCellFaces(2, v_h);
      for(int j = 0 ; j < 2 ; ++j){
        assert(v_h(j) == j*10);
      }
    }

    // ------- Mesh Cache -------
    // 1. Device Access (Default type of the mesh cache)
    Kokkos::parallel_for(
      "",
      ncells,
      KOKKOS_LAMBDA(const int i){
        // 1d
        assert(mc.getCellVolume(i) == i);
        // 2d
        Kokkos::View<const int*, DeviceSpace> v_d;
        mc.getCellFaces(2,v_d);
        for(int j = 0 ; j < 2 ; ++j){
          assert(v_d(j) == j*10);
        }
    });
    // 2. Host Access
    for(int i = 0 ; i < ncells ; ++i){
      // 1d
      assert(mc.getCellVolume_host(i) == i);
      // 2d
      Kokkos::View<const int*, Kokkos::HostSpace> v_h;
      mc.getCellFaces(2, v_h);
    }
} // test_mesh


int main(int argc, char * argv[]) {
  Kokkos::initialize(argc,argv);
  test_mesh();
  Kokkos::finalize();
  return 0;
}
