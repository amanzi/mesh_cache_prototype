#include "Kokkos_Core.hpp"
#include "Kokkos_DualView.hpp"

using Entity_ID = int;

enum memory {host, device};


//
// Note, this API is consistent with amanzi/ecoon/mesh_refactor src/mesh/MeshFramework.hh API
//
struct MeshFramework {
  MeshFramework(const int size): mesh_size_(size) {}

  // In MeshFramework this function "compute" the result on Host
  double getCellVolume(const Entity_ID i) const {
    return i;
  }

  // In MeshFramework this function "compute" the result on Host
  Kokkos::View<const Entity_ID*,Kokkos::HostSpace> getCellFaces(const Entity_ID id) const {
    // 2 Elements per entries
    Kokkos::View<Entity_ID*, Kokkos::HostSpace> tmp("",2);
    for(int i = 0 ; i < 2; ++i){
      tmp(i) = value*i;
    }
    return tmp;
  }

  Entity_ID getCellFace(Entity_ID c, std::size_t i) const {
    return value*i; 
  }

  Entity_ID meshSize() const { return mesh_size_; }

private:
  const Entity_ID value = 10;
  int mesh_size_;
};


// Type to get CRS on DualViews
template<typename T>
struct DualCrs{
  Kokkos::DualView<int*, Kokkos::DefaultExecutionSpace> row_map;
  Kokkos::DualView<T, Kokkos::DefaultExecutionSpace> entries;

  DualCrs() = default; 
  DualCrs(const DualCrs&) = default; 

  DualCrs& operator=(const DualCrs& o){
    row_map = o.row_map; 
    entries = o.entries; 
    return *this; 
  }  

};

// MeshCache templated on the memory type
// Defaulted to the device memory space
template<memory MEM = device>
struct MeshCache {

  template<memory M>
  MeshCache(const MeshCache<M>& mc){
    this->value_1d_ = mc.value_1d_; 
    this->value_2d_ = mc.value_2d_; 
    this->framework_mesh_ = mc.framework_mesh_;
    this->mesh_size_ = mc.mesh_size_;
  }

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
      tmp = mf->getCellFaces(i);
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
  KOKKOS_INLINE_FUNCTION auto getCellFaces(Entity_ID id) const {
    if constexpr (MEM == device){
      Kokkos::View<const int*,Kokkos::DefaultExecutionSpace> ret = Kokkos::subview(value_2d_.entries.view_device(),
        Kokkos::make_pair(value_2d_.row_map.view_device()(id),
        value_2d_.row_map.view_device()(id + 1)));
      return ret; 
    } else if (MEM == host) { 
      Kokkos::View<const Entity_ID*,Kokkos::HostSpace> ret = Kokkos::subview(value_2d_.entries.view_host(),
        Kokkos::make_pair(value_2d_.row_map.view_host()(id),
        value_2d_.row_map.view_host()(id + 1)));
      return ret; 
    }
  }

  KOKKOS_INLINE_FUNCTION Entity_ID getCellFace(Entity_ID c, int i) const {
    if constexpr (MEM == device){
      return value_2d_.entries.view_device()(value_2d_.row_map.view_device()(c)+i);
    } else if (MEM == host){
      return value_2d_.entries.view_host()(value_2d_.row_map.view_host()(c)+i); 
    }
  }

  // The following function cannot be overloaded with parameters
  // Default: only use Device function
  KOKKOS_INLINE_FUNCTION double getCellVolume(Entity_ID i) const {
    if constexpr (MEM == device){
      return value_1d_.view_device()(i);
    } else if (MEM == host) {
      return value_1d_.view_host()(i);
    }
  }

public:
  Kokkos::DualView<double*,Kokkos::DefaultExecutionSpace> value_1d_;
  DualCrs<Entity_ID*> value_2d_;
  MeshFramework* framework_mesh_;
  int mesh_size_;
};

void test_mesh()  {

    using DeviceSpace = Kokkos::CudaSpace;
    const int ncells = 10;
    MeshFramework m(ncells);


    // ------- Mesh Framework -------
    // Only accessible on Host
    for(int i = 0 ; i < ncells ; ++i){
      if(m.getCellVolume(i) != i)
        printf("ERROR %.1f != %d \n",m.getCellVolume(i),i);
      Kokkos::View<const int*, Kokkos::HostSpace> v_h;
      v_h = m.getCellFaces(2);
      for(int j = 0 ; j < 2 ; ++j){
        if(v_h(j) != j*10)
          printf("ERROR %d != %d \n",v_h(j),j*10);
      }
      for(int j = 0 ; j < 2 ; ++j){
        if(m.getCellFace(2,j) != j*10)
          printf("ERROR %d != %d \n",m.getCellFace(2,j),j*10);
      }
    }

    
    // ------- Mesh Cache -------
    // Create MeshCache, by default on device 
    MeshCache mc(&m);

    // 1. Device Access (Default type of the mesh cache)
    Kokkos::parallel_for(
      ncells, KOKKOS_LAMBDA(const int i){
        if(mc.getCellVolume(i) != i)
          printf("ERROR %.1f != %d \n",mc.getCellVolume(i),i);
        Kokkos::View<const int*, DeviceSpace> v_d;
        v_d = mc.getCellFaces(2);
        for(int j = 0 ; j < 2 ; ++j){
          if(v_d(j) != j*10)
            printf("ERROR %d != %d \n",v_d(j),j*10);
        }
        for(int j = 0 ; j < 2 ; ++j){
          if(mc.getCellFace(2,j) != j*10)
            printf("ERROR %d != %d \n",mc.getCellFace(2,j),j*10);
        }
    });

    MeshCache<host> mc_h(mc);

    // 2. Host Access
    for(int i = 0 ; i < ncells ; ++i){
      if(mc_h.getCellVolume(i) != i)
        printf("ERROR %.1f != %d \n",mc_h.getCellVolume(i),i);
      Kokkos::View<const int*, Kokkos::HostSpace> v_h;
      v_h = mc_h.getCellFaces(2);
      for(int j = 0 ; j < 2 ; ++j){
        if(v_h(j) != j*10)
          printf("ERROR %d != %d \n",v_h(j),j*10);
      }
      for(int j = 0 ; j < 2 ; ++j){
        if(mc_h.getCellFace(2,j) != j*10)
          printf("ERROR %d != %d \n",mc_h.getCellFace(2,j),j*10);
      }
    }
} // test_mesh


int main(int argc, char * argv[]) {
  Kokkos::initialize(argc,argv);
  test_mesh();
  Kokkos::finalize();
  return 0;
}
