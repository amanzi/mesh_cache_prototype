#include "Kokkos_Core.hpp"
#include "Kokkos_DualView.hpp"

using Entity_ID = int;
enum memory {host, device};
using Coordinate = std::array<double,3>;


template<memory MEM>
struct MeshCache;

struct MeshFramework; 
struct MeshCacheData;

namespace Algorithms {
template<typename Mesh_type,
  memory MEM = device>
void computeCellGeometry(Mesh_type& m) {
  if constexpr(MEM == device) {
    Kokkos::parallel_for(
      "",m.mesh_size_,KOKKOS_LAMBDA(const int c){
      auto nnodes = m.getCellNumNodes(c);
      m.cellCentroid(c) = {0,0,0};
      for (int i=0; i!=nnodes; ++i) {
        auto nc = m.getNodeCoordinate(m.getCellNode(c,i));
        m.cellCentroid(c)[0] += nc[0];
        m.cellCentroid(c)[1] += nc[1];
        m.cellCentroid(c)[2] += nc[2];
      }
      m.cellCentroid(c)[0] /= nnodes;
      m.cellCentroid(c)[1] /= nnodes;
      m.cellCentroid(c)[2] /= nnodes;
    }); 
  } else {
    for (Entity_ID c=0; c!=m.mesh_size_; ++c) {
      auto nnodes = m.getCellNumNodes(c);
      m.cellCentroid(c) = {0,0,0};
      for (int i=0; i!=nnodes; ++i) {
        auto nc = m.getNodeCoordinate(m.getCellNode(c,i));
        m.cellCentroid(c)[0] += nc[0];
        m.cellCentroid(c)[1] += nc[1];
        m.cellCentroid(c)[2] += nc[2];
      }
      m.cellCentroid(c)[0] /= nnodes;
      m.cellCentroid(c)[1] /= nnodes;
      m.cellCentroid(c)[2] /= nnodes;
    }
  }
}
}// namespace Algorithms

// MeshAlgorithm templated on the memory 
#if 1 
template<memory MEM> 
struct MeshAlgorithm{
  virtual void computeCellGeometry(MeshCache<MEM>& m) = 0;
  virtual void computeCellGeometry(MeshFramework& m) = 0;
  
  virtual MeshAlgorithm<device>* create_device() = 0; 
  virtual MeshAlgorithm<host>* create_host() = 0; 
};

template<memory MEM>
struct MeshAlgorithmsDefault: MeshAlgorithm<MEM>  {
  virtual void computeCellGeometry(MeshCache<MEM>& m) {
    Algorithms::computeCellGeometry(m);
  }
  virtual void computeCellGeometry(MeshFramework& m) {
    Algorithms::computeCellGeometry<MeshFramework, host>(m);
  }
  virtual MeshAlgorithmsDefault<device>* create_device(){
    return new MeshAlgorithmsDefault<device>(); 
  }
  virtual MeshAlgorithmsDefault<host>* create_host(){
    return new MeshAlgorithmsDefault<host>(); 
  }
};
#endif 

// WIP MeshAlgorithm templated on the Mesh type 
#if 0
template<typename MESH> 
struct MeshAlgorithm{
  virtual void computeCellGeometry(MESH& m) = 0;
  virtual MeshAlgorithm<device>* create_device() = 0;
  virtual MeshAlgorithm<host>* create_host() = 0; 
};

template<typename MESH>
struct MeshAlgorithmsDefault: MeshAlgorithm<MESH>  { 
  virtual void computeCellGeometry(MESH& m) {
    Algorithms::computeCellGeometry<MESH,MESH::mem>(m);
  }
  virtual MeshAlgorithmsDefault<device>* create_device(){
    return new MeshAlgorithmsDefault<device>();    
  } 
  virtual MeshAlgorithmsDefault<host>* create_host(){
    return new MeshAlgorithmsDefault<host>(); 
  }
};
#endif 

//
// Note, this API is consistent with amanzi/ecoon/mesh_refactor src/mesh/MeshFramework.hh API
//
struct MeshFramework {
  static constexpr memory mem = host; 
  using Coordinate = std::array<double,3>;
  MeshFramework(const int size): mesh_size_(size) {
    //maptr_ = MeshAlgorithmPointers(new MeshAlgorithmsDefault<host>(), new MeshAlgorithmsDefault<device>()); 
    // Init singleton 
    //algorithms_ = maptr_.mah_; 
    algorithms_ = new MeshAlgorithmsDefault<host>(); 
  }
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
  Entity_ID getCellFace(Entity_ID c, std::size_t i) const { return value*i;}
  Entity_ID meshSize() const { return mesh_size_; }
  std::size_t getCellNumNodes(const Entity_ID c) const { return 0; }
  Coordinate getNodeCoordinate(Entity_ID n) const { return {0,0,0};}
  Entity_ID getCellNode(Entity_ID c, std::size_t i) const {return 0;}
  auto cellCentroids(){return cell_centroids; }
  auto cellCentroid(Entity_ID c){ return cell_centroids(c); }

public:
  const Entity_ID value = 10;
  int mesh_size_;
  MeshAlgorithm<host>* algorithms_; 
  //MeshAlgorithmPointers maptr_; 
  Kokkos::View<Coordinate*,Kokkos::HostSpace> cell_centroids; 
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

struct MeshCacheData{

  // Nothing to do, View are destroyed and memory is freed
  ~MeshCacheData(){}

  Kokkos::DualView<double*,Kokkos::DefaultExecutionSpace> value_1d_;
  DualCrs<Entity_ID*> value_2d_;
  Kokkos::DualView<Coordinate*,Kokkos::DefaultExecutionSpace> cell_centroids;

  DualCrs<Entity_ID*> cell_nodes;
  Kokkos::DualView<Coordinate*,Kokkos::DefaultExecutionSpace> node_coordinates;
};

// MeshCache templated on the memory type
// Defaulted to the device memory space
template<memory MEM = device>
struct MeshCache {

  static constexpr memory mem = MEM; 

  template<memory M>
  MeshCache(const MeshCache<M>& mc){
    this->mcd = mc.mcd; 
    this->framework_mesh_ = mc.framework_mesh_;
    this->mesh_size_ = mc.mesh_size_;
  }

  ~MeshCache(){
    // Release the shared pointer on destruction 
    mcd.reset(); 
  }

  // Building a Mesh Cache on ExecutionSpace memory space.
  // Retrieve/compute data on the Host from the Mesh Framework then copy to device
  MeshCache(MeshFramework* mf): framework_mesh_(mf), mesh_size_(mf->meshSize()) {
    
    mcd = std::make_shared<MeshCacheData>(); 
    
    algorithms_->computeCellGeometry(*this);

    std::cout<<"Building cache"<<std::endl;

    ////// This could be in the algorithms 
    Kokkos::resize(mcd->value_1d_,mesh_size_);
    Kokkos::resize(mcd->value_2d_.row_map,mesh_size_+1);
    Kokkos::resize(mcd->value_2d_.entries,mesh_size_*2);
    for(int i = 0 ; i < mesh_size_ ; ++i){
      mcd->value_1d_.view_host()(i) = mf->getCellVolume(i);
    }
    for(int i = 0 ; i < mesh_size_ ; ++i){
      mcd->value_2d_.row_map.view_host()(i) = i*2;
      Kokkos::View<const int*, Kokkos::HostSpace> tmp;
      tmp = mf->getCellFaces(i);
      for(int j = 0 ; j < 2; ++j){
        mcd->value_2d_.entries.view_host()(i*2+j) = tmp(j);
      }
    }
    mcd->value_2d_.row_map.view_host()(mesh_size_) = mesh_size_*2;
    // Copy to device memory
    Kokkos::deep_copy(mcd->value_1d_.view_device(),mcd->value_1d_.view_host());
    Kokkos::deep_copy(mcd->value_2d_.row_map.view_device(),mcd->value_2d_.row_map.view_host());
    Kokkos::deep_copy(mcd->value_2d_.entries.view_device(),mcd->value_2d_.entries.view_host());


    std::cout<<"Building cache done"<<std::endl;
  }

  // This function can just be overloaded with the Kokkos::View type.
  KOKKOS_INLINE_FUNCTION auto getCellFaces(Entity_ID id) const {
    if constexpr (MEM == device){
      Kokkos::View<const int*,Kokkos::DefaultExecutionSpace> ret = Kokkos::subview(mcd->value_2d_.entries.view_device(),
        Kokkos::make_pair(mcd->value_2d_.row_map.view_device()(id),
        mcd->value_2d_.row_map.view_device()(id + 1)));
      return ret; 
    } else { 
      Kokkos::View<const Entity_ID*,Kokkos::HostSpace> ret = Kokkos::subview(mcd->value_2d_.entries.view_host(),
        Kokkos::make_pair(mcd->value_2d_.row_map.view_host()(id),
        mcd->value_2d_.row_map.view_host()(id + 1)));
      return ret; 
    } 
  }

  KOKKOS_INLINE_FUNCTION Entity_ID getCellFace(Entity_ID c, int i) const {
    if constexpr (MEM == device){
      return mcd->value_2d_.entries.view_device()(mcd->value_2d_.row_map.view_device()(c)+i);
    } else {
      return mcd->value_2d_.entries.view_host()(mcd->value_2d_.row_map.view_host()(c)+i); 
    }
    // Not needed, to silence NVCC spurious warning 
    return 0; 
  }

  // The following function cannot be overloaded with parameters
  // Default: only use Device function
  KOKKOS_INLINE_FUNCTION double getCellVolume(Entity_ID i) const {
    if constexpr (MEM == device){
      return mcd->value_1d_.view_device()(i);
    } else {
      return mcd->value_1d_.view_host()(i);
    }
    // Not needed, to silence NVCC spurious warning 
    return 0; 
  }

  KOKKOS_INLINE_FUNCTION Coordinate& cellCentroid(const Entity_ID c) const {
    if constexpr (MEM == device){
      return mcd->cell_centroids.view_device()(c); 
    } else {
      return mcd->cell_centroids.view_host()(c); 
    }
  }

  KOKKOS_INLINE_FUNCTION auto cellCentroids() const {
    if constexpr (MEM == device){
      return mcd->cell_centroids.view_device(); 
    } else { 
      return mcd->cell_centroids.view_host(); 
    }
  }

  KOKKOS_INLINE_FUNCTION std::size_t getCellNumNodes(const Entity_ID c) const {
    if constexpr (MEM == device){
    return mcd->cell_nodes.row_map.view_device()(c+1)-mcd->cell_nodes.row_map.view_device()(c); 
    } else {
      return mcd->cell_nodes.row_map.view_host()(c+1)-mcd->cell_nodes.row_map.view_host()(c); 
    }
  }


  KOKKOS_INLINE_FUNCTION auto getCellNodes(const Entity_ID c) const {
    if constexpr (MEM == device){
      Kokkos::View<const Entity_ID*, Kokkos::DefaultExecutionSpace> view; 
      return view; 
    } else {
      Kokkos::View<const Entity_ID*, Kokkos::HostSpace> view; 
      return view; 
    }
  }


  KOKKOS_INLINE_FUNCTION Entity_ID getCellNode(Entity_ID c, std::size_t i) const {
    if constexpr (MEM == device){
      return mcd->cell_nodes.entries.view_device()(mcd->cell_nodes.row_map.view_device()(c)+i); 
    }else{
      return mcd->cell_nodes.entries.view_host()(mcd->cell_nodes.row_map.view_host()(c)+i); 
    }
  }


  KOKKOS_INLINE_FUNCTION Coordinate getNodeCoordinate(const Entity_ID n) const {
    if constexpr (MEM == device){
      return mcd->node_coordinates.view_device()(n); 
    } else {
      return mcd->node_coordinates.view_host()(n); 
    }
  }

  MeshAlgorithm<MEM>* algorithms_;
  std::shared_ptr<MeshCacheData> mcd; 
  MeshFramework* framework_mesh_;
  int mesh_size_;
};

void test_mesh()  {

    using DeviceSpace = Kokkos::CudaSpace;
    const int ncells = 10;
    MeshFramework m(ncells);

    // ------- Mesh Framework -------
    std::cout<<"Mesh Framework"<<std::endl;
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
    std::cout<<"Mesh Cache Device"<<std::endl;

    MeshCache<device> mc(&m);
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

    std::cout<<"Mesh Cache Host"<<std::endl;
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
