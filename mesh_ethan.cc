#include "MeshDefs.hh"
#include "MeshFramework1D.hh"
#include "MeshCache_impl.hh"

void test_mesh()  {
  const int ncells = 10;
  MeshFramework1D m(ncells+1);

  // ------- Mesh Framework -------
  std::cout<<"Mesh Framework"<<std::endl;
  // Only accessible on Host
  auto cell_volumes = m.getCellVolumes();
  auto cell_centroids = m.getCellCentroids();

  Kokkos::View<const int*, Kokkos::HostSpace> v_h;
  v_h = m.getCellNodes(2);
  for(int j = 0 ; j < 2 ; ++j){
    if(v_h(j) != 2+j)
      printf("ERROR %d != %d \n",v_h(j),j*10);
  }
  for(int j = 0 ; j < 2 ; ++j){
    if(m.getCellNode(2,j) != 2+j)
      printf("ERROR %d != %d \n",m.getCellNode(2,j),2+j);
  }

  for(int i = 0 ; i < ncells ; ++i) {
    if(cell_volumes(i) != i+1) {
      printf("ERROR %.1f != %d \n",cell_volumes(i),i+1);
    }
  }


  // ------- Mesh Cache -------
  std::cout<<"Mesh Cache Device"<<std::endl;

  MeshCache<device> mc(&m);
  // 1. Device Access (Default type of the mesh cache)
  Kokkos::parallel_for(
    1, KOKKOS_LAMBDA(const int i){
      Kokkos::View<const int*, Kokkos::DefaultExecutionSpace> v_d;
      v_d = mc.getCellNodes(2);
      for(int j = 0 ; j < 2 ; ++j){
        if(v_d(j) != 2+j)
          printf("ERROR %d != %d \n",v_d(j),2+j);
      }
      for(int j = 0 ; j < 2 ; ++j){
        if(mc.getCellNode(2,j) != 2+j)
          printf("ERROR %d != %d \n",mc.getCellNode(2,j),2+j);
      }
    });

  Kokkos::parallel_for(
    ncells, KOKKOS_LAMBDA(const int i){
      if(mc.getCellVolume(i) != i+1)
        printf("ERROR %.1f != %d \n",mc.getCellVolume(i),i+1);
    });


  // ------- Mesh Cache on Host -------
  std::cout<<"Mesh Cache Host"<<std::endl;
  MeshCache<host> mc_h(mc);
  // 2. Host Access
  v_h = mc_h.getCellNodes(2);
  for(int j = 0 ; j < 2 ; ++j){
    if(v_h(j) != 2+j)
      printf("ERROR %d != %d \n",v_h(j),2+j);
  }
  for(int j = 0 ; j < 2 ; ++j){
    if(mc_h.getCellNode(2,j) != 2+j)
      printf("ERROR %d != %d \n",mc_h.getCellNode(2,j),2+j);
  }

  for(int i = 0 ; i < ncells ; ++i){
    if(mc_h.getCellVolume(i) != i+1)
      printf("ERROR %.1f != %d \n",mc_h.getCellVolume(i),i+1);
  }
}




int main(int argc, char * argv[]) {
  Kokkos::initialize(argc,argv);
  test_mesh();
  Kokkos::finalize();
  return 0;
}
