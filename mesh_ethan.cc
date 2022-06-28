#include "MeshDefs.hh"
#include "MeshFramework1D.hh"
#include "MeshCache_impl.hh"

#include "MeshCacheDumb_decl.hh"
#include "MeshCacheDumb_impl.hh"

#define NCELLS 5000;
#define NTIMES 100000;

void test_mesh()  {
  const int ncells = NCELLS;
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
  // Device access correctness
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

  // Device access timing
  const int ntimes = NTIMES;
  {
    Kokkos::Timer timer;
    double max = 0.;
    for (int count=0; count != ntimes; ++count) {
      double total = 0.;
      Kokkos::parallel_reduce(
        ncells, KOKKOS_LAMBDA(const int i, double& lsum){
          lsum += mc.getCellVolume(i);
        }, total);
      max = std::max(max, total);
    }
    double time = timer.seconds();
    std::cout << "MeshCache on device: " << time/ntimes << " got " << max << std::endl;
  }

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

  for(int i = 0 ; i != ncells ; ++i){
    if(mc_h.getCellVolume(i) != i+1)
      printf("ERROR %.1f != %d \n",mc_h.getCellVolume(i),i+1);
  }

  // Device access timing
  {
    Kokkos::Timer timer;
    double max = 0.;
    for (int count=0; count != ntimes; ++count) {
      double total = 0.;
      for (int i=0; i!=ncells; ++i) {
          total += mc.getCellVolume(i);
      }
      max = std::max(max, total);
    }
    double time = timer.seconds();
    std::cout << "MeshCache on host: " << time/ntimes << " got " << max << std::endl;
  }

  // -------- MeshCacheDumb only on host ----
  std::cout << "Mesh Cache DUMB on host" << std::endl;
  MeshCacheDumb<host> mcd(&m);
  for (int i = 0; i != ncells; ++i) {
    if (mcd.getCellVolume(i) != i+1)
      printf("ERROR %.1f != %d \n", mcd.getCellVolume(i), i+1);
  }

  // Device access timing
  {
    Kokkos::Timer timer;
    double max = 0.;
    for (int count=0; count != ntimes; ++count) {
      double total = 0.;
      for (int i=0; i!=ncells; ++i) {
          total += mcd.getCellVolume(i);
      }
      max = std::max(max, total);
    }
    double time = timer.seconds();
    std::cout << "MeshCacheDumb on host: " << time/ntimes << " got " << max << std::endl;
  }
}



int main(int argc, char * argv[]) {
  Kokkos::initialize(argc,argv);
  test_mesh();
  Kokkos::finalize();
  return 0;
}
