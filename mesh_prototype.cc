#include "MeshCache.hh"

void cache(MeshCacheData& mcd){
  // Init the coordinates (fake init)
  const int ncells = 10; 
  const int vpercells = 2; 
  mcd.cell_coordinates.row.resize(ncells); 
  for(int i = 0 ; i < ncells ; ++i){
    mcd.cell_coordinates.row.view_host()(i) = vpercells; 
  }
  mcd.cell_coordinates.entries.resize(ncells*2); 
  for(int i = 0 ; i < ncells*vpercells ; ++i){
    Coordinates tmp = {static_cast<double>(i/vpercells),0.,0.}; 
    mcd.cell_coordinates.entries.view_host()(i) = tmp; 
  }
}

int main(int argc, char* argv[]){

  MeshFramework mf; 
	MeshCacheData mcd; 
	MeshCache<MemSpace_type::DEVICE> mc(mcd,mf); 

  // Use default 
  for(int i = 0 ; i < 10 ; ++i){
    auto v = mc.getCellCoordinates(i); 
    std::cout<<v(i).c[0]<<std::endl;
  }


	return 0; 
}
