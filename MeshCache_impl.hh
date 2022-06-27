#pragma once
#include "MeshDefs.hh"


template<memory MEM>
template<memory M>
MeshCache<MEM>::MeshCache(const MeshCache<M>& mc){
  this->mcd = mc.mcd;
  this->num_cells = mc.getNumCells();
  this->num_nodes = mc.getNumNodes();
  this->framework_mesh_ = mc.framework_mesh_;
  this->algorithms_ = mc.algorithms_->template create_on<MEM>();
}



template<memory MEM>
MeshCache<MEM>::MeshCache(MeshFramework* mf)
  : framework_mesh_(mf),
    algorithms_(mf->algorithms_->template create_on<MEM>()),
    num_cells(mf->getNumCells()),
    num_nodes(mf->getNumNodes())
{
  mcd = std::make_shared<MeshCacheData>();
  std::cout<<"Building cache"<<std::endl;

  // copy coordinates
  Kokkos::resize(mcd->node_coordinates, num_nodes); // is this required?
  for (int i=0; i!=num_nodes; ++i)
    mcd->node_coordinates.view_host()[i] = mf->getNodeCoordinate(i);

  // copy cell-node connections
  Kokkos::resize(mcd->cell_nodes.row_map, num_cells+1);

  int nconn = 0;
  for (int i=0; i!=num_cells; ++i) {
    nconn += mf->getCellNumNodes(i);
  }
  Kokkos::resize(mcd->cell_nodes.entries, nconn);

  nconn = 0;
  for (int i=0; i!=num_cells; ++i) {
    mcd->cell_nodes.row_map.view_host()[i] = nconn;
    int c_nnodes = mf->getCellNumNodes(i);
    for (int j=0; j!=c_nnodes; ++j) {
      mcd->cell_nodes.entries.view_host()[nconn+j] = mf->getCellNode(i,j);
    }
    nconn += c_nnodes;
  }
  mcd->cell_nodes.row_map.view_host()[num_cells] = nconn;

  // cache cell centroids
  Kokkos::resize(mcd->cell_volumes, getNumCells());
  Kokkos::deep_copy(mcd->cell_volumes.view_device(), algorithms_->computeCellVolumes(*this));
  Kokkos::resize(mcd->cell_centroids, getNumCells());
  Kokkos::deep_copy(mcd->cell_centroids.view_device(), algorithms_->computeCellCentroids(*this));
  std::cout<<"Building cache done"<<std::endl;
}
