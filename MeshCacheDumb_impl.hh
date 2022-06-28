#pragma once
#include "MeshDefs.hh"


template<memory MEM>
MeshCacheDumb<MEM>::MeshCacheDumb(MeshFramework* mf)
  : framework_mesh_(mf),
    algorithms_(mf->algorithms_->template create_on<MEM>()),
    num_cells(mf->getNumCells()),
    num_nodes(mf->getNumNodes())
{
  std::cout<<"Building cache"<<std::endl;

  // copy coordinates
  node_coordinates.resize(num_nodes);
  for (int i=0; i!=num_nodes; ++i)
    node_coordinates[i] = mf->getNodeCoordinate(i);

  // copy cell-node connections
  cell_nodes.resize(num_cells);

  for (int i=0; i!=num_cells; ++i) {
    int c_nnodes = mf->getCellNumNodes(i);
    cell_nodes[i].resize(c_nnodes);
    for (int j=0; j!=c_nnodes; ++j) {
      cell_nodes[i][j] = mf->getCellNode(i,j);
    }
  }

  // cache cell centroids
  auto cv = algorithms_->computeCellVolumes(*this);
  cell_volumes.resize(num_cells);
  for (int i=0; i!=num_cells; ++i) {
    cell_volumes[i] = cv[i];
  }

  auto cc = algorithms_->computeCellCentroids(*this);
  cell_centroids.resize(num_cells);
  for (int i=0; i!=num_cells; ++i) {
    cell_centroids[i] = cc[i];
  }
  std::cout<<"Building cache done"<<std::endl;
}
