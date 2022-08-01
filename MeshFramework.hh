#ifndef _MESHFRAMEWORK_HH_
#define _MESHFRAMEWORK_HH_

#include "MeshDefs.hh"

struct MeshAlgorithms{
	  template<class Mesh_type>
		KOKKOS_INLINE_FUNCTION 
    static cPoint_View getCellCoordinates(const Mesh_type& mesh, const Entity_ID c)
		{
			cEntity_ID_View nodes = mesh.getCellNodes(c);

			Point_View coords;
      Kokkos::resize(coords,nodes.size()); 

			for (int n = 0;  n < nodes.size(); ++n) {
				coords[n] = mesh.getNodeCoordinate(nodes[n]);
			}
			return coords;
		}

    // Fake implementation
    template<class Mesh_type>
    KOKKOS_INLINE_FUNCTION static Entity_ID_List
    computeCellNodes(const Mesh_type& mesh, const Entity_ID c)
    {
      Entity_ID_List nodes;
      const int nnodes = 2; 
      // Warning __host__ function on __device__
      nodes.resize(nnodes); 
      for(int i = 0 ; i < nnodes ; ++i){
        nodes[i] = c*2+i; 
      }
      return nodes;
    }
};

struct MeshFramework  {

KOKKOS_INLINE_FUNCTION void getCellNodes(const Entity_ID c, cEntity_ID_View& nodes) const
{
  auto v = MeshAlgorithms::computeCellNodes(*this, c);
  nodes = VectorToView(v); 
}

KOKKOS_INLINE_FUNCTION void getCellFaces(
      const Entity_ID c,
      cEntity_ID_View& faces) const {
  getCellFacesAndDirs(c, faces, nullptr);
}

KOKKOS_INLINE_FUNCTION void getCellFacesAndDirs(
  const Entity_ID c,
  cEntity_ID_View& faces,
  cEntity_Direction_View * const dirs) const {

}

KOKKOS_INLINE_FUNCTION Coordinates getNodeCoordinate(const Entity_ID node) const {
  return {1.,1.,1.}; 
}


}; // MeshFramework 

#endif 