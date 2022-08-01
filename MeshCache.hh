#ifndef _MESHCACHE_HH_
#define _MESHCACHE_HH_

#include "MeshDefs.hh"
#include "MeshFramework.hh"


struct MeshCacheData{ 
	RaggedArray_DualView<Coordinates> cell_coordinates; 
  RaggedArray_DualView<Entity_ID> cell_nodes;
  RaggedArray_DualView<Entity_ID> cell_faces;
  Point_DualView node_coordinates;
  

}; 

template<MemSpace_type MEM = MemSpace_type::DEVICE> 
struct MeshCache{ 

	private: 
		MeshCacheData & data_; 
		bool cell_coordinates_cached = false; 
    bool cell_nodes_cached = false; 
    bool  node_coordinates_cached = false; 
    bool cell_geometry_cached = false; 
    const MeshFramework* mesh_framework_; 

	public:

		MeshCache(MeshCacheData& d, const MeshFramework& mf): data_(d), mesh_framework_(&mf) {}


		template<AccessPattern AP = AccessPattern::DEFAULT>
			KOKKOS_INLINE_FUNCTION
			cPoint_View getCellCoordinates(const Entity_ID c) const{
				return RaggedGetter<MEM,AP>::get(
						*this,
						cell_coordinates_cached,
						data_.cell_coordinates,
						nullptr,
						[&](const Entity_ID nn, const std::string& s){
						  return MeshAlgorithms::getCellCoordinates(*this,nn);
						},
						__func__, 
						c);  
			}
  
  template<AccessPattern AP = AccessPattern::DEFAULT>
  KOKKOS_INLINE_FUNCTION 
  decltype(auto) // Coordinates
  getNodeCoordinate(const Entity_ID n) const {
    return Getter<MEM,AP>::get(
        *this,
        node_coordinates_cached,        
        data_.node_coordinates,        
        [&](const Entity_ID nn, const std::string& s){
          if(mesh_framework_) 
            return mesh_framework_->getNodeCoordinate(nn);
          assert(false); 
          return Coordinates{};
          },
        nullptr,                            
        __func__,               
        n);                             
  }

  template<AccessPattern AP = AccessPattern::DEFAULT> 
  KOKKOS_INLINE_FUNCTION
  cEntity_ID_View getCellNodes(const Entity_ID c) const 
  {
    return RaggedGetter<MEM,AP>::get(
        *this,
        cell_nodes_cached,
        data_.cell_nodes,
        [&](const Entity_ID nn, const std::string& s){
          if(mesh_framework_){
            cEntity_ID_View cnodes; 
            mesh_framework_->getCellNodes(nn, cnodes);
            return cnodes; 
          }
          assert(false); 
          return cEntity_ID_View{}; 
        },
        nullptr,
        __func__, 
        c); 
  }

  template<AccessPattern AP = AccessPattern::DEFAULT>
  KOKKOS_INLINE_FUNCTION
  cEntity_ID_View getCellFaces(const Entity_ID c) const 
  {
    return RaggedGetter<MEM,AP>::get(
        *this,
        cell_geometry_cached,
        data_.cell_faces,
        [&](const Entity_ID nn, const std::string& s){
          if(mesh_framework_){
            cEntity_ID_View cfaces; 
            mesh_framework_->getCellFaces(nn,cfaces);
            return cfaces; 
          }
          assert(false); 
          return cEntity_ID_View{};
        },
        nullptr,
        __func__, 
        c);  
  }

};




#endif 