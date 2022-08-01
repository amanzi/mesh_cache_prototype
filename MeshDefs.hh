#ifndef _MESHDEFS_HH_
#define _MESHDEFS_HH_

#include "Kokkos_Core.hpp"
#include "Kokkos_DualView.hpp"
#include "Point.hh"

using Entity_ID = int; 
enum MemSpace_type {HOST, DEVICE};
enum AccessPattern {DEFAULT, CACHE, FRAMEWORK, COMPUTE}; 

using Coordinates = Point;

using cEntity_ID_List = const std::vector<Entity_ID>;
using Entity_ID_List = std::vector<Entity_ID>;
using Point_List = std::vector<Coordinates>; 

template<typename T> using View_type = Kokkos::View<T*, Kokkos::HostSpace>;
using Point_View = View_type<Coordinates>;
using cPoint_View = View_type<const Coordinates>;
using Entity_ID_View = View_type<Entity_ID>;
using cEntity_ID_View = View_type<const Entity_ID>;

template<typename T> using DualView_type = Kokkos::DualView<T*, Kokkos::HostSpace>;
using Entity_ID_DualView = DualView_type<Entity_ID>;
using Entity_Direction_DualView = DualView_type<int>;
using Point_DualView = DualView_type<Coordinates>;
using Double_DualView = DualView_type<double>;
using Entity_Direction_View = View_type<int>;
using cEntity_Direction_View = View_type<const int>;

template<MemSpace_type MEM, typename DVT>
KOKKOS_INLINE_FUNCTION
auto View(DVT& dv){
	if constexpr(MEM == MemSpace_type::HOST)
		return dv.view_host();
	else
		return dv.view_device();
}

template<typename T> 
struct RaggedArray_DualView { 
  using type_t = T;
  using constview = View_type<const T>; 
  Entity_ID_DualView row; 
  DualView_type<T> entries; 
};


template<typename V, typename VEC> 
KOKKOS_INLINE_FUNCTION
void ViewToVector(V view, VEC& v){ 
  v = std::vector<typename V::traits::non_const_value_type>(view.data(),view.data()+view.size()); 
}

template<typename T> 
KOKKOS_INLINE_FUNCTION
Kokkos::View<typename T::value_type*,Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
VectorToView(T& vector){
  Kokkos::View<typename T::value_type*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> view (vector.data(),vector.size()); 
  return view; 
}

// Getter for DualView
template<MemSpace_type MEM = MemSpace_type::HOST, AccessPattern AP = AccessPattern::DEFAULT>
struct Getter{ 
  template<typename MESH, typename DATA, typename F, typename C>
  static KOKKOS_INLINE_FUNCTION decltype(auto)
  get (MESH& m, bool cached, DATA& d, F&& f, C&&c,const std::string& s,  const Entity_ID n) { 
    using type_t = typename DATA::t_dev::traits::value_type; 
    if (cached) {
      return static_cast<type_t>(View<MEM>(d)[n]);
    }
    if constexpr(!std::is_same<F,decltype(nullptr)>::value)
      return f(n,s);
    if constexpr(!std::is_same<C,decltype(nullptr)>::value)
      return c(n,s); 
    assert(false);  
		return type_t{};
  }
};

template<MemSpace_type MEM>
struct Getter<MEM,AccessPattern::CACHE>{
  template<typename MESH, typename DATA, typename F, typename C>
  static KOKKOS_INLINE_FUNCTION decltype(auto)
  get(MESH& m, bool, DATA& d, F&&, C&&, const std::string&, const Entity_ID n)   { 
    return View<MEM>(d)[n];
  }
};

template<MemSpace_type MEM>
struct Getter<MEM,AccessPattern::FRAMEWORK>{
  template<typename MESH, typename DATA, typename F, typename C>
  static KOKKOS_INLINE_FUNCTION decltype(auto)
  get (MESH& m, bool, DATA&, F&& f, C&&, const std::string& s, const Entity_ID n) { 
    static_assert(!std::is_same<F,decltype(nullptr)>::value); 
    return f(n,s);
  }
};

template<MemSpace_type MEM>
struct Getter<MEM,AccessPattern::COMPUTE>{
  template<typename MESH, typename DATA, typename F, typename C>
  static KOKKOS_INLINE_FUNCTION decltype(auto)
  get (MESH& m, bool, DATA&, F&&, C&& c, const std::string& s, const Entity_ID n) {
    static_assert(!std::is_same<C,decltype(nullptr)>::value); 
    return c(n,s);  
  }
};


// Getters for raggedViews
template<MemSpace_type MEM, AccessPattern AP = AccessPattern::DEFAULT>
struct RaggedGetter{ 
	template<typename MESH, typename DATA, typename F, typename C>
		static KOKKOS_INLINE_FUNCTION typename DATA::constview
		get (MESH& mesh, bool cached, DATA& d, F&& f, C&& c, const std::string& s,  const Entity_ID n) {
			using type_t = typename DATA::type_t;  
			if (cached) return Kokkos::subview(View<MEM>(d.entries),Kokkos::make_pair(View<MEM>(d.row)(n),View<MEM>(d.row)(n+1)));
			if constexpr (!std::is_same<F,decltype(nullptr)>::value)
				return f(n,s); 
			if constexpr (!std::is_same<C,decltype(nullptr)>::value) 
				return c(n,s);
			assert(false);  
			return typename DATA::constview{};
		}
};

template<MemSpace_type MEM>
struct RaggedGetter<MEM,AccessPattern::CACHE>{
	template<typename MESH, typename DATA, typename F, typename C>
		static KOKKOS_INLINE_FUNCTION typename DATA::constview
		get (MESH&, bool , DATA& d, F&&, C&& , const std::string&, const Entity_ID n) { 
			return Kokkos::subview(View<MEM>(d.entries),Kokkos::make_pair(View<MEM>(d.row)(n),View<MEM>(d.row)(n+1)));
		}
};

template<MemSpace_type MEM>
struct RaggedGetter<MEM,AccessPattern::FRAMEWORK>{
	template<typename MESH, typename DATA, typename F, typename C>
		static KOKKOS_INLINE_FUNCTION typename DATA::constview
		get (MESH& m, bool cached, DATA&, F&& f, C&&, const std::string& s, const Entity_ID n) { 
			static_assert(!std::is_same<F,decltype(nullptr)>::value); 
			return f(n,s);
		}
};

template<MemSpace_type MEM>
struct RaggedGetter<MEM,AccessPattern::COMPUTE>{
	template<typename MESH, typename DATA, typename F, typename C>
		static KOKKOS_INLINE_FUNCTION typename DATA::constview
		get (MESH& m, bool cached, DATA&, F&&, C&& c, const std::string& s, const Entity_ID n) { 
			static_assert(!std::is_same<C,decltype(nullptr)>::value); 
			return c(n,s);  
		}
};




#endif 