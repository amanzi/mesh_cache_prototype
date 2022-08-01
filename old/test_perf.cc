#include "Kokkos_Core.hpp"
#include "Kokkos_DualView.hpp"
#include <iomanip>

enum memory {host, device};
enum storage {cached, algorithm}; 

using dtype_t = float; 

template<typename T = float> 
struct heavy { 
  static constexpr int SIZE = 3;
  KOKKOS_INLINE_FUNCTION heavy(){}
  KOKKOS_INLINE_FUNCTION heavy(T val){
    for(int i = 0 ; i < SIZE ; ++i)
      v[i] = val+i; 
  }
  KOKKOS_INLINE_FUNCTION heavy(const heavy& h){
    printf("Call copy\n"); 
    for(int i = 0 ; i < SIZE ; ++i)
      v[i] = h.v[i]; 
  } 
  KOKKOS_INLINE_FUNCTION heavy operator+(const T& d){
    printf("Call plus\n");
    heavy res; 
    for(int i = 0 ; i < SIZE; ++i){
      res.v[i] = v[i]+d; 
    }
    return res; 
  } 
  T v[SIZE]; 
};

KOKKOS_INLINE_FUNCTION heavy<dtype_t> operator+(const heavy<dtype_t>& h, const float& d){
  printf("Call plus\n"); 
  heavy res; 
  for(int i = 0 ; i < heavy<dtype_t>::SIZE; ++i)
    res.v[i] = h.v[i]+d; 
  return res; 
}
 
struct test{

  test(const test& t) {
    // Copy all except vector 
    h_vector = t.h_vector; 
    h_view = t.h_view; 
    h_s_view_d = t.h_s_view_d; 
    h_s_view_h = t.h_s_view_h; 
  }

  ~test() {}

  test(int size){
    Kokkos::resize(h_view,size);
    h_vector.resize(size); 
    Kokkos::resize(h_s_view_d,size); 
    Kokkos::resize(h_s_view_h,size);  
  }

  template<memory MEM, typename DV_type>
  KOKKOS_INLINE_FUNCTION const auto& view(DV_type& dv) const{
    if constexpr (MEM == device)
      return dv.d_view; 
    else
      return dv.h_view; 
  }

  template<memory MEM>
  KOKKOS_INLINE_FUNCTION const auto& s_view() const{
    if constexpr (MEM == device)
      return h_s_view_d; 
    else
      return h_s_view_h; 
  }


  void fill(heavy<dtype_t> v){
    for(int i = 0 ; i < h_vector.size(); ++i){
      view<host>(h_view)(i) = v; 
      h_vector[i] = v; 
    }
    Kokkos::deep_copy(h_view.d_view,h_view.h_view);  
    Kokkos::deep_copy(h_s_view_d,h_view.h_view);
    Kokkos::deep_copy(h_s_view_h,h_view.h_view); 
  }

  template<memory MEM, storage ST = algorithm>
  KOKKOS_INLINE_FUNCTION 
  decltype(auto) getVal(const int i) {
    if constexpr(ST == cached){
      return view<MEM>(h_view)(i);  
    }else{
      // Compute the value and return 
      heavy<dtype_t> dt;
      // Compute ... Brr
      return dt; 
    } 
  }

  private:
  Kokkos::DualView<heavy<dtype_t>*> h_view;
  Kokkos::View<heavy<dtype_t>*> h_s_view_d; 
  Kokkos::View<heavy<dtype_t>*,Kokkos::HostSpace> h_s_view_h; 
  std::vector<heavy<dtype_t>> h_vector;

};

int main(int argc, char* argv[]){

  Kokkos::initialize(argc,argv); 

  heavy<dtype_t> h(5);
  std::vector<double> times(4);  

  for(int k = 1 ; k < 2; ++k)
  {
    const int size = pow(10,k);
    std::cout<<"Size = "<<size<<std::endl; 
    test tt(size);
    tt.fill(h); 
    test* t = &tt; 
    Kokkos::Timer timer;  
    std::vector<heavy<dtype_t>> res(size); 
    Kokkos::DualView<heavy<dtype_t>*> res_v("",size); 
    int cur = 0; 

    // VIEW HOST ==============================================================
    {
      // View by value HOST
      std::cout<<"HOST VIEW VAL"<<std::endl;
      timer.reset(); 
      for(int i = 0 ; i < size ; ++i){
        auto&& h = t->getVal<host>(i); 
        res_v.h_view(i).v[0] = h.v[0]+2; 
      }   
      double time = timer.seconds();
      times[cur++] = time;
    }

    {
      // View by ref HOST
      std::cout<<"HOST VIEW REF"<<std::endl;
      timer.reset(); 
      for(int i = 0 ; i < size ; ++i){
        auto&& h = t->getVal<host,cached>(i); 
        res_v.h_view(i).v[0] = h.v[0]+2;
      }   
      double time = timer.seconds();
      times[cur++] = time;
    }


    // VIEW CUDA ==============================================================
    {
      // View by value DEVICE
      std::cout<<"DEVICE VIEW VAL"<<std::endl;
      timer.reset(); 
      Kokkos::parallel_for("",size, KOKKOS_LAMBDA(int i){
        auto&& h = t->getVal<device>(i); 
        res_v.d_view(i).v[0] = h.v[0]+2; 
      }); 
      Kokkos::fence(); 
      double time = timer.seconds();
      // Copy back
      Kokkos::deep_copy(res_v.h_view,res_v.d_view);   
      times[cur++] = time;
    }

    {
      // View by ref DEVICE
      std::cout<<"DEVICE VIEW REF"<<std::endl;
      timer.reset(); 
      Kokkos::parallel_for("",size, KOKKOS_LAMBDA(int i){
        auto&& h = t->getVal<device,cached>(i); 
        res_v.d_view(i).v[0] = h.v[0]+2; 
      });
      Kokkos::fence();  
      double time = timer.seconds();
      // Copy back
      Kokkos::deep_copy(res_v.h_view,res_v.d_view); 
      times[cur++] = time;
    }

    //std::vector<std::string> vtype = {"Vector ","DView H","DView D","View H ","View D "}; 
    std::vector<std::string> vtype = {"DView H","DView D"}; 

    std::cout<<"           Algo.   Cache.  "<<std::endl;
    for(int i = 0 ; i < 2; ++i){ 
      std::cout<<vtype[i]<<" = ";
      for(int j = 0 ; j < 2; ++j){
        std::cout<<std::setfill(' ') << std::setw(5)<<std::setprecision(2)<<times[i*2+j]<<"  ";
      }
      std::cout<<std::endl;
    }
    std::cout<<std::endl;


  }  // for 
  Kokkos::finalize(); 
  return 0;
}


#if 0 
template<memory M> 


const Point_View
KOKKOS_INLINE_FUNCTION MeshCache::getEdgeCoordinates(const Entity_ID n) const
{
  if(edge_coordinates_cached) return getEdgeCoordinates(n); 
  return MeshAlgorithms::getEdgeCoordinates(*this, n);
}

const Point_View&
KOKKOS_INLINE_FUNCTION MeshCache::edge_coordinates(const Entity_ID n) const
{
  assert(edge_coordinates_cached); 
  return view<M>(edge_coordinates_)(n);  
}

// User 
Mesh<device>* m;

Kokkos::parallel_for("",size,KOKKOS_LAMBDA(const int c){     
  // Algo or Cached
  auto&& v = m->getEdgeCoordinates(c); 
  // Cached
  auto& v = m->edge_coordinates(c); 
});
#endif 

