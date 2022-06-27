#pragma once

#include "Kokkos_Core.hpp"
#include "Kokkos_DualView.hpp"

using Entity_ID = int;
enum memory {host, device};
using Coordinate = std::array<double,3>;


template<memory mem = device>
struct Space { using execution_space = Kokkos::DefaultExecutionSpace; };
template<>
struct Space<host> { using execution_space = Kokkos::HostSpace; };


// generic "viewer"
template<memory MEM, typename DV_type>
auto&
view(DV_type& dv) {
  if constexpr (MEM == device){
    return dv.d_view;
  } else {
    return dv.h_view;
  }
}




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

  int size() const { return row_map.extent(0)-1; }

  // template<memory MEM = device>
  // T& get(int i, int j) {
  //   if constexpr (MEM == device) {
  //     return entries.view_device()[row_map[i]+j];
  //   } else {
  //     return entries.view_host()[row_map[i]+j];
  //   }
  // }

  // template<memory MEM = device>
  // const T& get(int i, int j) const {
  //   if constexpr (MEM == device) {
  //     return entries.view_device()[row_map[i]+j];
  //   } else {
  //     return entries.view_host()[row_map[i]+j];
  //   }
  // }

};


template<typename T, typename execution_space>
struct Crs {
  Kokkos::View<int*, execution_space> row_map;
  Kokkos::View<T, execution_space> entries;
  int size() const { return row_map.size()-1; }
};

