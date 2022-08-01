cmake_minimum_required(VERSION 3.10)
project(mesh_prototype)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED)
find_package(Kokkos REQUIRED)
find_package(CUDA REQUIRED)

#add_executable(mesh_DV_constexpr mesh_DV_constexpr.cc)
#target_link_libraries(mesh_DV_constexpr PUBLIC Kokkos::kokkoscore)

#add_executable(mesh_DV_layer mesh_DV_layer.cc)
#target_link_libraries(mesh_DV_layer PUBLIC Kokkos::kokkoscore)

#add_executable(test_perf test_perf.cc)
#target_link_libraries(test_perf PUBLIC Kokkos::kokkoscore)

add_executable(mesh mesh_prototype.cc)
target_link_libraries(mesh PUBLIC Kokkos::kokkoscore)
