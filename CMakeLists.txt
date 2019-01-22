cmake_minimum_required(VERSION 3.1)
project(untitled)

set(CMAKE_CXX_STANDARD 11)


find_package(PkgConfig)
pkg_check_modules(EIGEN eigen3)
include_directories(${EIGEN_INCLUDE_DIRS})

pkg_check_modules(BLAS blas)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(MKL)
include_directories(${MKL_INCLUDE_DIRS})

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_executable(untitled main.cpp)
message(${MKL_LIBRARIES})
target_link_libraries(untitled  -Wl,--start-group $ENV{MKLROOT}/lib/intel64/libmkl_intel_lp64.a $ENV{MKLROOT}/lib/intel64/libmkl_gnu_thread.a $ENV{MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp -lpthread -lm -ldl)