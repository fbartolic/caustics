cmake_minimum_required(VERSION 3.12...3.18)
project(caustics LANGUAGES CXX)

# Need C++14 support
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

message(STATUS "Using CMake version " ${CMAKE_VERSION})

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

include_directories("lib/extern/cub-1.15.0")
include_directories("lib/extern/thrust-1.15.0")

# CPU op library
pybind11_add_module(ehrlich_aberth_cpu_op ${CMAKE_CURRENT_LIST_DIR}/lib/ehrlich_aberth/cpu_ops.cc)
pybind11_add_module(integrate_image_cpu_op ${CMAKE_CURRENT_LIST_DIR}/lib/integrate_image/cpu_ops.cc)
install(TARGETS ehrlich_aberth_cpu_op DESTINATION caustics) # caustics is directory
install(TARGETS integrate_image_cpu_op DESTINATION caustics) 


if (CAUSTICS_CUDA)
  enable_language(CUDA)
  include_directories("/usr/local/cuda-11/include")
  include_directories("/usr/local/cuda-11.2/include")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
  
  pybind11_add_module(
    ehrlich_aberth_gpu_op
    ${CMAKE_CURRENT_LIST_DIR}/lib/ehrlich_aberth/kernels.cc.cu
    ${CMAKE_CURRENT_LIST_DIR}/lib/ehrlich_aberth/gpu_ops.cc)

  pybind11_add_module(
    integrate_image_gpu_op
    ${CMAKE_CURRENT_LIST_DIR}/lib/integrate_image/kernels.cc.cu
    ${CMAKE_CURRENT_LIST_DIR}/lib/integrate_image/gpu_ops.cc)

  install(TARGETS ehrlich_aberth_gpu_op DESTINATION caustics)
  install(TARGETS integrate_image_gpu_op DESTINATION caustics)
else()
  message(STATUS "Building without CUDA")
endif()
