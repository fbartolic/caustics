cmake_minimum_required(VERSION 3.12...3.18)

find_package(Enzyme REQUIRED)


message("LLVM Version: " ${Enzyme_LLVM_VERSION_MAJOR})
message("Found LLVM at: " ${Enzyme_LLVM_BINARY_DIR})

# C and C++ Compilers can only be configured before project(...)
set(CMAKE_C_COMPILER "${Enzyme_LLVM_BINARY_DIR}/bin/clang")
set(CMAKE_CXX_COMPILER "${Enzyme_LLVM_BINARY_DIR}/bin/clang")

project(caustics LANGUAGES CXX)

find_package(Enzyme REQUIRED COMPONENTS ClangEnzyme)
get_target_property(ClangEnzyme ClangEnzyme-${Enzyme_LLVM_VERSION_MAJOR} LOCATION)
message("Found Enzyme at: " ${ClangEnzyme})


# Need C++14 support
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

message(STATUS "Using CMake version " ${CMAKE_VERSION})

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

include_directories("lib/extern/cub-1.15.0")
include_directories("lib/extern/thrust-1.15.0")
include_directories("/usr/local/cuda-11.2/include")

# CPU op library
pybind11_add_module(ehrlich_aberth_cpu_op ${CMAKE_CURRENT_LIST_DIR}/lib/ehrlich_aberth/cpu_ops.cc)
install(TARGETS ehrlich_aberth_cpu_op DESTINATION caustics) # caustics is directory

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-experimental-new-pass-manager -Xclang -load -Xclang ${ClangEnzyme}  -O2 --gcc-toolchain=/share/apps/gcc112/ -lstdc++")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-experimental-new-pass-manager -Xclang -load -Xclang ${ClangEnzyme}")


pybind11_add_module(integrate_image_cpu_op ${CMAKE_CURRENT_LIST_DIR}/lib/integrate_image/cpu_ops.cc)
install(TARGETS integrate_image_cpu_op DESTINATION caustics) # caustics is directory


if (CAUSTICS_CUDA)
#  enable_language(CUDA)
#  include_directories("/usr/local/cuda-11/include")
#  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
#  
#  pybind11_add_module(
#    ehrlich_aberth_gpu_op
#    ${CMAKE_CURRENT_LIST_DIR}/lib/ehrlich_aberth/kernels.cc.cu
#    ${CMAKE_CURRENT_LIST_DIR}/lib/ehrlich_aberth/gpu_ops.cc)
#
#  install(TARGETS ehrlich_aberth_gpu_op DESTINATION caustics)
else()
  message(STATUS "Building without CUDA")
endif()
