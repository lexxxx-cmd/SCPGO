cmake_minimum_required(VERSION 3.16)
project(SCPGO VERSION 1.0.0 LANGUAGES CXX)

# C++17 required (std::optional, structured bindings, etc.)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -------------------- Dependencies --------------------
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters io registration segmentation octree search kdtree)
find_package(Ceres REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(GTSAM REQUIRED)
find_package(Boost REQUIRED COMPONENTS serialization thread timer chrono filesystem)
find_package(yaml-cpp REQUIRED)

# -------------------- CUDA imported targets (compatibility) --------------------
# PCL and other deps may reference CUDA:: targets that the old FindCUDA created
# but CUDAToolkit doesn't. Create them if missing.
foreach(_lib cublas cudart cusolver cusparse)
  if(NOT TARGET CUDA::${_lib})
    add_library(CUDA::${_lib} UNKNOWN IMPORTED)
    set_target_properties(CUDA::${_lib} PROPERTIES
      IMPORTED_LOCATION "${CUDAToolkit_LIBRARY_DIR}/lib${_lib}.so"
      INTERFACE_INCLUDE_DIRECTORIES "${CUDAToolkit_INCLUDE_DIRS}")
    message(STATUS "  -> Created CUDA::${_lib}")
  endif()
endforeach()

# -------------------- Executable --------------------
add_executable(alaserPGO
  src/laserPosegraphOptimization.cpp
  include/SCPGO/scancontext/Scancontext.cpp
)

target_include_directories(alaserPGO PRIVATE
  include
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_options(alaserPGO PRIVATE ${OpenMP_CXX_FLAGS})

target_link_libraries(alaserPGO PRIVATE
  ${PCL_LIBRARIES}
  ${CERES_LIBRARIES}
  gtsam
  yaml-cpp
  Boost::serialization
  Boost::thread
  Boost::timer
  Boost::chrono
  Boost::filesystem
  OpenMP::OpenMP_CXX
  CUDA::cudart
)
