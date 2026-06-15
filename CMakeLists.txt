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
find_package(nlohmann_json REQUIRED)
find_package(OpenSSL REQUIRED)

# websocketpp — header-only WebSocket library, auto-downloaded from GitHub
include(FetchContent)
FetchContent_Declare(
  websocketpp
  GIT_REPOSITORY https://github.com/jcelerier/websocketpp.git
  GIT_TAG        ossia/2024-12-18
)
FetchContent_MakeAvailable(websocketpp)

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
  src/websocket_publisher.cpp
  include/SCPGO/scancontext/Scancontext.cpp
  src/foxglove/base64.cpp
  src/foxglove/parameter.cpp
  src/foxglove/serialization.cpp
  src/foxglove/server_factory.cpp
)

target_include_directories(alaserPGO PRIVATE
  include
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
  ${websocketpp_SOURCE_DIR}
)

target_compile_options(alaserPGO PRIVATE
  ${OpenMP_CXX_FLAGS}
  $<$<CXX_COMPILER_ID:MSVC>:/utf-8>  # 强制 MSVC 用 UTF-8 编码，避免 GBK 误解析中文注释
  $<$<CXX_COMPILER_ID:MSVC>:/bigobj>  # websocketpp 模板展开导致节数超限
)

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
  nlohmann_json::nlohmann_json
  OpenSSL::SSL
  OpenSSL::Crypto
)
