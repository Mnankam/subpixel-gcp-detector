cmake_minimum_required(VERSION 3.22)

project(subpixel_gcp
  VERSION 0.1.0
  DESCRIPTION "Robust sub-pixel detection of checkerboard ground control points"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(GCP_BUILD_TESTS          "Build unit tests"                     ON)
option(GCP_BUILD_APPS           "Build command line tools"             ON)
option(GCP_WARNINGS_AS_ERRORS   "Treat compiler warnings as errors"    OFF)
option(GCP_ENABLE_SANITIZERS    "Enable ASan + UBSan (Debug builds)"   OFF)
option(GCP_WITH_PYLON           "Basler pylon camera support (needs pylon SDK)" OFF)
option(GCP_WITH_GUI             "Live preview window via OpenCV highgui" OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
if(GCP_WITH_GUI)
  find_package(OpenCV REQUIRED COMPONENTS highgui)
endif()
if(GCP_WITH_PYLON)
  # The pylon installer provides pylonConfig.cmake (target pylon::pylon).
  # Windows: set CMAKE_PREFIX_PATH to "C:/Program Files/Basler/pylon/Development/CMake"
  # if it is not found automatically.
  find_package(pylon REQUIRED)
endif()

# ---------------------------------------------------------------------------
# Common compile options
# ---------------------------------------------------------------------------
add_library(gcp_options INTERFACE)
target_compile_options(gcp_options INTERFACE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wshadow>
  $<$<AND:$<CXX_COMPILER_ID:GNU,Clang>,$<BOOL:${GCP_WARNINGS_AS_ERRORS}>>:-Werror>)
if(GCP_ENABLE_SANITIZERS)
  target_compile_options(gcp_options INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(gcp_options INTERFACE -fsanitize=address,undefined)
endif()

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
add_library(gcp
  src/saddle_refiner.cpp
  src/target_detector.cpp
  src/synthetic.cpp
  src/synthetic_source.cpp
  src/tracker.cpp)
add_library(gcp::gcp ALIAS gcp)

target_include_directories(gcp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(gcp
  PUBLIC  opencv_core opencv_imgproc
  PRIVATE Eigen3::Eigen gcp_options)

# ---------------------------------------------------------------------------
# I/O library: image files and cameras (optional Basler pylon / GenICam)
# ---------------------------------------------------------------------------
add_library(gcp_io
  src/io/image_file_source.cpp
  src/io/pylon_source.cpp)
add_library(gcp::io ALIAS gcp_io)
target_link_libraries(gcp_io PUBLIC gcp PRIVATE opencv_imgcodecs gcp_options)
if(GCP_WITH_PYLON)
  target_link_libraries(gcp_io PRIVATE pylon::pylon)
  target_compile_definitions(gcp_io PRIVATE GCP_HAVE_PYLON)
endif()

# ---------------------------------------------------------------------------
# Applications
# ---------------------------------------------------------------------------
if(GCP_BUILD_APPS)
  add_executable(gcp_detect apps/gcp_detect.cpp)
  target_link_libraries(gcp_detect PRIVATE gcp opencv_imgcodecs gcp_options)

  add_executable(gcp_eval apps/gcp_eval.cpp)
  target_link_libraries(gcp_eval PRIVATE gcp opencv_imgcodecs gcp_options)

  add_executable(gcp_live apps/gcp_live.cpp)
  target_link_libraries(gcp_live PRIVATE gcp_io opencv_imgcodecs opencv_imgproc gcp_options)
  if(GCP_WITH_GUI)
    target_link_libraries(gcp_live PRIVATE opencv_highgui)
    target_compile_definitions(gcp_live PRIVATE GCP_HAVE_HIGHGUI)
  endif()

  install(TARGETS gcp_detect gcp_eval gcp_live RUNTIME DESTINATION bin)
endif()

install(TARGETS gcp gcp_io ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(GCP_BUILD_TESTS)
  enable_testing()
  find_package(GTest QUIET)
  if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()
  add_subdirectory(tests)
endif()
