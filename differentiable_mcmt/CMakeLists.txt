cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(CMAKE_CUDA_ARCHITECTURES 80)

project(fast_MCMT LANGUAGES CXX CUDA)
 
SET(SRCS 
	fast_mcmt.cpp
	fast_mcmt.hpp
  kdtree.hpp
  nanoflann.hpp
  KDTreeVectorOfVectorsAdaptor.hpp
  )

add_library(fast_mcmt STATIC ${SRCS})
add_executable(fastCVT 	main.cpp sdfs.hpp)
 

find_package(TBB REQUIRED)
#add ThirdPartyInclude
list( APPEND ThirdPartIncludePath "/home/jeeva.selvam/geogram/src/lib")
list( APPEND ThirdPartIncludePath "/home/jeeva.selvam/geogram/src/lib/third_party")
target_include_directories(fast_mcmt PUBLIC ${ThirdPartIncludePath})
target_include_directories(fast_mcmt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
message(STATUS "ThirdPartIncludePath: ${ThirdPartIncludePath}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
#add ThirdPartylib
list( APPEND ThirdPartLibPath "/home/jeeva.selvam/geogram/build/Linux64-gcc-dynamic-Release/lib/libgeogram.so")
target_link_libraries(fast_mcmt PUBLIC ${ThirdPartLibPath} TBB::tbb)
target_link_libraries(fastCVT PUBLIC fast_mcmt)


set_property(TARGET fastCVT 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

 
