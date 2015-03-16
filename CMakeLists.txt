cmake_minimum_required(VERSION 2.6)

project(videosearch)

if(NOT CMAKE_BUIILD_TYPE)
	set(CMAKE_BUIILD_TYPE "release")
endif()

find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(src)

set (CMAKE_CXX_COMPILER clang++)
add_definitions(-std=c++11)
add_subdirectory(src)
add_subdirectory(test)
