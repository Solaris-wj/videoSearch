cmake_minimum_required(VERSION 2.6)

project(videosearch)

if(NOT CMAKE_BUIILD_TYPE)
	set(CMAKE_BUIILD_TYPE "release")
endif()

find_package(Boost REQUIRED)
find_package(Opencv REQUIRED)

include_directories(src)
add_subdirectory(src)
add_subdirectory(test)