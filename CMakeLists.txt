cmake_minimum_required(VERSION 2.6)

project(videosearch C CXX)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release")
endif()

if(WIN32)
	add_definitions(-DWIN32)
endif()

find_package(OpenCV REQUIRED)

#boost
if(WIN32)
include_directories(C:/local/boost_1_57_0)
link_directories(C:/local/boost_1_57_0/lib64-msvc-12.0)
else()
find_package(Boost REQUIRED)
add_definitions(-std=c++11)
set(CMAKE_CXX_COMPILER clang++)
endif()


#jni header files
include_directories(C:/Program\ Files/Java/jdk1.7.0_75/include)
include_directories(C:/Program\ Files/Java/jdk1.7.0_75/include/win32)
#include_directories(src)


add_subdirectory(src)
add_subdirectory(test)
