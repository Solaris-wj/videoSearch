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

#jni header files
include_directories(C:/Program\ Files/Java/jdk1.7.0_75/include)
include_directories(C:/Program\ Files/Java/jdk1.7.0_75/include/win32)

else()
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

message(STATUS ${Boost_LIBRARY_DIRS})
message(STATUS ${Boost_LIBRARIES})

add_definitions(-std=c++11)
set(CMAKE_CXX_COMPILER clang++)

#jni header files
include_directories(/home/ufo/programs/jdk1.8.0_25/include)
include_directories(/home/ufo/programs/jdk1.8.0_25/include/linux)

endif()






macro(source_group_by_dir source_files)
    if(WIN32)
        set(sgbd_cur_dir ${CMAKE_CURRENT_SOURCE_DIR})
        foreach(sgbd_file ${${source_files}})			
			string(REPLACE ${sgbd_cur_dir}/  "" groupName1 ${sgbd_file})			
			string(REGEX REPLACE  \(/?[^/]+$\) "" groupName2  ${groupName1})			
			#组名不空
			if(groupName2)
				string(REPLACE "/" "\\\\" groupName3 ${groupName2})
				#message(STATUS ${groupName3})
			else()  #组名为空串
				#头文件
				string(REGEX MATCH \(\(\\.hpp\)|\(\\.h\)\)$  ext1  ${groupName1} )				
				string(REGEX MATCH \(\(\\.c\)|\(\\.cpp\)|\(\\.cpp\)\)$ ext2 ${groupName1})				
				if(ext1)
					set(groupName3 "Z-Header Files")
				elseif(ext2)
				#source file
					set(groupName3 "Z-Source Files")				
				else()
					set(groupName3 "ZZ-Misc Files")
				endif()				
			endif()					
            source_group(${groupName3} FILES ${sgbd_file})
        endforeach(sgbd_file)
    endif(WIN32)
endmacro(source_group_by_dir)



add_subdirectory(src)




