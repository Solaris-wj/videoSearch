

#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/config.h)


set(videosearch_src 
	colorFeat.cpp
	DataTable.cpp	
	FeatExactor.cpp
	KeyFrameExactor.cpp
	Preprocessor.cpp
	VideoIndexEngine.cpp	
    VideoReader.cpp    
    VideoSearchParam.cpp      
    flann/ext/lz4.c
    flann/ext/lz4hc.c

	colorFeat.h
	DataTable.h
	defines.h
	FeatExactor.h
	index_sort.h
	KeyFrameExactor.h
	NonCopyable.h
	Preprocessor.h
	Serialization.h
	VideoIndexEngine.h
	VideoReader.h
	VideoSearchParam.h
    )

set(video_search_jni_src impl.cpp)

include_directories(D:/CodeCollection/javaCodes/videoSearchWS/jni)
#include_directories(flann)
add_library(video_search SHARED ${videosearch_src})
target_link_libraries(video_search ${OpenCV_LIBS})



add_library(video_search_jni  SHARED ${video_search_jni_src})
TARGET_LINK_LIBRARIES(video_search_jni video_search)
target_link_libraries(video_search_jni ${OpenCV_LIBS})