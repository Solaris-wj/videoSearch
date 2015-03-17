

file(GLOB_RECURSE ALL_FILES *)

source_group_by_dir(ALL_FILES)


include_directories(D:/CodeCollection/javaCodes/videoSearchWS/jni)

add_library(video_search_jni  SHARED ${ALL_FILES})
TARGET_LINK_LIBRARIES(video_search_jni video_search)
target_link_libraries(video_search_jni ${OpenCV_LIBS})