



file(GLOB_RECURSE ALL_FILES *)

source_group_by_dir(ALL_FILES)


add_library(video_search SHARED ${ALL_FILES})
target_link_libraries(video_search ${OpenCV_LIBS})
target_link_libraries(video_search boost_serialization boost_filesystem boost_system)
