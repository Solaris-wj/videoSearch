

add_executable(test_search test_search.cpp)
add_dependencies(test_search video_search)
target_link_libraries(test_search video_search)

add_executable(test_keyframe_exactor test_keyframe_exactor.cpp)
add_dependencies(test_keyframe_exactor video_search)
target_link_libraries(test_keyframe_exactor video_search)

add_executable(test_preproc test_preproc.cpp)
add_dependencies(test_preproc video_search)
target_link_libraries(test_preproc video_search)




