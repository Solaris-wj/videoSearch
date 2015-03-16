
set(videosearch_src *.cpp)

add_library(videosearch SHARED ${videosearch_src})

add_library(videosearch_jni SHARED ${videosearch_src})