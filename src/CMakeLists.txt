
set(videosearch_src 
    VideoReader.cpp
    colorFeat.cpp
    Preprocessor.cpp
    VideoSearchParam.cpp
    DataTable.cpp
    KeyFrameExactor.cpp
    VideoIndexEngine.cpp
    FeatExactor.cpp
    impl.cpp
    )


include_directories(flann)
add_library(videosearch SHARED ${videosearch_src})

add_library(videosearch_jni SHARED ${videosearch_src})
