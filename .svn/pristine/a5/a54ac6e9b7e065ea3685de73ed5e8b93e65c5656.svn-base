#ifndef VIDEO_SEARCH_PARAM_H_H
#define VIDEO_SEARCH_PARAM_H_H

#include <string>

namespace vs
{
    class VideoSearchParam
    {
    public:
        VideoSearchParam()
        {
        }
        VideoSearchParam(std::string configFilePath)
        {
            loadFromFile(configFilePath);
        }
    public:
        int loadFromFile(std::string configFilePath);
        int maxFrameSize=500;
        int timeInterval=5;
        float colorThres=0.7;
        float orbThres=50;
        float usedFps=1;
    };
}
#endif
