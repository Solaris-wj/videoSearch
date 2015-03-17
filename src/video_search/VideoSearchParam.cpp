
#include "VideoSearchParam.h"
#include <iostream>

#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

namespace vs
{
    using std::string;
    using std::cout;

    using boost::property_tree::ptree;
    using boost::filesystem::path;

    int VideoSearchParam::loadFromFile(string configFilePath)
    {
        if (configFilePath.empty())
        {
            cout << "Video_det_param::loadFromFile:""未指定 配置文件";

            return -1;
        }

        ptree pt;
        read_xml(configFilePath, pt, boost::property_tree::xml_parser::trim_whitespace);

        try
        {
            maxFrameSize = pt.get<int>("config.maxFrameSize");
            if (maxFrameSize <300 && maxFrameSize>1000)
            {
                maxFrameSize = 500;
            }
            colorThres = pt.get<float>("config.colorThres");
            if (colorThres<0.0 || colorThres>1.0f)
            {
                colorThres = 0.7f;
            }

            usedFps = pt.get<float>("config.used_fps");
            if (usedFps > 10 || usedFps<0.2)
            {
                usedFps = 1;
            }

            orbThres = pt.get<float>("config.orbThres");

            if (orbThres > 200 || orbThres < 10)
            {
                orbThres = 50;
            }
            timeInterval = pt.get<int>("config.timeInterval");
            if (timeInterval>10 || timeInterval < 1)
            {
                timeInterval = 5;
            }
        }
        catch (boost::property_tree::ptree_error &e)
        {
            cout << string("Video_det_param::loadFromFile: 读取检测算法参数错误！") + e.what();
            return -1;
        }

        return 0;
    }
}