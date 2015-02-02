#ifndef FEAT_EXACTOR_H_H
#define FEAT_EXACTOR_H_H

#include <opencv2/opencv.hpp>

#include "colorFeat.h"
#include "KeyFrameExactor.h"
#include "VideoSearchParam.h"
namespace vs
{
    class FeatExactor
    {
    private:
        const int MAX_LOCAL_FEAT_NUM = 500;
        VideoSearchParam param_;
        KeyFrameExactor kfExactor;
        ColorCoherenceVec ccvExactor;
        //ColorHist ccvExactor;
        cv::ORB orb_detector;
        
    public:
        FeatExactor(VideoSearchParam &param) :param_(param),kfExactor(param_.usedFps),orb_detector(MAX_LOCAL_FEAT_NUM){}
        int exactFeatures(const std::string &videoName, std::vector<KeyFrame> &videoShots, cv::Mat &feat,
                          std::vector<std::vector<cv::KeyPoint>> &keys, std::vector<cv::Mat> &desc);
        int getMaxLocalFeatNum()
        {
            return MAX_LOCAL_FEAT_NUM;
        }

    };
}

#endif