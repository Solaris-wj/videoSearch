#ifndef KEY_FRAME_EXACTOR_H_H
#define KEY_FRAME_EXACTOR_H_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

#include "index_sort.h"
#include "colorFeat.h"


namespace vs
{
    struct KeyFrame
    {
        KeyFrame(){}
        KeyFrame(int kf_ind, int start, int end) :key_fmIndex_(kf_ind), start_index_(start), end_index_(end)
        {}
        int key_fmIndex_;
        int start_index_;
        int end_index_;
    };

    class KeyFrameExactor
    {
    private:
        const int MINIMUM_FRAME_IN_SHOT = 2;
        float  retainedVariance_ ;
        float usedFps_ ;
        float colorThres_ = 0.9f;
        ColorCoherenceVec featExactor;
        //ColorHist featExactor;
    public:

        //KeyFrameExactor(){};
        KeyFrameExactor(float usedFps, float retainedVaiance=0.99) :usedFps_(usedFps),retainedVariance_(retainedVaiance){};
        
        int exact(std::string videoName, cv::Mat &mask, std::vector<KeyFrame> & keyFrames);
        float getRetainedVariance()
        {
            return retainedVariance_;
        }
    protected:
        void kfDetection(cv::Mat &feat, std::vector<int> &frame_index, std::vector<KeyFrame> & keyFrames);
        int PCAExact(cv::Mat  &src, std::vector<int> &indices);
    };



}


#endif