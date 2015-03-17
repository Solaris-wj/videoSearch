#ifndef VIDEO_BUFFER_H_H
#define VIDEO_BUFFER_H_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include <thread>
#include <mutex>
#include <atomic>
#include <memory>
#include <condition_variable>

#include "defines.h"

namespace vs
{
    class VS_EXPORTS VideoReader
    {
    private:         
        float trueFps_;
        float usedFps_ = 1.0f;
        size_t next_pos_=0;

        cv::VideoCapture cap;
        bool isOpened_=false;
        int maxFmCnt_;
        cv::Mat videoBuffer;
    public:
        VideoReader(std::string videoName, float usedFps=1.0f);
        ~VideoReader();

        bool isOpened()
        {
            return isOpened_;
        }
        int getFrame(cv::Mat &frame);
        int getFrame(cv::Mat &frame, int fmIndex);
        static void scaleFrame(cv::Mat &img, cv::Mat & out, int maxFrameSize);

    };
}
#endif