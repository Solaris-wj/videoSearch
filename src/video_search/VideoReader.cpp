
#include "VideoReader.h"

#include <thread>
#include <functional>

using namespace std;
using namespace cv;

namespace vs
{

    VideoReader::VideoReader(std::string videoName, float usedFps) : usedFps_(usedFps)
    {
        if (!cap.open(videoName))
        {
            isOpened_ = false;
            return;
        }
        maxFmCnt_ = (int)cap.get(CV_CAP_PROP_FRAME_COUNT);
        if (maxFmCnt_ < 20)
        {
            isOpened_ = false;
            return;
        }
        isOpened_ = true;
        trueFps_ = cap.get(CV_CAP_PROP_FPS);
    }
    VideoReader::~VideoReader()
    {

    }

    void VideoReader::scaleFrame(Mat &img, Mat & out, int maxFrameSize)
    {
        int r, c;
        if (img.rows > img.cols)
        {
            r = maxFrameSize;
            c = (float)maxFrameSize / img.rows * img.cols;
        }
        else
        {
            c = maxFrameSize;
            r = (float)maxFrameSize / img.cols * img.rows;
        }
        resize(img, out, Size(c, r));
    }

    //return absolute frame in the video 
    int VideoReader::getFrame(cv::Mat &frame, int fmIndex)
    {
        if (!isOpened())
            return -1;
        //read frame by index
        //fmIndex = fmIndex * trueFps_ / usedFps_;
        if (fmIndex < 0 || fmIndex >= maxFmCnt_ / trueFps_*usedFps_)
            return -1;
        cap.set(CV_CAP_PROP_POS_FRAMES, fmIndex / usedFps_ *trueFps_);


        bool ret = cap.read(frame);
        if (ret == true)
        {
            return fmIndex;
        }
        else
        {
            return -1;
        }
    }
    //return absolute frame in the video 
    int VideoReader::getFrame(cv::Mat &frame)
    {
        if (!isOpened())
            return -1;

        //read frame in order 
        while (cap.read(videoBuffer))
        {
            if (next_pos_++ % (int)(trueFps_ / usedFps_) == 0)
            {
                videoBuffer.copyTo(frame);
                return (next_pos_-1)/trueFps_/usedFps_;
            }
            //next_pos_++;
        }

        return -1;
    }
}