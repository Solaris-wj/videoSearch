#ifndef PREPROCESSOR_H_H
#define PREPROCESSOR_H_H

#include <string>

#include <opencv2/opencv.hpp>

namespace vs
{
    using namespace cv;
    using namespace std;
    class Preprocessor
    {
    protected:
        float top_;
        float bottom_;
        float left_;
        float right_;
    public:
        Preprocessor() : top_(0.15f), bottom_(0.25f),left_(0.1f), right_(0.1f)
        {};
        Preprocessor(float top, float bottom, float left, float right)
            :top_(top), bottom_(bottom), left_(left), right_(right)
        {};
        int getDefaultMask(string videoName,Mat &mask );
    
    protected:
        tuple<int, int, int, int> eraseBorder(VideoCapture &cap);
    };

}


#endif