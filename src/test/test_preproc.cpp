
#include "../video_search//Preprocessor.h"

#include <iostream>
#include <string>
#include <vector>
#include <time.h>

#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;

int main()
{
    string videoName = "D:\\videos\\test\\det.mp4";

    VideoCapture cap(videoName);
    if (!cap.isOpened())
    {
        printf("can not open file %s", videoName.c_str());
        return 0;
    }

    int width = (int)cap.get(CV_CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT);

    //int left, right, top, bottom;

    vs::Preprocessor pre;

    Mat mask;
    pre.getDefaultMask(videoName, mask);

    vector<Mat> vec{ mask, mask, mask };
    Mat colorMask;
    cv::merge(vec,colorMask);

    namedWindow("frame");
    namedWindow("valid");
    Mat frame;
    while (true)
    {
        clock_t tt = clock();
        if (!cap.read(frame))
            break;

        Mat valid = frame & colorMask;
        imshow("valid", valid);

        imshow("frame", frame);

        
        if ('q' == waitKey(5))
            break;

        //printf("time = %lf \n ", (double)(clock() - tt) / CLOCKS_PER_SEC);
    }
}