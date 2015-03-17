

#include "../video_search//KeyFrameExactor.h"
#include <iostream>
#include <string>
#include <vector>
#include <time.h>

#include <opencv2/opencv.hpp>

#include "../video_search/Preprocessor.h"
#include "../video_search/VideoReader.h"

using namespace std;
using namespace cv;

int main()
{
    string videoName = "D:\\test_video\\MPEG2_Jan07_185021_0.mpg";

    vs::Preprocessor preProcessor;
    Mat mask;
    if (preProcessor.getDefaultMask(videoName, mask) == -1)
        return -1;
    vs::VideoReader::scaleFrame(mask, mask, 500);
    vs::KeyFrameExactor kfExactor(1);

    cout << kfExactor.getRetainedVariance() << endl;
    vector<vs::KeyFrame> kf_indices;

    clock_t start;
    start = clock();
    kfExactor.exact(videoName, mask, kf_indices);

    printf("kf time=%lf s\n", (double)(clock() - start) / CLOCKS_PER_SEC);
    vs::VideoReader VideoReader(videoName);

    printf("kf num = %d \n", kf_indices.size());

    Mat frame;
    for (size_t i = 0; i != kf_indices.size(); ++i)
    {
        VideoReader.getFrame(frame,kf_indices[i].key_fmIndex_);
        char fileName[200] = { 0 };
        sprintf(fileName, "%s_keyframe_%d.jpg", videoName.c_str(), kf_indices[i].key_fmIndex_);
        imwrite(fileName, frame);
    }
}