#include "FeatExactor.h"
#include "VideoReader.h"
#include "Preprocessor.h"


using namespace std;
using namespace cv;

namespace vs
{
    extern clock_t start;
    int FeatExactor::exactFeatures(string &videoName, vector<KeyFrame> &keyFrames, 
                                    Mat &feat, vector<vector<KeyPoint>> & keys, vector<Mat> &desc)
    {
        start = clock();
        Preprocessor preProcessor;
        Mat mask;
        if (preProcessor.getDefaultMask(videoName, mask) == -1)
            return -1;
        VideoReader::scaleFrame(mask, mask, param_.maxFrameSize);

        printf("getdefault mask time=%lf s\n", (double)(clock() - start) / CLOCKS_PER_SEC);

        start = clock();
        kfExactor.exact(videoName, mask, keyFrames);
        printf("video kf num=%d\n", keyFrames.size());
        printf("video exact kf time=%lf s\n", (double)(clock() - start) / CLOCKS_PER_SEC);

        VideoReader videoReader(videoName);

        if (!videoReader.isOpened())
            return -1;
        Mat frame;

        start = clock();

        vector<KeyFrame> filtered_shots;
        filtered_shots.reserve(keyFrames.size());
        for (size_t i = 0; i != keyFrames.size(); ++i)
        {
  
            if (-1 == videoReader.getFrame(frame, keyFrames[i].key_fmIndex_))
                    break;
                VideoReader::scaleFrame(frame, frame, param_.maxFrameSize);

                Mat hist;
                ccvExactor.computeFeat(frame, mask, hist, COLOR_SPACE_BGR, cv::NORM_L1);

                vector<KeyPoint> kp;
                Mat ds;
                orb_detector(frame, mask, kp, ds);
                if (kp.size() < param_.orbThres)
                    continue;
                //filtered_shots.push_back(videoShots[i]);
                filtered_shots.push_back(keyFrames[i]);
                keys.push_back(kp);
                desc.push_back(ds);
                feat.push_back(hist);


        }
        keyFrames.swap(filtered_shots);

        printf("exact video kf feat time = %lf s\n", (double)(clock() - start) / CLOCKS_PER_SEC);
        return 0;
    }

}