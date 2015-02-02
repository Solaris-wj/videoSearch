#ifndef VideoIndex_ENGINE_H_H
#define VideoIndex_ENGINE_H_H

#include <flann/flann.hpp>

#include "NonCopyable.h"
#include "DataTable.h"
#include "FeatExactor.h"
#include "VideoSearchParam.h"

#include "MultiThreadIndex.h"

namespace vs
{
    class VideoIndexEngine:public NonCopyable
    {
        typedef ::flann::MultiThreadIndex<::flann::L1<float>> FrameIndex;
    private:
        std::string indexDataPath_;
        std::string videoDataPath_;
        //std::string algoConfFilePath;
        VideoSearchParam param_;
        DataTable videoTable_;
        FeatExactor featExactor_;
        std::shared_ptr<FrameIndex> frameIndex_;        
    public:
        //VideoIndexEngine(std::string root="./");
        VideoIndexEngine(std::string indexDataPath, std::string videoDataPath, std::string algoConfFilePath);
        ~VideoIndexEngine();
        int addVideo(std::vector<std::string> &videoNames);
        int addVideo(std::string &videoName);
        int searchVideo(std::string &videoName, std::vector<std::string> &jsonResult);
        int deleteVideo(std::string &videoName);
        int deleteVideos(std::vector<std::string> &videoNames);
    private:
        void assembleMatches(cv::SparseMat &flagMat, int beg0, int end0, int beg1, int end1
                        , std::vector<std::pair<int, int>> &matches0, std::vector<std::pair<int, int>> &matches1);
        std::tuple<int, int> go_bottom_right(cv::Mat &recordMap, cv::SparseMat & flagMat, int r, int c, int interval);

        int genJsonStr(std::string &targetVideoName, std::vector<std::pair<int, int>> & finalMatchResult_det,
                       std::vector<std::pair<int, int>> &finalMatchResult_tar, std::string &strJsonResult);
        void videoRansac(std::vector<std::vector<cv::KeyPoint>> &keys1, std::vector<cv::Mat> desc1, std::vector<std::vector<cv::KeyPoint>> &keys2,
                         std::vector<cv::Mat> desc2, std::vector<cv::DMatch> &matches, std::vector<cv::DMatch> &fliteredMatches);
        int ransac(std::vector<cv::KeyPoint> &keys1, std::vector<cv::KeyPoint> &keys2, std::vector<cv::DMatch> &matches, /*vector<DMatch> &fined_matches,*/ int threshold, int iteration);

    };
}

#endif