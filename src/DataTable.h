#ifndef DATA_BASE_H_H
#define DATA_BASE_H_H

#include <string>
#include <vector>
#include <memory>
#include <numeric>
#include <mutex>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <vector>
#include <tuple>

#include "flann/flann.hpp"
#include "KeyFrameExactor.h"
#include "FeatExactor.h"


#include <boost/serialization/serialization.hpp>

namespace vs
{
    const char PATH_SEPARATOR[] = "/";
    const char DATA_FILE_EXT[] = ".dat";
    class VideoIndexEngine;
    const int PATH_BUF_SIZE = 128;
    //typedef ::flann::Index<::flann::Hamming<uchar>> DescIndex;
    class DataTable
    {        
        typedef ::flann::MultiThreadIndex<::flann::L1<float>> FrameIndex;
        friend boost::serialization::access;
    private:
        FeatExactor featExactor_;
        std::shared_ptr<FrameIndex> frameIndex_;

        //path to save the data into
        std::string dataPath_;

        //full name
        //std::vector<std::string> videoPaths_;
        //std::vector<std::string> videoDataPaths_;
        //std::list<std::string> videoPaths_;
        //std::list<std::string> videoDataPaths_;

        //map of video name to video id and video data path
        std::unordered_map<std::string, size_t> vn2vId_;

        std::unordered_map<size_t, std::string> vId2vDataPath_;
        //one global feature of every video frames         
        std::unordered_map<size_t , std::shared_ptr<cv::Mat>> videoFrameFeats_;
        //map video id to global frame id
        std::unordered_map < size_t, std::vector<size_t>> vId2gFmId_;
        //map global frame id to video id, used in video search phrase (vote for videos)
        std::unordered_map<size_t, size_t> gFmId2VId_;
        //map global frame id to frame local id (frame index in corresponding video), used in video ransac function
        std::unordered_map<size_t, size_t> gFmId2lFmId_;


        //frame counts of every videos
        std::vector<int> frameCnts_;   
        bool isChanged = false;
    public:
        DataTable(){};
        explicit DataTable(std::string dataPath);
        explicit DataTable(const DataTable &);
        ~DataTable();
        //swap assignment 
        DataTable &operator=(DataTable &);
        bool find(std::string videoName);
//         int insertVideo(const std::string videoName, const std::shared_ptr<std::vector<KeyFrame>> keyFrames,
//                          const std::shared_ptr<cv::Mat> feat, const std::shared_ptr<std::vector<std::vector<cv::KeyPoint>>> keypoints, 
//                          const std::shared_ptr<std::vector<cv::Mat>> desc, const std::shared_ptr<DescIndex> descIndex);

        int insertVideo(const std::string videoName);

        void deleteVideo(std::string videoName);
        void deleteVideos(std::vector<std::string> videoNames);
        //return valid video number
        int size();
        int getVideoFmCnt(int vid);
        int gFmInd2Vid(int gFmInd)
        {
            return gFmId2VId_.at(gFmInd);
        }
        int gFmInd2LFmInd(int gFmInd)
        {
            return gFmId2lFmId_.at(gFmInd);
        }
        const std::string& getVideoName(int vid)
        {
            return videoPaths_[vid];
        }

        void getVideoData(int vid, std::vector<KeyFrame> &kfm, std::vector<std::vector<cv::KeyPoint>> &keys, std::vector<cv::Mat> &desc);

//         std::vector<std::vector<cv::KeyPoint>> & getVideoKeyPoints(int vid);
//         std::vector<cv::Mat> & getVideoDesc(int vid);
//         std::vector<KeyFrame>& getVideoKeyFrames(int vid);

        std::vector<std::shared_ptr<cv::Mat>>& getVideoFrameFeat();
        
    protected:
        friend class VideoIndexEngine;
        void insertVideoGlobalInfo(std::string videoName);
        void swap(DataTable &other);
        void save(std::ostream & outStream);
        int load(std::istream & inStream);
    };

    inline int DataTable::size()
    {
        return videoPaths_.size();
    }
    

}

#endif
