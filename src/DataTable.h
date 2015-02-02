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

#include <opencv2/opencv.hpp>
#include <vector>
#include <flann/flann.hpp>
#include "KeyFrameExactor.h"


namespace vs
{
    const char PATH_SEPARATOR[] = "/";
    const char DATA_FILE_EXT[] = ".dat";
    class VideoIndexEngine;
    const int PATH_BUF_SIZE = 128;
    typedef ::flann::Index<::flann::Hamming<uchar>> DescIndex;
    class DataTable
    {        
        friend boost::serialization::access;
    private:
        //path to save the data into
        std::string dataPath_;
        //std::vector<bool> deletedFlags_;
        //full name
        std::vector<std::string> videoNames_;
        std::vector<std::string> videoDataPaths_;
        //map of video name to video id
        std::unordered_map<std::string, int> vn2vid_;
        //one global feature of every video frames
        std::vector<std::shared_ptr<cv::Mat>> frameFeats_;
        //std::unordered_map<std
        //frame counts of every videos
        std::vector<int> frameCnts_;        
        //global frame index to video id map
        std::vector<int> globalFmInd2Vid_;
        //global frame index to local frame index (frame index in corresponding video)
        std::vector<int> globalFmInd2LocalFmInd_;

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

        int insertVideo(const std::string videoName, const std::shared_ptr<std::vector<KeyFrame>> keyFrames,
                        const std::shared_ptr<cv::Mat> feat, const std::shared_ptr<std::vector<std::vector<cv::KeyPoint>>> keypoints,
                        const std::shared_ptr<std::vector<cv::Mat>> desc);

        void deleteVideo(std::string videoName);
        void deleteVideos(std::vector<std::string> videoNames);
        //return valid video number
        int size();
        int getVideoFmCnt(int vid);
        int gFmInd2Vid(int gFmInd);
        int gFmInd2LFmInd(int gFmInd);
        const std::string& getVideoName(int vid)
        {
            return videoNames_[vid];
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
        void save(ostream & outStream);
        int load(istream & inStream);
    };

    inline int DataTable::size()
    {
        return videoNames_.size();
    }
    //inline bool DataTable::find(std::string videoName)
    //{
    //    return vn2vid_.find(videoName) != vn2vid_.end();
    //}       

}

#endif
