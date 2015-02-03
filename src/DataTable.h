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

#include <boost/filesystem.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/split_member.hpp>

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
        std::string indexDataFileName;
        std::string indexFileName;
        FeatExactor featExactor_;
        std::shared_ptr<FrameIndex> frameIndex_;

        //path to save the data into
        std::string dataPath_;
        //map of video name to video id and video data path
        std::unordered_map<std::string, size_t> vn2vid_;
        std::unordered_map<size_t, std::string> vid2vn_;
        std::unordered_map<size_t, std::string> vid2vDataPath_;
        //one global feature of every video frames         
        std::unordered_map<size_t , std::shared_ptr<cv::Mat>> videoFrameFeats_;
        //map video id to global frame id
        std::unordered_map < size_t, std::vector<size_t>> vid2gFmId_;
        //map global frame id to video id, used in video search phrase (vote for videos)
        std::unordered_map<size_t, size_t> gFmId2vid_;
        //map global frame id to frame local id (frame index in corresponding video), used in video ransac function
        std::unordered_map<size_t, size_t> gFmId2lFmId_;
        mutable bool isChanged = false;
    public:
        explicit DataTable(std::string dataPath, VideoSearchParam &params);
        explicit DataTable(const DataTable &);
        ~DataTable();
 
        
        int insertVideo(const std::string videoName);

        void deleteVideo(std::string videoName);
        void deleteVideos(std::vector<std::string> videoNames);
        void load();
        void save();
        //return valid video number
        
        FeatExactor & getFeatExactor()
        {
            return featExactor_;
        }
        FrameIndex & getFrameIndex()
        {
            return *frameIndex_.get();
        }
        int getVideoFmCnt(int vid)
        {
            return videoFrameFeats_.at(vid)->rows;
        }
        size_t gFmInd2Vid(int gFmInd)
        {
            return gFmId2vid_.at(gFmInd);
        }
        int gFmInd2LFmInd(int gFmInd)
        {
            return gFmId2lFmId_.at(gFmInd);
        }
        const std::string& getVideoName(int vid)
        {
            return vid2vn_.at(vid);
        }

        void getVideoData(int vid, std::vector<KeyFrame> &kfm, std::vector<std::vector<cv::KeyPoint>> &keys, std::vector<cv::Mat> &desc);

        std::vector<std::shared_ptr<cv::Mat>>& getVideoFrameFeat();
        BOOST_SERIALIZATION_SPLIT_MEMBER();
    protected:
        friend class VideoIndexEngine;
        void insertVideoGlobalInfo(std::string videoName);
        void swap(DataTable &other);
//         void save(std::ostream & outStream);
//         int load(std::istream & inStream);

        template<class Archive>
        void save(Archive & oar, const unsigned int version) const; 
        template<class Archive>
        void load(Archive & iar, const unsigned int version);

        void deleteVideoDataFrameDisk(size_t vid);

        std::string getVideoDataPath(std::string videoPath)
        {
            boost::filesystem::path pt(videoPath);
            std::string videoDataPath = dataPath_ + PATH_SEPARATOR + pt.filename().string() + "videodata" + DATA_FILE_EXT;
            return videoDataPath;
        }
       

    };
  

}

#endif
