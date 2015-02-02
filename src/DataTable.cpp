#include "DataTable.h"
#include <boost/filesystem.hpp>


#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/filesystem/path.hpp>


using namespace std;
using namespace cv;

namespace vs
{
    DataTable::DataTable(std::string dataPath):dataPath_(dataPath)
    {
        //load();
    }
    DataTable::~DataTable()
    {
        //save();
    }

//     DataTable::DataTable(const DataTable & other)
//     {
//         //this->dataPath_ = other.dataPath_;        
//         //for (size_t i = 0; i != other.deletedFlags_.size(); ++i)
//         //{
//         //    if (other.deletedFlags_[i] == false)
//         //    {
//         //        insertVideo(other.videoNames_[i],other.keyFrames_[i],other.frameFeats_[i],
//         //                    other.frameKeyPoints_[i],other.frameDescriptors_[i],other.descIndex_[i]);
//         //    }
//         //}
//     }

//     DataTable & DataTable::operator=(DataTable & other)
//     {
//         this->swap(other);
//         return *this;
//     }

//     void DataTable::swap(DataTable &other)
//     {
//         this->dataPath_.swap(other.dataPath_);
//         this->videoPaths_.swap(other.videoPaths_);
//         this->frameFeats_.swap(other.frameFeats_);
//         this->frameCnts_.swap(other.frameCnts_);
//         this->globalFmId2Vid_.swap(other.globalFmId2Vid_);
//         this->globalFmInd2LocalFmInd_.swap(other.globalFmInd2LocalFmInd_);    
//         isChanged = true;
//     }


    int DataTable::insertVideo(const string videoName)
    {
        //already exist 
        auto iter=vn2vId_.find(videoName);
        if (iter == vn2vId_.end())
            return -1;

        size_t vid = vn2vId_.size() + 1;
        vn2vId_.insert(make_pair(videoName, vid));


        shared_ptr<vector<KeyFrame>> kfm(new vector<KeyFrame>);
        shared_ptr<Mat> feat(new Mat);
        shared_ptr<vector<vector<KeyPoint>>> keys(new vector<vector<KeyPoint>>);
        shared_ptr<vector<Mat>> desc(new vector<Mat>);

        if (-1 == featExactor_.exactFeatures(videoName, *kfm.get(), *feat.get(), *keys.get(), *desc.get()))
        {
            return -1;
        }

        videoFrameFeats_.insert(make_pair(vid, feat));
        frameCnts_.push_back(feat->rows);



        ::flann::Matrix<float> points((float *)(feat->data), feat->rows, feat->cols);
        vector<size_t> frameIds = frameIndex_->addPoints(points);

        for (size_t i = 0; i = !frameIds.size(); ++i)
        {
            gFmId2VId_.insert(make_pair(frameIds[i], vid));
            gFmId2lFmId_.insert(make_pair(frameIds[i], i));
        }

        vId2gFmId_.insert(make_pair(vid, std::move(frameIds)));



        boost::filesystem::path pt(videoName);
        ostringstream ostrstream;
        ostrstream << dataPath_ << PATH_SEPARATOR << pt.filename().string() << DATA_FILE_EXT;
        string videoDataPath = ostrstream.str();

        vId2vDataPath_.insert(make_pair(vid, videoDataPath));

        ofstream ofs(videoDataPath);
        boost::archive::binary_oarchive oar(ofs);
        
        if (kfm != NULL)
        {
            oar & kfm;
        }
        if (keys != NULL)
        {
            oar & keys;
        }
        if (desc != NULL)
        {
            oar & desc;
        }
        
        isChanged = true;
        return 0;
    }

    void DataTable::deleteVideos(vector<string> videoNames)
    {
         for (size_t i = 0; i < videoNames.size(); i++)
        {
             deleteVideo(videoNames[i]);
        }           

    }
    void DataTable::deleteVideo(std::string videoName)
    {
        auto iter = vn2vId_.find(videoName);
        if (iter == vn2vId_.end())
            return;

        size_t vid = iter->second;

        vId2vDataPath_.erase(vid);
        videoFrameFeats_.erase(vid);
        
        auto iter2 = vId2gFmId_.find(vid);

        auto &vec = iter2->second;
        for (size_t i = 0; i != vec.size(); ++i)
        {
            gFmId2VId_.erase(vec[i]);
            gFmId2lFmId_.erase(vec[i]);
        }
        vId2gFmId_.erase(iter2);

        deleteVideoDataFromDisk();
     
        vn2vId_.erase(iter); 
        isChanged = true;
    }

    void DataTable::save(ostream & outStream)
    {
        if (!isChanged)
            return;
        boost::archive::binary_oarchive oar(outStream);

        oar & videoNames_;

        for (size_t i = 0; i < videoNames_.size(); i++)
        {
            oar & frameFeats_[i];
        }
        
        isChanged = false;
    }

    int DataTable::load(istream & inStream)
    {
        boost::filesystem::path p(dataPath_);
        if (!exists(p) || !is_directory(p))
            return -1;

        boost::archive::binary_iarchive iar(inStream);

        iar & videoNames_;
        
        for (auto vn : videoNames_)
        {
            shared_ptr<cv::Mat> feat;
            iar & feat;
            this->insertVideo(vn, NULL, feat, NULL, NULL);
        }

        isChanged = false;  //insertVideo will set this value to true
        return 0;
    }
    int DataTable::gFmInd2Vid(int gFmInd)
    {
        return this->globalFmId2Vid_[gFmInd];
    }

    int DataTable::gFmInd2LFmInd(int gFmInd)
    {
        return this->globalFmInd2LocalFmInd_[gFmInd];
    }


    void DataTable::getVideoData(int vid, vector<KeyFrame> &kfm, vector<vector<KeyPoint>> &keys, vector<Mat> &desc)
    {
        ifstream ifs(videoDataPaths_[vid]);
        boost::archive::binary_iarchive iar(ifs);

        iar & kfm;
        iar & keys;
        iar & desc;
    }

    int DataTable::getVideoFmCnt(int vid)
    {
        return frameCnts_[vid];
    }

    std::vector<std::shared_ptr<cv::Mat>>& DataTable::getVideoFrameFeat()
    {
        return frameFeats_;
    }


}