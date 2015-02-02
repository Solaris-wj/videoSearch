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
    DataTable::DataTable(const DataTable & other)
    {
        //this->dataPath_ = other.dataPath_;        
        //for (size_t i = 0; i != other.deletedFlags_.size(); ++i)
        //{
        //    if (other.deletedFlags_[i] == false)
        //    {
        //        insertVideo(other.videoNames_[i],other.keyFrames_[i],other.frameFeats_[i],
        //                    other.frameKeyPoints_[i],other.frameDescriptors_[i],other.descIndex_[i]);
        //    }
        //}
    }

    DataTable & DataTable::operator=(DataTable & other)
    {
        this->swap(other);
        return *this;
    }
    void DataTable::swap(DataTable &other)
    {
        this->dataPath_.swap(other.dataPath_);
        this->videoNames_.swap(other.videoNames_);
        this->frameFeats_.swap(other.frameFeats_);
        this->frameCnts_.swap(other.frameCnts_);
        this->globalFmInd2Vid_.swap(other.globalFmInd2Vid_);
        this->globalFmInd2LocalFmInd_.swap(other.globalFmInd2LocalFmInd_);    
        isChanged = true;
    }


    int DataTable::insertVideo(const string videoName, const shared_ptr<vector<KeyFrame>> kfm,
                     const shared_ptr<Mat> feat, const shared_ptr<vector<vector<KeyPoint>>> keypoints,
                     const shared_ptr<vector<Mat>> desc)
    {
        //already exist 
        auto iter=vn2vid_.find(videoName);
        if (iter==vn2vid_.end())
            return -1;
     
        videoNames_.push_back(videoName);  
        vn2vid_.insert(make_pair(videoName, videoNames_.size()-1));

        frameFeats_.push_back(feat);
        frameCnts_.push_back(feat->rows);

        globalFmInd2Vid_.insert(globalFmInd2Vid_.end(), feat->rows, videoNames_.size() - 1);
        std::vector<int> temp(feat->rows);
        std::iota(temp.begin(), temp.end(), 0);
        globalFmInd2LocalFmInd_.insert(globalFmInd2LocalFmInd_.end(), temp.begin(), temp.end());

        boost::filesystem::path pt(videoName);

        ostringstream ostrstream;
        ostrstream << dataPath_ << PATH_SEPARATOR << pt.filename().string() << DATA_FILE_EXT;
        string outFileName=ostrstream.str();
        videoDataPaths_.push_back(outFileName);

        ofstream ofs(outFileName);
        boost::archive::binary_oarchive oar(ofs);
        
        if (kfm != NULL)
        {
            oar & kfm;
        }
        if (keypoints != NULL)
        {
            oar & keypoints;
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
        vector<size_t> delIndex;

        DataTable temp(this->dataPath_);

        for (size_t i = 0; i < videoNames.size(); i++)
        {
            auto iter=vn2vid_.find(videoNames[i]);
            if (iter != vn2vid_.end())
                delIndex.push_back(iter->second);
        }

        
    }
    void DataTable::deleteVideo(std::string videoName)
    {
        auto iter = vn2vid_.find(videoName);
        if (iter == vn2vid_.end())
            return;
        size_t vid = iter->second;

        DataTable temp(this->dataPath_);
        for (size_t i = 0; i != videoNames_.size(); ++i)
        {
            if ()
            temp.insertVideo(videoNames_[i], NULL, frameFeats_[i], NULL, NULL);
        }

        vn2vid_.erase(videoName);
        frameFeats_[vid] = nullptr;

        int deletedNum = std::count_if(deletedFlags_.begin(), deletedFlags_.end(), [](bool v){return v == true; });
        if (deletedNum > videoNames_.size() / 3.0f)
        {
            DataTable temp(*this);
            *this = temp;
            //DataTable temp;
            //temp.swap(*this);
        }
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
        return this->globalFmInd2Vid_[gFmInd];
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