#include "DataTable.h"

#include <stdio.h>

#include <boost/filesystem.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/serialization.hpp>

#include "Serialization.h"


using namespace std;
using namespace cv;

namespace vs
{
    DataTable::DataTable(std::string dataPath, VideoSearchParam &params) :dataPath_(dataPath), featExactor_(params)
    {
        indexDataFileName = dataPath_ + PATH_SEPARATOR + "indexGlobalInfo" + DATA_FILE_EXT;
        indexFileName = dataPath_ + PATH_SEPARATOR + "index" + DATA_FILE_EXT;

        frameIndex_.reset(new FrameIndex(::flann::MultiThreadHierarchicalIndexParams()));
    }
    DataTable::~DataTable()
    {
        //save();
    }

    int DataTable::insertVideo(const string videoName)
    {
        //already exist 
        auto iter = vn2vid_.find(videoName);
        if (iter != vn2vid_.end())
            return -1;

        size_t vid = vn2vid_.size();
        vn2vid_.insert(make_pair(videoName, vid));
        vid2vn_.insert(make_pair(vid, videoName));


        shared_ptr<vector<KeyFrame>> kfm(new vector<KeyFrame>);
        shared_ptr<Mat> feat(new Mat);
        shared_ptr<vector<vector<KeyPoint>>> keys(new vector<vector<KeyPoint>>);
        shared_ptr<vector<Mat>> desc(new vector<Mat>);

        if (-1 == featExactor_.exactFeatures(videoName, *kfm.get(), *feat.get(), *keys.get(), *desc.get()))
        {
            return -1;
        }

        videoFrameFeats_.insert(make_pair(vid, feat));

        ::flann::Matrix<float> points((float *)(feat->data), feat->rows, feat->cols);
        vector<size_t> frameIds = frameIndex_->addPoints(points);

        for (size_t i = 0; i != frameIds.size(); ++i)
        {
            gFmId2vid_.insert(make_pair(frameIds[i], vid));
            gFmId2lFmId_.insert(make_pair(frameIds[i], i));
        }

        vid2gFmId_.insert(make_pair(vid, std::move(frameIds)));

        string videoDataPath = getVideoDataPath(videoName);
        vid2vDataPath_.insert(make_pair(vid, videoDataPath));

        ofstream ofs(videoDataPath, std::ios::binary);
        boost::archive::binary_oarchive oar(ofs);

        oar & *kfm.get();
        oar & *keys.get();
        oar & *desc.get();


        isChanged = true;
        return 0;
    }


    void DataTable::getVideoData(int vid, vector<KeyFrame> &kfm, vector<vector<KeyPoint>> &keys, vector<Mat> &desc)
    {
        string &path = vid2vDataPath_.at(vid);
        ifstream ifs(path,std::ios::binary);
        //ifstream ifs(path);
        boost::archive::binary_iarchive iar(ifs);

        iar & kfm;
        iar & keys;
        iar & desc;
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
        auto iter = vn2vid_.find(videoName);
        if (iter == vn2vid_.end())
            return;

        size_t vid = iter->second;

        vid2vDataPath_.erase(vid);
        videoFrameFeats_.erase(vid);

        auto iter2 = vid2gFmId_.find(vid);

        auto &vec = iter2->second;
        for (size_t i = 0; i != vec.size(); ++i)
        {
            gFmId2vid_.erase(vec[i]);
            gFmId2lFmId_.erase(vec[i]);
        }
        vid2gFmId_.erase(iter2);

        deleteVideoDataFrameDisk(vid);

        vn2vid_.erase(iter);
        isChanged = true;
    }

    void DataTable::deleteVideoDataFrameDisk(size_t vid)
    {
        string vn = vid2vn_.at(vid);
        string videoDataPath = getVideoDataPath(vn);

        remove(videoDataPath.c_str());

    }

    void DataTable::save()
    {
        ofstream ifs(indexDataFileName);
        boost::archive::binary_oarchive iar(ifs);
        iar & (*this);
    }
    template<class Archive>
    void DataTable::save(Archive & oar, const unsigned int version) const
    {
        if (!isChanged)
            return;

        int videoNum = vn2vid_.size();
        oar & vn2vid_;

        for (auto & pa : vn2vid_)
        {
            size_t vid = pa.second;
            oar & pa.first;
            oar & pa.second;

            oar & *(videoFrameFeats_.at(vid).get());
            oar & vid2gFmId_.at(vid);
        }

        isChanged = false;
    }

    void DataTable::load()
    {
        boost::filesystem::path p(indexDataFileName);
        if (!exists(p) || !is_directory(p))
            return;

        ifstream ifs(indexDataFileName);
        boost::archive::binary_iarchive iar(ifs);

        iar & (*this);
    }

    template<class Archive>
    void DataTable::load(Archive & iar, const unsigned int version)
    {
        boost::filesystem::path p(indexFileName);
        if (!exists(p) || !is_directory(p))
            return;

        frameIndex_->save(p.string());

        int videoNum = vn2vid_.size();

        iar & videoNum;

        for (size_t i = 0; i < videoNum; i++)
        {
            string vn;
            size_t vid;
            iar & vn;
            iar & vid;
            vn2vid_.insert(make_pair(vn, vid));
            vid2vn_.insert(make_pair(vid, vn));

            vid2vDataPath_.insert(make_pair(vid, getVideoDataPath(vn)));

            shared_ptr<Mat> feat(new Mat);
            iar & *feat.get();
            videoFrameFeats_.insert(make_pair(vid, feat));

            vector<size_t> gfm;
            iar & gfm;

            for (size_t j = 0; j != gfm.size(); ++j)
            {
                gFmId2vid_.insert(make_pair(gfm[j], vid));
                gFmId2lFmId_.insert(make_pair(gfm[j], j));
            }

            vid2gFmId_.insert(make_pair(vid, move(gfm)));

        }
        //isChanged = false;  //insertVideo will set this value to true
        return;
    }



}