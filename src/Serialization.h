
#ifndef SERIALIZATION_H
#define SERIALIZATION_H

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>

#include <memory>
#include <opencv2/opencv.hpp>

#include "KeyFrameExactor.h"

namespace boost
{
    namespace serialization
    {
        //shared ptr
        template<class Archive, class T>
        void serialize(Archive & ar, const std::shared_ptr<T> &pT, const unsigned int version)
        {
            ar & *pT.get();
        }

        //cv::Mat 
        template<class Archive>
        void serialize(Archive & ar, cv::Mat &mat, const unsigned int file_version)
        {
            split_free(ar, mat, file_version);
        }

        template<class Archive>
        void save(Archive &ar, const cv::Mat &mat, const unsigned int version)
        {
            int r, c, t;
            r = mat.rows;
            c = mat.cols;
            t = mat.type();
            ar & r;
            ar & c;
            ar & t;
            ar.save_binary(mat.data, mat.total()*mat.elemSize());
        }
        template<class Archive>
        void load(Archive &ar, Mat &mat, const unsigned int version)
        {
            int r, c, t;
            ar &r;
            ar &c;
            ar &t;
            Mat temp(r, c, t);
            ar.load_binary(temp.data, temp.total()*temp.elemSize());
            mat = temp;
        }

        //keyframe
        template<class Archive>
        void serialization(Archive &ar, const KeyFrame &kfm, const unsigned int version)
        {
            ar & kfm.start_index_;
            ar & kfm.end_index_;
            ar & kfm.key_fmIndex_;
        }

        //cv::KeyPoint
        template<class Archive>
        void serialization(Archive &ar, const KeyPoint &kp, const unsigned int version)
        {
            ar & kp.pt.x;
            ar & kp.pt.y;
            ar & kp.angle;
            ar & kp.class_id;
            ar & kp.octave;
            ar & kp.response;
            ar & kp.size;
        }
    }
}

#endif