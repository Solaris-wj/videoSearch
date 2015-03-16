#include "VideoIndexEngine.h"

#include <memory>
#include <string>

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <rapidjson/prettywriter.h>

#include <time.h>

using namespace std;
using namespace rapidjson;
//using namespace cv;

using cv::Mat;
using cv::KeyPoint;
using cv::SparseMat;
using cv::DMatch;
using cv::BFMatcher;
using cv::Point2f;
using cv::RNG;
using cv::getAffineTransform;

namespace vs
{
    VideoIndexEngine::VideoIndexEngine(string dataDir, string logDir, string algoConfFilePath) :
        dataDir_(dataDir), logDir_(logDir), param_(algoConfFilePath),
        videoTable_(dataDir, param_)
    {
        videoTable_.load();
    }

    VideoIndexEngine::~VideoIndexEngine()
    {
        videoTable_.save();
    }

    int VideoIndexEngine::addVideo(vector<string> &videoNames)
    {
        for (size_t i = 0; i != videoNames.size(); ++i)
        {
            addVideo(videoNames[i]);
        }
        return 0;
    }
    int VideoIndexEngine::addVideo(string &videoName)
    {
        return videoTable_.insertVideo(videoName);
    }

    int VideoIndexEngine::deleteVideo(string &videoName)
    {
        videoTable_.deleteVideo(videoName);

        return 0;
    }
    int VideoIndexEngine::deleteVideos(vector<string> &videoNames)
    {
        videoTable_.deleteVideos(videoNames);
        return 0;
    }

    bool static set_cmp(const Point2f &pt1, const Point2f &pt2, int threshold)
    {
        if (pt1.x - (pt2.x) > threshold)
            return true;
        else if (abs(pt1.x - pt2.x) <threshold && pt1.y - pt2.y>threshold)
            return true;
        else
            return false;
    }

    int VideoIndexEngine::ransac(vector<KeyPoint> &keys1, vector<KeyPoint> &keys2, vector<DMatch> &matches, /*vector<DMatch> &fined_matches,*/ int threshold, int iteration)
    {
        if (matches.size() < 3)
        {
            return -1;
        }
        vector<Point2f> pt1;
        vector<Point2f> pt2;

        vector<size_t> orginal_index_pt1;
        vector<size_t> orginal_index_pt2;

        for (size_t i = 0; i != matches.size(); ++i)
        {
            pt1.push_back(keys1[matches[i].queryIdx].pt);
            pt2.push_back(keys2[matches[i].trainIdx].pt);

            orginal_index_pt1.push_back(matches[i].queryIdx);
            orginal_index_pt2.push_back(matches[i].trainIdx);
        }

        ////erase duplicate points 
        auto fn = bind(set_cmp, placeholders::_1, placeholders::_2, threshold);
        map<Point2f, int, function<bool(Point2f, Point2f)>> map_pt1(fn);
        map<Point2f, int, function<bool(Point2f, Point2f)>> map_pt2(fn);

        vector<Point2f> filtered_pt1;
        vector<Point2f> filtered_pt2;
        for (size_t i = 0; i != pt1.size(); ++i)
        {
            auto tmp1 = ++map_pt1[pt1[i]];
            auto tmp2 = ++map_pt2[pt2[i]];

            if (tmp1 > 1 || tmp2 > 1)
                continue;

            filtered_pt1.push_back(pt1[i]);
            filtered_pt2.push_back(pt2[i]);
        }

        if (filtered_pt1.size() < 3 || filtered_pt2.size() < 3)
            return -1;
        filtered_pt1.swap(pt1);
        filtered_pt2.swap(pt2);

        int max_cnt = 0;
        //int n = 100;
        //RNG rng = cv::theRNG();
        RNG rng;
        while (iteration-- > 0)
        {
            vector<Point2f> srcPt;
            vector<Point2f> dstPt;
            size_t id1 = rng(pt1.size());
            size_t id2 = rng(pt1.size());
            size_t id3 = rng(pt1.size());

            srcPt.push_back(pt1[id1]);
            srcPt.push_back(pt1[id2]);
            srcPt.push_back(pt1[id3]);

            dstPt.push_back(pt2[id1]);
            dstPt.push_back(pt2[id2]);
            dstPt.push_back(pt2[id3]);

            Mat affineMat = getAffineTransform(srcPt, dstPt);
            
            vector<Point2f> outPt;
            transform(pt1, outPt, affineMat);

            //vector<DMatch> temp_match;
            //temp_match.reserve(pt2.size() / 2);
            int inLier = 0;
            for (int i = 0; i != pt2.size(); ++i)
            {
                if (std::fabs(pt2[i].x - outPt[i].x) < threshold && std::fabs(pt2[i].y - outPt[i].y) <threshold)
                {
                    inLier++;
                    //temp_match.push_back(DMatch(orginal_index_pt1[i], orginal_index_pt2[i], 0));
                    //temp_match.emplace_back(orginal_index_pt1[i], orginal_index_pt2[i], 0);
                }
            }

            if (inLier> max_cnt)
            {
                max_cnt = inLier;
                //fined_matches = temp_match;
            }
        }
        return max_cnt;
    }

    void VideoIndexEngine::videoRansac(vector<vector<KeyPoint>> &keys1, vector<Mat> desc1, vector<vector<KeyPoint>> &keys2, vector<Mat> desc2,
                     vector<DMatch> &matches, vector<DMatch> &filteredMatches)
    {
        BFMatcher desc_matcher(cv::NORM_HAMMING);
        //loop over initial match frame 
        for (size_t i = 0; i != matches.size(); ++i)
        {
            vector<DMatch> kp_matches;

            desc_matcher.match(desc1[matches[i].queryIdx], desc2[matches[i].trainIdx], kp_matches);

            vector<DMatch> temp;

            for (auto &v : kp_matches)
            {
                //max distance is 256
                if (v.distance < 50)
                {
                    temp.push_back(v);
                }
            }
            temp.swap(kp_matches);

            int inLier = ransac(keys1[matches[i].queryIdx], keys2[matches[i].trainIdx], kp_matches, 5, 100);

            //可将orbthreshold 改为按比例计算 10%？
            if (inLier > param_.orbThres)
            {
                filteredMatches.emplace_back(matches[i].queryIdx, matches[i].trainIdx, inLier);
            }
        }
    }

    clock_t start;
    int VideoIndexEngine::searchVideo(std::string &searchVideoName, std::string jsonResult)
    {
        shared_ptr<vector<KeyFrame>> keyFrames(new vector<KeyFrame>);
        shared_ptr<Mat> feat(new Mat);
        shared_ptr<vector<vector<KeyPoint>>> keys(new vector<vector<KeyPoint>>);
        shared_ptr<vector<Mat>> desc(new vector<Mat>);

        if (-1 == videoTable_.getFeatExactor().exactFeatures(searchVideoName, *keyFrames.get(), *feat.get(), *keys.get(), *desc.get()))
        {
            vec_jsonResult = vector<string>();
            return -1;
        }                

        ::flann::Matrix<float> flann_queries((float*)(feat->data), feat->rows, feat->cols);
        vector<vector<int>> flann_indices;
        vector<vector<float>> flann_dist;


        videoTable_.getFrameIndex().radiusSearch(flann_queries, flann_indices, flann_dist,2-2*param_.colorThres, ::flann::SearchParams());

        //printf("flann sear frame = %lf s\n", (double)(clock() - start) / CLOCKS_PER_SEC);


        //vote for videos 
        map<int, int> votes;
        //map(vid,vector< tuple< pos_in_qury, pos_in_data, score> > 
        //map<int, vector<tuple<int, int, int>>> match_frames;
        map<int, vector<DMatch>> match_frames;
        for (int i = 0; i != flann_indices.size(); ++i)
        {
            for (int j = 0; j != flann_indices[i].size(); ++j)
            {
                int pos_inTotalFeat = flann_indices[i][j];
                float ratio = 1 - flann_dist[i][j] / 2;
                int videoId = videoTable_.gFmInd2Vid(pos_inTotalFeat);                 
                votes[videoId] += 1;                
                match_frames[videoId].emplace_back(i, videoTable_.gFmInd2LFmInd(pos_inTotalFeat), ratio);
            }
        }

        printf("match_frames num=%d\n", match_frames.size());
        //sort vote result
        vector<pair<int, int>> votesMap(votes.begin(), votes.end());
        std::sort(votesMap.begin(), votesMap.end(), [](pair<int, int> &v1, pair<int, int> &v2){ return v1.second > v2.second; });

        int vFrameCnt = feat->rows;

        vector<std::string> vec_jsonResult;

        for (size_t i = 0; i != votesMap.size(); ++i)
        {
            int match_num=match_frames[votesMap[i].first].size();
            printf("match_num=%d\n", match_num);
            //if (match_num < param_.timeInterval)
            //    continue;

            vector<DMatch> &matches = match_frames[votesMap[i].first];
            vector<DMatch> filtered_matches;
            
            start = clock();

            vector<KeyFrame> tar_kf;
            vector<vector<KeyPoint>> tar_keys;
            vector<Mat> tar_desc;
            videoTable_.getVideoData(votesMap[i].first, tar_kf, tar_keys, tar_desc);


            videoRansac(*keys.get(), *desc.get(),tar_keys,
                        tar_desc, matches,filtered_matches);
            
            printf("videoRansac time = %lf s\n", (double)(clock() - start) / CLOCKS_PER_SEC);

            //int frameCnt_tar = frameCnts_[static_cast<int>(votesMap[i].first)];
            int frameCnt_tar = videoTable_.getVideoFmCnt(votesMap[i].first);

            int sparseMatSize[2] = { vFrameCnt, frameCnt_tar };
            SparseMat flagMat(2, sparseMatSize, CV_8UC1);

            //vector<tuple<int, int, int>> &matches = match_frames[votesMap[i].first];
            
            for (size_t j = 0; j != filtered_matches.size(); ++j)
            {
                flagMat.ref<uchar>(filtered_matches[j].queryIdx, filtered_matches[j].trainIdx)= filtered_matches[j].distance / videoTable_.getFeatExactor().getMaxLocalFeatNum() * 255;
            }

            //debug
            //Mat flag;
            //flagMat.copyTo(flag);

            //match frame index
            vector<pair<int, int> > det_initFrameMatches;
            vector<pair<int, int> > tar_initFrameMatches;

            assembleMatches(flagMat, 0, vFrameCnt, 0, frameCnt_tar, det_initFrameMatches, tar_initFrameMatches);


            //match time in second
            vector<pair<int, int>> finalMatchResult_det;
            vector<pair<int, int>> finalMatchResult_tar;

            //convert init match index to time sequence 
            for (size_t j = 0; j != det_initFrameMatches.size(); ++j)
            {
                vector<KeyFrame> &det_kf = *keyFrames.get();
                finalMatchResult_det.emplace_back(det_kf[det_initFrameMatches[j].first].start_index_, det_kf[det_initFrameMatches[j].second].end_index_);

                //vector<KeyFrame> &tar_kf = videoTable_.getVideoKeyFrames(votesMap[i].first);
                finalMatchResult_tar.emplace_back(tar_kf[tar_initFrameMatches[j].first].start_index_, tar_kf[tar_initFrameMatches[j].second].end_index_);
            }
            

            if (finalMatchResult_det.size() == 0)
            {
                continue;
            }

            string result;

            genJsonStr(videoTable_.getVideoName(votesMap[i].first), finalMatchResult_det, finalMatchResult_tar, result);

            vec_jsonResult.push_back(result);
            
            //Mat resultImg = drawDetResult(vFrameCnt / param_.used_fps, finalMatchResult_det, frameCnt_tar / param_.used_fps, finalMatchResult_tar);
            //saveResult(videoName, videoNames_[votesMap[i].first], result, resultImg);
        }

        Document d;
        d.SetArray();

        auto & alloc = d.GetAllocator();
        for (size_t i = 0; i != vec_jsonResult.size(); ++i)
        {
            Value v(vec_jsonResult[i].c_str(), alloc);
            d.PushBack(v,alloc);
        }

        StringBuffer buffer;
        PrettyWriter<StringBuffer> writer(buffer);
        d.Accept(writer);
        jsonResult= buffer.GetString();

        return 0;
    }

    tuple<int, int> VideoIndexEngine::go_bottom_right(Mat &recordMap, SparseMat & flagMat, int r, int c, int interval)
    {
//         //debug code
//         Mat flag;
//         flagMat.copyTo(flag);
//         //debug code over

        //int gap_len = 0;

        int valid_r = r;
        int valid_c = c;

        //while loop avoid recursion
        while (true)
        {
            //given a (valid_r,valid_c) find for next one, if success,break out for, 
            //use the found (valid_r,valid_c) to start with again
            //if for loop end normally, then no next one, so return
            // in the for loop visit the element in diagonal line first,the method found on stackoverflow.com
            bool next = false;
            for (int diag = 1; diag < interval; ++diag)
            {
                auto pred = [&](int pos_r, int pos_c)
                {
                    return pos_r < recordMap.rows && pos_c
                        && recordMap.data[pos_r * recordMap.cols + pos_c] == 0
                        && flagMat.ptr(pos_r, pos_c, false) != NULL;
                };

                int test_row = valid_r + diag;
                int test_col = valid_c + diag;

                if (pred(test_row, test_col))
                {
                    next = true;
                    valid_r = test_row;
                    valid_c = test_col;
                }
                else
                {
                    for (int delta = 1; delta <= diag; ++delta)
                    {

                        test_row = valid_r + diag;
                        test_col = valid_c + diag - delta;
                        if (pred(test_row, test_col))
                        {
                            next = true;
                            valid_r = test_row;
                            valid_c = test_col;
                            break;
                        }

                        test_row = valid_r + diag - delta;
                        test_col = valid_c + diag;
                        if (pred(test_row, test_col))
                        {
                            next = true;
                            valid_r = test_row;
                            valid_c = test_col;
                            break;
                        }
                    }//end of for delta


                }
                if (next == true)
                {
                    break;
                }
            }//end of for diag

            if (next == false)
            {
                return make_tuple(valid_r, valid_c);
            }

        }//end of while

    }

    void VideoIndexEngine::assembleMatches(SparseMat &flagMat, int beg0, int end0, int beg1, int end1
                                           , vector<pair<int, int>> &matches0, vector<pair<int, int>> &matches1)
    {
        //debug code
        //Mat flag;
        //flagMat.copyTo(flag);
        //debug code over
        Mat recordMap = Mat::zeros(end0 - beg0, end1 - beg1, CV_8UC1);
        const int interval = static_cast<int>(param_.timeInterval*param_.usedFps);

        for (int i = beg0; i < end0; ++i)
        {

            for (int j = beg1; j < end1; ++j)
            {
                if (recordMap.data[i*recordMap.cols + j] != 0 || flagMat.ptr(i, j, false) == NULL)
                {
                    continue;
                }
                int valid_r, valid_c;
                std::tie(valid_r, valid_c) = go_bottom_right(recordMap, flagMat, i, j, interval);

                //record visited location
                for (int m = i; m < valid_r + 1; ++m)
                {
                    for (int n = j; n < valid_c + 1; ++n)
                    {
                        recordMap.data[m*recordMap.cols + n] = 255;
                    }
                }
//                 if (valid_r - i + 1 < interval || valid_c - j + 1 < interval)
//                 {
//                     continue;
//                 }
                matches0.push_back(make_pair(i, valid_r));
                matches1.push_back(make_pair(j, valid_c));
            }
        }
    }

    int VideoIndexEngine::genJsonStr(const string &targetVideoName, vector<pair<int, int>> & finalMatchResult_det,
                   vector<pair<int, int>> &finalMatchResult_tar, string &strJsonResult)
    {

        Document d;
        d.SetObject();
        auto & alloc = d.GetAllocator();

        d.AddMember("filename", Value(targetVideoName.c_str(),alloc).Move(),alloc);
        Value timeArr;
        timeArr.SetArray();
        for (size_t i = 0; i != finalMatchResult_det.size(); ++i)
        {
            pair<int, int> & tempTar = finalMatchResult_tar[i];
            pair<int, int> & tempDet = finalMatchResult_det[i];

            Value val(kObjectType);
            val.AddMember("det_beg", tempDet.first, alloc);
            val.AddMember("det_end", tempDet.second, alloc);
            val.AddMember("tar_beg", tempTar.first, alloc);
            val.AddMember("tar_end", tempTar.second, alloc);

            timeArr.PushBack(val,alloc);
        }

        d.AddMember("time", timeArr,alloc);

        StringBuffer buffer;
        PrettyWriter<StringBuffer> writer(buffer);
        d.Accept(writer);
        strJsonResult = buffer.GetString();

        return 0;
    }
}