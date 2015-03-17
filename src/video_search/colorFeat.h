#ifndef COLOR_FEAT_H_H
#define COLOR_FEAT_H_H

#include <opencv2/opencv.hpp>

#include <stack>

#include "defines.h"


#define COLOR_SPACE_BGR 1
#define COLOR_SPACE_HLS 2


namespace vs
{
    /************************************************************************/
    /* 颜色聚合向量                                                                     */
    /************************************************************************/

    class VS_EXPORTS ColorCoherenceVec
    {
    public:
        ColorCoherenceVec()
        {
            m_binsNum[0] = 6;
            m_binsNum[1] = 3;
            m_binsNum[2] = 4;
            m_thresholdFactor = 0.01f;
            initSteps();
        }
        ColorCoherenceVec(int binsNum1, int binsNum2, int binsNum3, float threshold = 0.01)
        {
            m_binsNum[0] = binsNum1;
            m_binsNum[1] = binsNum2;
            m_binsNum[2] = binsNum3;
            m_thresholdFactor = threshold;
            initSteps();
        }
        void computeFeat(cv::Mat &src, cv::Mat &mask, cv::Mat &ccv, int colorSpaceType = COLOR_SPACE_BGR, int normType = cv::NORM_L1);
        int getFeatDim();

    private:
        void initSteps();
        void RegionGrow(cv::Mat &img, cv::Mat &flag, std::stack<cv::Point>* &pStack, int &connectedPixNum);

    public:
        int m_binsNum[3];
        int m_steps[3];
        float m_thresholdFactor;    //连通区域像素计算比例，默认图像总像素的1%
    };

    class VS_EXPORTS ColorHist
    {
    public:
        ColorHist()
        {
            m_binsNum[0] = 6;
            m_binsNum[1] = 6;
            m_binsNum[2] = 6;
            initSteps();
        }
        ColorHist(int binsNum1, int binsNum2, int binsNum3)
        {
            m_binsNum[0] = binsNum1;
            m_binsNum[1] = binsNum2;
            m_binsNum[2] = binsNum3;
            initSteps();
        }
        void computeFeat(cv::Mat &src, cv::Mat &mask, cv::Mat &hist, int colorSpaceType = COLOR_SPACE_BGR, int normType = cv::NORM_L1);
        int getFeatDim();

    private:
        void initSteps();

    public:
        int m_binsNum[3];
        int m_steps[3];
    };

}// namespace vs

#endif