
#include "colorFeat.h"

#include <stack>

using namespace cv;
using namespace std;
namespace vs
{
    /************************************************************************/
    /*  class implementation of color coherence vector                                                  */
    /************************************************************************/

    int ColorCoherenceVec::getFeatDim()
    {
        return m_binsNum[0] * m_binsNum[1] * m_binsNum[2] * 2;
    }

    void ColorCoherenceVec::initSteps()
    {
        m_steps[0] = std::ceil((float)256 / m_binsNum[0]);
        m_steps[1] = std::ceil((float)256 / m_binsNum[1]);
        m_steps[2] = std::ceil((float)256 / m_binsNum[2]);
    }


    void ColorCoherenceVec::computeFeat(Mat &src, Mat &mask, Mat &ccv, int colorSpaceType, int normType)
    {
        assert(src.channels() == 3);
        assert(src.type() == CV_8UC3);
        assert(src.rows == mask.rows);
        assert(src.cols == mask.cols);
        ccv.release();
        ccv = Mat::zeros(1, getFeatDim(), CV_32FC1);

        Mat img;

        if (colorSpaceType == COLOR_SPACE_HLS)
        {
            cvtColor(src, img, CV_BGR2HLS);
            for (int r = 0; r < img.rows; ++r)
            {
                for (int c = 0; c < img.cols; ++c)
                {
                    int tmp = img.at<Vec3b>(r, c)[0];
                    img.at<Vec3b>(r, c)[0] = (float)tmp * 2 / 360 * 255;
                }
            }
        }
        else
        {
            img = src;
        }

        //initSteps();

        Mat bluredImg;
        //blur(img,bluredImg,Size(3,3));
        bluredImg = img;
        
        Mat flag = Mat::zeros(bluredImg.rows, bluredImg.cols, CV_8UC1);

        std::stack<Point> *pStack = new std::stack<Point>();

        for (int r = 0; r < bluredImg.rows; ++r)
        {
            for (int c = 0; c < bluredImg.cols; ++c)
            {
                if (mask.at<uchar>(r,c) !=0 && flag.at<uchar>(r, c) == 0)
                {
                    int connectedPixNum = 0;
                    Point pt(c, r);
                    pStack->push(pt);
                    flag.at<uchar>(pt) = 1;
                    RegionGrow(bluredImg, flag, pStack, connectedPixNum);

                    Vec3b index = img.at<Vec3b>(pt);
                    index[0] = index[0] / m_steps[0];
                    index[1] = index[1] / m_steps[1];
                    index[2] = index[2] / m_steps[2];

                    //根据颜色量化bin确定在结果直方图中的位置
                    int location = index[0] * m_binsNum[1] * m_binsNum[2] + index[1] * m_binsNum[2] + index[2];
                    location *= 2;

                    //这里根据位置在把新得到的聚合像素或者非聚合像素加到相应位置，需要原始ccv矩阵是0初始化的
                    if (connectedPixNum >= int(m_thresholdFactor*bluredImg.total()))
                        ccv.at<float>(location) += connectedPixNum;
                    else
                        ccv.at<float>(location + 1) += connectedPixNum;
                }
            }
        }

        if (normType == -1)
        {
            return;
        }
        normalize(ccv, ccv, 1.0, 0, normType);

        delete pStack;
    }

    //四连通,img 原图像，flag标记是否访问过的图像
    //访问下一个结点满足两个条件，一是没超边界并且每访问过，二是满足颜色相似

    void ColorCoherenceVec::RegionGrow(Mat &img, Mat &flag, std::stack<Point>*& pStack, int &connectedPixNum)
    {

        while (pStack->size() != 0)
        {
            Point pt = pStack->top();
            connectedPixNum += 1;
            //flag.at<uchar>(pt)=1;

            pStack->pop();

            Vec3b index = img.at<Vec3b>(pt);
            index[0] = index[0] / m_steps[0];
            index[1] = index[1] / m_steps[1];
            index[2] = index[2] / m_steps[2];

            //go up
            if (pt.y - 1 >= 0 && flag.at<uchar>(pt.y - 1, pt.x) == 0)
            {
                Vec3b indexNext = img.at<Vec3b>(pt.y - 1, pt.x);
                indexNext[0] = indexNext[0] / m_steps[0];
                indexNext[1] = indexNext[1] / m_steps[1];
                indexNext[2] = indexNext[2] / m_steps[2];
                if (index[0] == indexNext[0] && index[1] == indexNext[1] && index[2] == indexNext[2])
                {
                    pStack->push(Point(pt.x, pt.y - 1));
                    flag.at<uchar>(Point(pt.x, pt.y - 1)) = 1;
                }
            }
            //go down
            if (pt.y + 1 < img.rows && flag.at<uchar>(pt.y + 1, pt.x) == 0)
            {
                Vec3b indexNext = img.at<Vec3b>(pt.y + 1, pt.x);
                indexNext[0] = indexNext[0] / m_steps[0];
                indexNext[1] = indexNext[1] / m_steps[1];
                indexNext[2] = indexNext[2] / m_steps[2];
                if (index[0] == indexNext[0] && index[1] == indexNext[1] && index[2] == indexNext[2])
                {
                    pStack->push(Point(pt.x, pt.y + 1));
                    flag.at<uchar>(Point(pt.x, pt.y + 1)) = 1;
                }
            }
            //go right
            if (pt.x + 1 < img.cols && flag.at<uchar>(pt.y, pt.x + 1) == 0)
            {
                Vec3b indexNext = img.at<Vec3b>(pt.y, pt.x + 1);
                indexNext[0] = indexNext[0] / m_steps[0];
                indexNext[1] = indexNext[1] / m_steps[1];
                indexNext[2] = indexNext[2] / m_steps[2];
                if (index[0] == indexNext[0] && index[1] == indexNext[1] && index[2] == indexNext[2])
                {
                    pStack->push(Point(pt.x + 1, pt.y));
                    flag.at<uchar>(Point(pt.x + 1, pt.y)) = 1;
                }
            }
            //go left
            if (pt.x - 1 >= 0 && flag.at<uchar>(pt.y, pt.x - 1) == 0)
            {
                Vec3b indexNext = img.at<Vec3b>(pt.y, pt.x - 1);
                indexNext[0] = indexNext[0] / m_steps[0];
                indexNext[1] = indexNext[1] / m_steps[1];
                indexNext[2] = indexNext[2] / m_steps[2];
                if (index[0] == indexNext[0] && index[1] == indexNext[1] && index[2] == indexNext[2])
                {
                    pStack->push(Point(pt.x - 1, pt.y));
                    flag.at<uchar>(Point(pt.x - 1, pt.y)) = 1;
                }
            }
        }
    }


    void ColorHist::initSteps()
    {
        m_steps[0] = static_cast<int>(std::ceil((float)256 / m_binsNum[0]));
        m_steps[1] = static_cast<int>(std::ceil((float)256 / m_binsNum[1]));
        m_steps[2] = static_cast<int>(std::ceil((float)256 / m_binsNum[2]));
    }

    void ColorHist::computeFeat(Mat &src, Mat &mask, Mat &hist, int colorSpaceType, int normType)
    {
        assert(src.channels() == 3);
        assert(src.type() == CV_8UC3);

        hist.release();
        hist = Mat::zeros(1, getFeatDim(), CV_32FC1);

        Mat img;

        if (colorSpaceType == COLOR_SPACE_HLS)
        {
            cvtColor(src, img, CV_BGR2HLS);
            for (int r = 0; r < img.rows; ++r)
            {
                for (int c = 0; c < img.cols; ++c)
                {
                    int tmp = img.at<cv::Vec3b>(r, c)[0];
                    img.at<cv::Vec3b>(r, c)[0] = static_cast<uchar>((float)tmp * 2 / 360 * 255);
                }
            }
        }
        else
        {
            img = src;
        }
        //initSteps();
        
        for (int r = 0; r < img.rows; ++r)
        {
            for (int c = 0; c < img.cols; ++c)
            {
                if (mask.at<uchar>(r, c) == 0)
                    continue;

                cv::Vec3b index = img.at<cv::Vec3b>(r, c);
                index[0] = index[0] / m_steps[0];
                index[1] = index[1] / m_steps[1];
                index[2] = index[2] / m_steps[2];

                //int location = index[0] * m_binsNum[1] * m_binsNum[2] + index[1] * m_binsNum[2] + index[2];
                int location = index[0] * m_binsNum[1] + index[1];
                hist.at<float>(location) += 1;
            }
        }

        //L1 normalization

        if (normType == -1)
        {
            return;
        }
        normalize(hist, hist, 1.0, 0, normType);
    }


    int ColorHist::getFeatDim()
    {
        //return m_binsNum[0] * m_binsNum[1] * m_binsNum[2];
        return m_binsNum[0] * m_binsNum[1];
    }

}