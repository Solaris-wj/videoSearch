
#include "Preprocessor.h"
#include <tuple>
#include <numeric>
#include <random>

namespace vs
{

    int Preprocessor::getDefaultMask(string videoName, Mat &mask)
    {
        VideoCapture cap(videoName);

        if (!cap.isOpened())
        {
            printf("can not open video file %s\n", videoName.c_str());
            return -1;
        }
        if (cap.get(CV_CAP_PROP_FRAME_COUNT)<10)
        {
            return -1;
        }
        int rows = (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT);
        int cols = (int)cap.get(CV_CAP_PROP_FRAME_WIDTH);

        int row_start, row_end, col_start, col_end;
        std::tie(row_start, row_end, col_start, col_end) = eraseBorder(cap);

        mask.release();
        mask = Mat::zeros(rows, cols, CV_8UC1);

        Mat withoutBorder = mask(Range(row_start, row_end), Range(col_start, col_end));

        Range row_ran = Range(withoutBorder.rows*top_, withoutBorder.rows*(1 - bottom_));
        Range col_ran = Range(withoutBorder.cols*left_, withoutBorder.cols*(1 - right_));
        Mat validRegion = withoutBorder(row_ran, col_ran);

        validRegion = 255 * Mat::ones(validRegion.rows, validRegion.cols, CV_8UC1);

        return 0;
    }

    tuple<int, int, int, int> Preprocessor::eraseBorder(VideoCapture &cap)
    {
        //float num_thres = 0.01;
        int piexl_thres = 5;

        vector<int> row_start_rec;
        vector<int> row_end_rec;
        vector<int> col_start_rec;
        vector<int> col_end_rec;

        int frameCnt = (int)cap.get(CV_CAP_PROP_FRAME_COUNT);
        std::default_random_engine generator;
        std::uniform_int_distribution<int> distribution(0, frameCnt - 1);

        int cnt = 0;
        while (cnt++ < 50)
        {
            cap.set(CV_CAP_PROP_POS_FRAMES, distribution(generator));
            Mat frame;
            cap.read(frame);
            if (!cap.isOpened())
            {
                break;
            }

            Scalar values = cv::sum(frame);
            float summ = values[0] + values[1] + values[2];
            if (summ< frame.total()*piexl_thres || summ>frame.total() *(255 - piexl_thres))
            {
                continue;
            }

            int rows = frame.rows;
            int cols = frame.cols;

            int i = 1;

            while (i<rows)
            {
                if (cv::sum(frame(Range(0, i), Range(0, cols)))[0]>piexl_thres *i *cols)
                    break;

                i += 1;
            }
            row_start_rec.push_back(i);


            i = rows - 1;
            while (i >= 0)
            {
                if (cv::sum(frame(Range(i, rows), Range(0, cols)))[0] > piexl_thres * (rows - i)*cols)
                    break;
                i--;
            }
            row_end_rec.push_back(i);

            i = 1;
            while (i<cols)
            {
                if (cv::sum(frame(Range(0, rows), Range(0, i)))[0] >piexl_thres*i*rows)
                    break;
                i++;
            }
            col_start_rec.push_back(i);

            i = cols - 1;
            while (i >= 0)
            {
                if (cv::sum(frame(Range(0, rows), Range(i, cols)))[0] > piexl_thres*(cols - i)*rows)
                    break;
                i--;
            }
            col_end_rec.push_back(i);
        }

        int height = (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT);
        int width = (int)cap.get(CV_CAP_PROP_FRAME_WIDTH);

        int start_row = 0, end_row = height, start_col = 0, end_col = width;

        if (row_start_rec.size() != 0)
        {
            start_row = std::accumulate(row_start_rec.begin(), row_start_rec.end(), 0) / row_start_rec.size();
        }
        if (row_end_rec.size() != 0)
        {
            end_row = std::accumulate(row_end_rec.begin(), row_end_rec.end(), 0) / row_end_rec.size();
        }
        if (col_start_rec.size() != 0)
        {
            start_col = std::accumulate(col_start_rec.begin(), col_start_rec.end(), 0) / col_start_rec.size();
        }
        if (col_end_rec.size() != 0)
        {
            end_col = std::accumulate(col_end_rec.begin(), col_end_rec.end(), 0) / col_end_rec.size();
        }

        //maxvalue
        //int start_row = *std::min_element(row_start_rec.begin(), row_start_rec.end());
        //int end_row = *std::max_element(row_end_rec.begin(), row_end_rec.end());

        //int start_col = *std::min_element(col_start_rec.begin(), col_start_rec.end());
        //int end_col = *std::max_element(col_end_rec.begin(), col_end_rec.end());
        tuple<int, int, int, int> ret_tuple(0, 0, 0, 0);
        ret_tuple = make_tuple(start_row, end_row, start_col, end_col);
        cap.set(CV_CAP_PROP_POS_FRAMES, 0);

        return ret_tuple;
    }
}