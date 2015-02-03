#include <vld.h>

#include <time.h>
#include <iostream>
#include <string>
#include <vector>
#include <time.h>
#include <algorithm>
#include <sstream>

#include <opencv2/opencv.hpp>

#include <boost/filesystem.hpp>
//#include <boost/property_tree/json_parser.hpp>

#include "../src/VideoReader.h"
#include "../src/VideoIndexEngine.h"

#ifdef WIN32
#ifdef _DEBUG
#pragma comment(lib,"opencv_core249d.lib")
#pragma comment(lib,"opencv_highgui249d.lib")
#pragma comment(lib,"opencv_imgproc249d.lib")
#pragma comment(lib,"opencv_features2d249d.lib")

#else
#pragma comment(lib,"opencv_core249.lib")
#pragma comment(lib,"opencv_highgui249.lib")
#pragma comment(lib,"opencv_imgproc249.lib")
#pragma comment(lib,"opencv_features2d249.lib")
// #pragma comment(lib,"libboost_filesystem-vc120-mt-1_57.lib")
// #pragma comment(lib,"libboost_serialization-vc120-mt-1_57.lib")

#pragma comment(lib,"boost_filesystem-vc120-mt-1_57.lib")
#pragma comment(lib,"boost_serialization-vc120-mt-1_57.lib")

#endif
#endif
using namespace std;
using namespace cv;
//using namespace boost;
//using namespace boost::filesystem;

using boost::filesystem::directory_iterator;
using boost::filesystem::path;
//using boost::property_tree::write_json;
int main()
{
    string root = "d:/videoSearchServer/data";
    string algoConf = "D:\\videoSearchServer\\conf\\algo.conf";

    string videoDir = "D:/videos/tar";

    vector<path> vec_path;

    path p(videoDir);

    copy_if(directory_iterator(p), directory_iterator(), back_inserter(vec_path),
            [](path  pp)
            { return is_regular_file(pp); }
            );

    vector<string> tar_videoNames;
    transform(vec_path.begin(), vec_path.end(), back_inserter(tar_videoNames), [](path &p){ return p.string(); });

    vs::VideoIndexEngine indexEngine(root,algoConf);

    clock_t beg;
    
    beg = clock();
    indexEngine.addVideo(tar_videoNames);
    double addTime = (double)(clock() - beg) / CLOCKS_PER_SEC;


    printf("\n\n");
    string det_videoName = "d:/videos/det/det.mp4";

    beg = clock();
    vector<string> jsonResult;
    indexEngine.searchVideo(det_videoName, jsonResult);
    double searTime = (double)(clock() - beg) / CLOCKS_PER_SEC;

    stringstream strstream;
    for (auto &str : jsonResult)
    {
        //boost::property_tree::write_json(str, strstream);
        strstream << str;
    }

    FILE *fout = fopen("result.json", "w");

    fprintf(fout, "%s", strstream.str().c_str());


    printf("add video time=%lf\n", addTime);
    printf("search time = %lf\n", searTime);

    return 0;

}