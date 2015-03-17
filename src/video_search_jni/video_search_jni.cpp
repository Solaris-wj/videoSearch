
#include "casia_isiteam_videosearch_slave_IndexJNI.h"

#include "../video_search/VideoIndexEngine.h"

#include <string>
#include <vector>

using namespace std;
using namespace vs;

shared_ptr<VideoIndexEngine> videoIndexEngine;

static shared_ptr<VideoIndexEngine>& getIndexEngine()
{
    return videoIndexEngine;
}

JNIEXPORT jint JNICALL Java_casia_isiteam_slave_IndexJNIimpl_initIndex
(JNIEnv * env, jobject, jstring jstrDataDir, jstring jstrLogDir, jstring jstrAlgoConfPath)
{
    const char * cstrDataDir = env->GetStringUTFChars(jstrDataDir, NULL);
    const char * cstrLogDir = env->GetStringUTFChars(jstrLogDir, NULL);
    const char * cstrAlgoConfPath = env->GetStringUTFChars(jstrAlgoConfPath, NULL);


    string dataDir(cstrDataDir);
    string logDir(cstrLogDir);
    string algoConfPath(cstrAlgoConfPath);

    env->ReleaseStringUTFChars(jstrDataDir,cstrDataDir);
    env->ReleaseStringUTFChars(jstrLogDir,cstrLogDir);
    env->ReleaseStringUTFChars(jstrAlgoConfPath,cstrAlgoConfPath);

    if (dataDir.empty() || logDir.empty() || algoConfPath.empty())
    {
        return -1;
    }

    getIndexEngine().reset(new VideoIndexEngine(dataDir, logDir, algoConfPath));

    return 0;
}

JNIEXPORT jint JNICALL Java_casia_isiteam_slave_IndexJNIimpl_addVideo
(JNIEnv *env, jobject, jstring jstrFilePath)
{
    const char * cstrFilePath = env->GetStringUTFChars(jstrFilePath, NULL);
    string filePath(cstrFilePath);
    if (filePath.empty())
    {
        return -1;
    }
    env->ReleaseStringUTFChars(jstrFilePath, cstrFilePath);

    return getIndexEngine()->addVideo(filePath);
}

JNIEXPORT jstring JNICALL Java_casia_isiteam_slave_IndexJNIimpl_searchVideo
(JNIEnv * env , jobject, jstring jstrFilePath)
{
    const char * cstrFilePath = env->GetStringUTFChars(jstrFilePath, NULL);
    string filePath(cstrFilePath);
    if (filePath.empty())
    {
        return NULL;
    }
    env->ReleaseStringUTFChars(jstrFilePath,  cstrFilePath);

    string jsonResult;
    getIndexEngine()->searchVideo(filePath, jsonResult);

    jstring jstrResult = env->NewStringUTF(jsonResult.c_str());

    return jstrResult;
}

JNIEXPORT jint JNICALL Java_casia_isiteam_slave_IndexJNIimpl_deleteVideo
(JNIEnv * env, jobject, jstring jstrFilePath)
{
    const char * cstrFilePath = env->GetStringUTFChars(jstrFilePath, NULL);
    string filePath(cstrFilePath);
    if (filePath.empty())
    {
        return -1;
    }
    env->ReleaseStringUTFChars(jstrFilePath, cstrFilePath);

   return getIndexEngine()->deleteVideo(filePath);
}