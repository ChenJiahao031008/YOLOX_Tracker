#ifndef _OBJECTTRACKING_H_
#define _OBJECTTRACKING_H_

#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <dirent.h>
#include <algorithm>
#include <sys/time.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "json.h"
#include "yolox.h"
#include "kcftracker.h"

struct sys_config
{
    bool hog;
    bool fixed_window;
    bool multi_scale;
    bool silent;
    bool lab;
    float scale_step;
    int num_scales;
};

// struct Object;

class ObjectTracking
{
private:
    bool HOG = true;
    bool FIXEDWINDOW = false;
    bool MULTISCALE = true;
    bool SILENT = false;
    bool LAB = false;

    sys_config config;

    // KCFTracker tracker();

    std::vector<KCFTracker> vTrackers;

    cv::Mat keyFrame;

public:
    ObjectTracking(const std::string &jsonFile);

    ~ObjectTracking();

    bool parse_config(const std::string &path, sys_config &config);

    void RunTracker(cv::Mat &image, std::vector<Object> &vObject);

    void InitTracker(cv::Mat &frame, std::vector<Object> &vObject);

    void ModifyTracker(cv::Mat &image, std::vector<Object> &vObject);

    void Histogram(const cv::Mat &image, cv::Mat &hist);

    int CompareHist(const cv::Mat &hist1, const cv::Mat &hist2);

    int ImageSimilarityFlag(const cv::Mat &currentFrame, std::vector<Object> &vObject);
};

#endif

