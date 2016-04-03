#pragma once
#include <opencv2/opencv.hpp>

class enviroment
{
public:
  static double defaultF;
  static void dumpMat(const cv::Mat& mat, const std::string& name);
  static cv::Mat loadMat(const std::string& name);
};

