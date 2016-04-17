#pragma once
#include <opencv2/opencv.hpp>

namespace cw
{
  class enviroment
  {
  public:
    static double defaultF;
    static cv::Scalar blackColor;
    static cv::Scalar whiteColor;
    static void dumpMat(const cv::Mat& mat, const std::string& name);
    static cv::Mat loadMat(const std::string& name);
    static double distance(cv::Vec3f line, cv::Point2f point);
    static double distance(cv::Point2f point1, cv::Point2f point2);
  };
}


