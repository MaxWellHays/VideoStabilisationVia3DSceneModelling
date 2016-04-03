#pragma once
#include <opencv2/core/core.hpp>
#include <pba.h>
#include "cloud2d.h"
#include "cloud3d.h"

class cvPba
{
  static Point2D cvPba::ConvertCvPoint(const cv::Point2f &point);
  static vector<Point3D_<float>> generateRough3dPoints(const vector<cv::Point2f>& points, int width, int height);
public:
  static cloud3d RunBundleAdjustment(const std::pair<cloud2d, cloud2d> &imagePoints, cv::Mat &R, cv::Mat &T);
  static float getLockedMask(bool lockFocal, bool lockPosition = false, bool lockRotation = false, bool lockDistortion = false);
};