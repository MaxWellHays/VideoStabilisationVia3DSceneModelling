#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "cloud2d.h"

class keypoints
{
  static cv::SIFT detector;
  static cv::FlannBasedMatcher matcher;

  std::vector<cv::KeyPoint> key_points;
  cv::Mat descriptor;
  cv::Mat source_image;

  static std::vector<cv::DMatch> getAllMatches(cv::Mat& descriptor1, cv::Mat& descriptor2);
  static std::vector<cv::DMatch> getGoodMatches(cv::Mat& descriptor1, cv::Mat& descriptor2);
public:
  explicit keypoints(cv::Mat &image);
  static std::vector<keypoints> createKeypoints(std::vector<cv::Mat> &images);
  static std::vector<std::pair<cloud2d, cloud2d>> descriptorFilter(std::vector<keypoints>& keypointses);
};

