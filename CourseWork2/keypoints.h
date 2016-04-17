#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "cloud2d.h"

namespace cw
{
  class keypoints
  {
    static cv::SIFT detector;
    static cv::FlannBasedMatcher matcher;

    std::vector<cv::KeyPoint> key_points;
    cv::Mat descriptor;
    cv::Mat source_image;

    static std::vector<cv::DMatch> getAllFirstBestMatches(cv::Mat& descriptor1, cv::Mat& descriptor2);
    static std::vector<cv::DMatch> getAllKFirstBestMatches(const cv::Mat& descriptor1, const cv::Mat& descriptor2, int k);
    static std::vector<cv::DMatch> getGoodMatches(cv::Mat& descriptor1, cv::Mat& descriptor2);
  public:
    explicit keypoints(cv::Mat &image);
    cv::Mat drawKeypoints(bool withBackground = false) const;
    cloud2d toCloud2d() const;
    static std::vector<keypoints> createKeypoints(std::vector<cv::Mat> &images);
    static std::vector<cv::Mat> drawKeypoints(const std::vector<keypoints> &keypointsList, bool withBackground = false);

    static std::vector<std::pair<cloud2d, cloud2d>> descriptorFilter(const std::vector<keypoints>& keypointsList);

    static std::pair<cloud2d, cloud2d> descriptorFilter(const std::pair<const keypoints&, const keypoints&>& keypointsPair);
    static std::pair<cloud2d, cloud2d> smartFilter(const std::pair<const keypoints&, const keypoints&>& keypointsPair, cv::Mat &fundamentalMatrix);
  };
}
