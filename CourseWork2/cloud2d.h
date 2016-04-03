#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class cloud2d
{
public:
  cloud2d();
  explicit cloud2d(const cv::Mat& pointsMatrix);
  ~cloud2d();
  std::vector<cv::Point2f> points;

  void addPoint(cv::Point2f point);

  static cv::Mat epipolarFilter(std::pair<cloud2d, cloud2d>& clouds);

  static void drawMatches(std::pair<cloud2d, cloud2d> &pair, cv::Mat &image1, cv::Mat &image2);

  static void drawPointsAndEpipolarLines(std::pair<cloud2d, cloud2d> &pair, cv::Mat fundamental, cv::Mat& image1, cv::Mat& image2);

  cloud2d shiftAll(cv::Point2f offset) const;

  cloud2d center(cv::Size imageSize) const;

  cv::Mat drawPoints(const cv::Mat& backgroundImage) const;

  cv::Mat drawPoints() const;
};

