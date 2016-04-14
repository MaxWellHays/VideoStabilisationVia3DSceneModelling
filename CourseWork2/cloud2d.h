#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class cloud2d
{
public:
  cloud2d();
  explicit cloud2d(const cv::Mat& pointsMatrix);
  explicit cloud2d(const std::string& nameForLoad);
  ~cloud2d();
  void dump(const std::string& name) const;
  void dump(std::ostream& out) const;
  std::vector<cv::Point2f> points;

  void addPoint(cv::Point2f point);

  static cv::Mat epipolarFilter(std::pair<cloud2d, cloud2d>& clouds, int method = cv::FM_RANSAC);
  static void filterWithFundamentalMatrix(std::pair<cloud2d, cloud2d>& clouds, const cv::Mat& fundamentalMat);

  static void drawMatches(std::pair<cloud2d, cloud2d> &pair, const cv::Mat &image1, const cv::Mat &image2);
  static cv::Mat drawMatches(const cloud2d& cloud1, const cloud2d& cloud2, bool drawLine = false);
  static cv::Mat drawMatches(const cloud2d& cloud1, const cloud2d& cloud2, const cv::Mat& backgroud, bool drawLine = false);

  static void drawPointsAndEpipolarLines(std::pair<cloud2d, cloud2d> &pair, const cv::Mat& fundamental, const cv::Mat& image1, const cv::Mat& image2);

  cloud2d shiftAll(cv::Point2f offset) const;

  cloud2d center(cv::Size imageSize) const;

  cv::Mat drawPoints(const cv::Mat& backgroundImage) const;

  cv::Mat drawPoints() const;
};

