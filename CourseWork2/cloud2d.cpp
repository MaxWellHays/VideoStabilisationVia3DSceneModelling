#include "cloud2d.h"
#include <basetsd.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

cloud2d::cloud2d()
{
}

cloud2d::cloud2d(const cv::Mat& pointsMatrix)
{
  points = std::vector<cv::Point2f>(pointsMatrix.rows);
  pointsMatrix.copyTo(points);
}

cloud2d::cloud2d(const std::string& nameForLoad)
{
  std::ifstream stream;
  stream.open("data\\cloud2d_" + nameForLoad + ".pl2");
  if (!stream.is_open())
  {
    throw std::runtime_error("Could not open file");
  }
  float x, y;
  while (stream >> x >> y)
  {
    addPoint(cv::Point2f(x, y));
  }
  stream.close();
}

cloud2d::~cloud2d()
{
}

void cloud2d::dump(const std::string& name) const
{
  std::ofstream stream;
  stream.open("data\\cloud2d_" + name + ".pl2");
  this->dump(stream);
  stream.close();
}

void cloud2d::dump(std::ostream& out) const
{
  for (auto& point : points)
  {
    out << point.x << " " << point.y << std::endl;
  }
}

void cloud2d::addPoint(cv::Point2f point)
{
  points.push_back(point);
}

cv::Mat cloud2d::epipolarFilter(std::pair<cloud2d, cloud2d>& clouds)
{
  cv::Mat mask;
  auto fundamentalMat = findFundamentalMat(cv::Mat(clouds.first.points), cv::Mat(clouds.second.points), mask, cv::FM_RANSAC);

  std::vector<cv::Point2f> filteredPoints1, filteredPoints2;
  for (int i = 0; i < mask.rows; ++i)
  {
    if (mask.at<UINT8>(i))
    {
      filteredPoints1.push_back(clouds.first.points[i]);
      filteredPoints2.push_back(clouds.second.points[i]);
    }
  }
  clouds.first.points = filteredPoints1;
  clouds.second.points = filteredPoints2;
  return fundamentalMat;
}

void cloud2d::drawMatches(std::pair<cloud2d, cloud2d> &pair, cv::Mat& image1, cv::Mat& image2)
{
  cv::Mat m1(cv::Mat::zeros(image1.size(), CV_8UC3));
  cv::Mat m2(cv::Mat::zeros(image2.size(), CV_8UC3));
  cv::Mat m3(cv::Mat::zeros(image2.size(), CV_8UC3));
  for (int i = 0; i < pair.first.points.size(); ++i)
  {
    auto c = cv::Scalar(rand() & 255, rand() & 255, rand() & 255);
    cv::circle(m1, pair.first.points[i], 5, c, -1);
    cv::circle(m3, pair.first.points[i], 5, c, -1);
    cv::circle(m2, pair.second.points[i], 5, c, -1);
    cv::circle(m3, pair.second.points[i], 5, c, -1);
    cv::line(m3, pair.first.points[i], pair.second.points[i], c, 2);
  }
}

void cloud2d::drawPointsAndEpipolarLines(std::pair<cloud2d, cloud2d>& pair, cv::Mat fundamental, cv::Mat& image1, cv::Mat& image2)
{
  std::vector<cv::Vec3f> lines1;
  std::vector<cv::Vec3f> lines2;
  computeCorrespondEpilines(cv::Mat(pair.first.points), 2, fundamental, lines1);
  computeCorrespondEpilines(cv::Mat(pair.second.points), 1, fundamental, lines2);
  for (int i = 0; i < pair.first.points.size(); ++i)
  {
    auto c = cv::Scalar(rand() & 255, rand() & 255, rand() & 255);
    cv::Vec3f *it = &lines1[i];
    cv::line(image1, cv::Point(0, -(*it)[2] / (*it)[1]), cv::Point(image1.cols, -((*it)[2] + (*it)[0] * image2.cols) / (*it)[1]), c);
    cv::circle(image1, pair.first.points[i], 5, c, -1);
    it = &lines2[i];
    cv::line(image2, cv::Point(0, -(*it)[2] / (*it)[1]), cv::Point(image1.cols, -((*it)[2] + (*it)[0] * image2.cols) / (*it)[1]), c);
    cv::circle(image2, pair.second.points[i], 5, c, -1);
  }
}

cloud2d cloud2d::shiftAll(cv::Point2f offset) const
{
  cloud2d result(*this);
  for (cv::Point2f& point : result.points)
  {
    point += offset;
  }
  return result;
}

cloud2d cloud2d::center(cv::Size imageSize) const
{
  return shiftAll(cv::Point2f(-imageSize.width / 2.0, -imageSize.height / 2.0));
}

cv::Mat cloud2d::drawPoints(const cv::Mat& backgroundImage) const
{
  cv::Mat resultImage = backgroundImage.clone();
  for (const cv::Point2f& currentPoint : points)
  {
    auto color = cv::Scalar(rand() & 255, rand() & 255, rand() & 255);
    cv::circle(resultImage, currentPoint, 5, color, -1);
  }
  return resultImage;
}

cv::Mat cloud2d::drawPoints() const
{
  cv::Mat background(cv::Mat::zeros(501, 750, CV_8UC3));
  return drawPoints(background);
}
