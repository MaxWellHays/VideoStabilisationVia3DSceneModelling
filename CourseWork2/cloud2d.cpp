#include "cloud2d.h"
#include <basetsd.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "enviroment.h"
#include <math.h>

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

cv::Mat cloud2d::epipolarFilter(std::pair<cloud2d, cloud2d>& clouds, int method)
{
  cv::Mat mask;
  cv::Mat pointsMat1(clouds.first.points), pointsMat2(clouds.second.points);
  auto fundamentalMat = findFundamentalMat(pointsMat1, pointsMat2, mask, method);

  if (method == cv::FM_RANSAC || method == cv::FM_LMEDS)
  {
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
  }
  else
  {
    cloud2d::filterWithFundamentalMatrix(clouds, fundamentalMat);
  }
  return fundamentalMat;
}

void cloud2d::filterWithFundamentalMatrix(std::pair<cloud2d, cloud2d>& clouds, const cv::Mat& fundamentalMat)
{
  std::vector<cv::Vec3f> lines1;
  std::vector<cv::Vec3f> lines2;
  cv::Mat pointsMat1(clouds.first.points), pointsMat2(clouds.second.points);
  computeCorrespondEpilines(pointsMat1, 1, fundamentalMat, lines1);
  computeCorrespondEpilines(pointsMat2, 2, fundamentalMat, lines2);

  /*enviroment::dumpMat(pointsMat1, "TestPoints1");
  enviroment::dumpMat(pointsMat2, "TestPoints2");
  cv::Mat linesMat1(lines1), linesMat2(lines2);
  enviroment::dumpMat(linesMat1, "TestLines1");
  enviroment::dumpMat(linesMat2, "TestLines2");
  enviroment::dumpMat(fundamentalMat, "TestFundmentalMatrix");*/

  std::vector<cv::Point2f> newPoints1;
  std::vector<cv::Point2f> newPoints2;

  int count1 = 0;
  int count5 = 0;
  int count10 = 0;
  for (size_t i = 0; i < lines1.size(); i++)
  {
    double distance1 = enviroment::distance(lines1[i], clouds.second.points[i]);
    double distance2 = enviroment::distance(lines2[i], clouds.first.points[i]);
    double maxDistance = MAX(distance1, distance2);
    if (maxDistance<10)
    {
      count10++;
      if (maxDistance<5)
      {
        newPoints1.push_back(clouds.first.points[i]);
        newPoints2.push_back(clouds.second.points[i]);
        count5++;
        if (maxDistance<1)
        {
          count1++;
        }
      }
    }
  }
  clouds.first.points = newPoints1;
  clouds.second.points = newPoints2;
}

void cloud2d::drawMatches(std::pair<cloud2d, cloud2d> &pair, const cv::Mat& image1, const cv::Mat& image2)
{
  cv::Mat m1(image1.clone());
  cv::Mat m2(image2.clone());
  //cv::Mat m1(cv::Mat::zeros(image2.size(), CV_8UC3));
  //cv::Mat m2(cv::Mat::zeros(image2.size(), CV_8UC3));
  cv::Mat m3(cv::Mat::zeros(image2.size(), CV_8UC3));
  for (int i = 0; i < pair.first.points.size(); ++i)
  {
    auto c = cv::Scalar(rand() & 255, rand() & 255, rand() & 255);
    cv::circle(m1, pair.first.points[i], 6, enviroment::blackColor, -1);
    cv::circle(m2, pair.second.points[i], 6, enviroment::blackColor, -1);

    cv::circle(m1, pair.first.points[i], 5, c, -1);
    cv::circle(m2, pair.second.points[i], 5, c, -1);

    cv::line(m3, pair.first.points[i], pair.second.points[i], enviroment::blackColor, 3);
    cv::line(m3, pair.first.points[i], pair.second.points[i], c, 2);

    cv::circle(m3, pair.first.points[i], 6, enviroment::blackColor, -1);
    cv::circle(m3, pair.second.points[i], 6, enviroment::blackColor, -1);
    cv::circle(m3, pair.first.points[i], 5, c, -1);
    cv::circle(m3, pair.second.points[i], 5, c, -1);
  }
}

cv::Mat cloud2d::drawMatches(const cloud2d& cloud1, const cloud2d& cloud2, bool drawLine)
{
  cv::Mat backgroud(cv::Mat::zeros(501, 750, CV_8UC3));
  return drawMatches(cloud1, cloud2, backgroud, drawLine);
}

cv::Mat cloud2d::drawMatches(const cloud2d& cloud1, const cloud2d& cloud2, const cv::Mat& backgroud, bool drawLine)
{
  if (cloud1.points.size() != cloud2.points.size())
  {
    throw std::runtime_error("Different size of clouds");
  }
  cv::Mat result = backgroud.clone();
  for (int i = 0; i < cloud1.points.size(); ++i)
  {
    auto c = cv::Scalar(rand() & 255, rand() & 255, rand() & 255);
    cv::circle(result, cloud1.points[i], 5, c, -1);
    cv::circle(result, cloud2.points[i], 5, c, -1);
    if (drawLine)
    {
      cv::line(result, cloud1.points[i], cloud2.points[i], c, 2);
    }
  }
  return result;
}

void cloud2d::drawPointsAndEpipolarLines(std::pair<cloud2d, cloud2d>& pair, const cv::Mat& fundamental, const cv::Mat& image1, const cv::Mat& image2)
{
  std::vector<cv::Vec3f> lines1;
  std::vector<cv::Vec3f> lines2;
  computeCorrespondEpilines(cv::Mat(pair.first.points), 2, fundamental, lines1);
  computeCorrespondEpilines(cv::Mat(pair.second.points), 1, fundamental, lines2);
  cv::Mat m1(image1.clone()), m2(image2.clone());
  for (int i = 0; i < pair.first.points.size(); i++)
  {
    auto c = cv::Scalar(rand() & 255, rand() & 255, rand() & 255);
    cv::Vec3f *it = &lines1[i];
    double dist1 = enviroment::distance(lines1[i], pair.first.points[i]);
    cv::line(m1, cv::Point(0, -(*it)[2] / (*it)[1]), cv::Point(image1.cols, -((*it)[2] + (*it)[0] * image2.cols) / (*it)[1]), c);
    cv::circle(m1, pair.first.points[i], 6, enviroment::blackColor, -1);
    cv::circle(m1, pair.first.points[i], 5, c, -1);
    it = &lines2[i];
    double dist2 = enviroment::distance(lines2[i], pair.first.points[i]);
    cv::line(m2, cv::Point(0, -(*it)[2] / (*it)[1]), cv::Point(image1.cols, -((*it)[2] + (*it)[0] * image2.cols) / (*it)[1]), c);
    cv::circle(m2, pair.second.points[i], 6, enviroment::blackColor, -1);
    cv::circle(m2, pair.second.points[i], 5, c, -1);
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
    cv::circle(resultImage, currentPoint, 6, enviroment::blackColor, -1);
    cv::circle(resultImage, currentPoint, 5, color, -1);
  }
  return resultImage;
}

cv::Mat cloud2d::drawPoints() const
{
  cv::Mat background(cv::Mat::zeros(501, 750, CV_8UC3));
  return drawPoints(background);
}
