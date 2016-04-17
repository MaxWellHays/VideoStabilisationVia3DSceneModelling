#include "enviroment.h"
#include <basetsd.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

double cw::enviroment::defaultF = 900;
cv::Scalar cw::enviroment::blackColor = cv::Scalar(0, 0, 0);
cv::Scalar cw::enviroment::whiteColor = cv::Scalar(255, 255, 255);

void cw::enviroment::dumpMat(const cv::Mat& mat, const std::string& name)
{
  std::string fileName = "data\\mat_" + name + ".xml";
  cv::FileStorage fs(fileName, cv::FileStorage::WRITE);
  fs << name << mat;
}

cv::Mat cw::enviroment::loadMat(const std::string& name)
{
  std::string fileName = "data\\mat_" + name + ".xml";
  cv::FileStorage fs(fileName, cv::FileStorage::READ);
  cv::Mat result;
  fs[name] >> result;
  return result;
}

double cw::enviroment::distance(cv::Vec3f line, cv::Point2f point)
{
  double a = line[0];
  double b = line[1];
  double c = line[2];
  return abs(a*point.x + b*point.y + c) / sqrt(a*a + b*b);
}

double cw::enviroment::distance(cv::Point2f point1, cv::Point2f point2)
{
  return cv::norm(cv::Mat(point1), cv::Mat(point2));
}
