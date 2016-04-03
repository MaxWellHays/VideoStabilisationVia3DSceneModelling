#include "enviroment.h"
#include <basetsd.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

double enviroment::defaultF = 900;

void enviroment::dumpMat(const cv::Mat& mat, const std::string& name)
{
  std::string fileName = "data\\mat_" + name + ".xml";
  cv::FileStorage fs(fileName, cv::FileStorage::WRITE);
  fs << name << mat;
}

cv::Mat enviroment::loadMat(const std::string& name)
{
  std::string fileName = "data\\mat_" + name + ".xml";
  cv::FileStorage fs(fileName, cv::FileStorage::READ);
  cv::Mat result;
  fs[name] >> result;
  return result;
}
