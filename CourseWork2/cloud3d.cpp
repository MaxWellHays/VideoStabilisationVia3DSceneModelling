#include "cloud3d.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

cw::cloud3d::cloud3d()
{
}

cw::cloud3d::~cloud3d()
{
}

cw::cloud3d::cloud3d(const std::vector<cv::Point3d>& points)
{
  vertexes = std::vector<cv::Point3d>(points);
}

cw::cloud3d::cloud3d(const std::string& nameForLoad) : cloud3d()
{
  std::ifstream stream;
  stream.open("data\\cloud3d_" + nameForLoad + ".ply");
  if (!stream.is_open())
  {
    throw std::runtime_error("Could not open file");
  }
  loadFromPLY(stream);
  stream.close();
}

int cw::cloud3d::addPoint(cv::Point3d point)
{
  vertexes.push_back(point);
  return vertexes.size() - 1;
}

cw::cloud2d cw::cloud3d::projectPoints(double f, const cv::Mat &R, const cv::Mat &T) const
{
  cv::Mat A(cv::Mat::zeros(3, 3, CV_64F));

  A.at<double>(0, 0) = A.at<double>(1, 1) = f;
  A.at<double>(2, 2) = 1;

  A.at<double>(0, 2) = 750.0 / 2;
  A.at<double>(1, 2) = 501.0 / 2;

  cv::Mat rotationVector;
  cv::Rodrigues(R, rotationVector);

  cv::Mat objectPoints(vertexes);
  cv::Mat resultMat;
  cv::projectPoints(objectPoints, rotationVector, T, A, cv::Mat(), resultMat);

  return cloud2d(resultMat);
}

cw::cloud2d cw::cloud3d::projectPoints(double f) const
{
  cv::Mat R(cv::Mat::eye(3, 3, CV_64F));
  cv::Mat T(cv::Mat::zeros(3, 1, CV_64F));
  return projectPoints(f, R, T);
}

void cw::cloud3d::dump(const std::string& name) const
{
  dumpPLY("data\\cloud3d_" + name + ".ply");
}

cw::cloud3d cw::cloud3d::load(const std::string& name)
{
  return cloud3d(name);
}

void cw::cloud3d::dumpPLY(std::ostream& out) const
{
  out << "ply" << std::endl;
  out << "format ascii 1.0" << std::endl;
  out << "element vertex " << vertexes.size() << std::endl;
  out << "property float x" << std::endl;
  out << "property float y" << std::endl;
  out << "property float z" << std::endl;
  out << "end_header" << std::endl;

  for (unsigned i = 0; i < vertexes.size(); i++)
  {
    out << vertexes[i].x << " "
      << vertexes[i].y << " "
      << vertexes[i].z << " ";
    out << std::endl;
  }
}

void cw::cloud3d::loadFromPLY(std::ifstream& in)
{
  std::string line;
  for (size_t i = 0; i < 7; i++)
  {
    std::getline(in, line);
  }

  double x, y, z;
  while (in >> x >> y >> z)
  {
    addPoint(cv::Point3d(x, y, z));
  }
}

void cw::cloud3d::dumpPLY(const std::string& filePath) const
{
  std::ofstream stream;
  stream.open(filePath);
  this->dumpPLY(stream);
  stream.close();
}
