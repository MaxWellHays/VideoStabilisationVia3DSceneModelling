#include "cloud3d.h"
#include <iostream>
#include <fstream>

cloud3d::cloud3d()
{
}

cloud3d::~cloud3d()
{
}

int cloud3d::addPoint(cv::Point3d point)
{
  vertexes.push_back(point);
  return vertexes.size() - 1;
}

cloud2d cloud3d::projectPoints(double f, const cv::Mat &R, const cv::Mat &T) const
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

void cloud3d::dumpPLY(std::ostream& out)
{
  out << "ply" << std::endl;
  out << "format ascii 1.0" << std::endl;
  out << "comment made by ViMouse software" << std::endl;
  out << "comment This file is a saved stereo-reconstruction" << std::endl;
  out << "element vertex " << vertexes.size() << std::endl;
  out << "property float x" << std::endl;
  out << "property float y" << std::endl;
  out << "property float z" << std::endl;
  out << "property uchar red" << std::endl;
  out << "property uchar green" << std::endl;
  out << "property uchar blue" << std::endl;
  out << "end_header" << std::endl;

  for (unsigned i = 0; i < vertexes.size(); i++)
  {
    out << vertexes[i].x << " "
      << vertexes[i].y << " "
      << vertexes[i].z << " ";
    out << unsigned(128) << " "
      << unsigned(128) << " "
      << unsigned(128) << std::endl;
  }
  //    SYNC_PRINT(("This 0x%X. Edges %d", this, edges.size()));

}

void cloud3d::dumpPLY(std::string filePath)
{
  std::ofstream stream;
  stream.open(filePath);
  this->dumpPLY(stream);
  stream.close();
}
