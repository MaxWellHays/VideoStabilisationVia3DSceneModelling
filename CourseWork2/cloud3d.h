#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "cloud2d.h"


class cloud3d
{
public:
	cloud3d();
	~cloud3d();
	std::vector<cv::Point3d> vertexes;
  cv::Scalar currentColor;

	int addPoint(cv::Point3d point);
  cloud2d projectPoints(double f, const cv::Mat &R, const cv::Mat &T) const;

  void dumpPLY(std::ostream &out);
  void dumpPLY(std::string filePath);
};

