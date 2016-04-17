#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "cloud2d.h"

namespace cw
{
  class cloud3d
  {
  private:
    cloud3d();
  public:
    ~cloud3d();
    explicit cloud3d(const std::vector<cv::Point3d>& points);
    explicit cloud3d(const std::string& nameForLoad);
    std::vector<cv::Point3d> vertexes;

    int addPoint(cv::Point3d point);
    cloud2d projectPoints(double f, const cv::Mat &R, const cv::Mat &T) const;
    cloud2d projectPoints(double f) const;

    void dump(const std::string& name) const;
    static cloud3d load(const std::string& name);
    void dumpPLY(const std::string& filePath) const;
    void dumpPLY(std::ostream &out) const;
    void loadFromPLY(std::ifstream &in);
  };
}
