#include "cloud2d.h"
#include <basetsd.h>
#include <fstream>
#include "enviroment.h"

cw::cloud2d::cloud2d()
{
}

cw::cloud2d::cloud2d(const cv::Mat& pointsMatrix)
{
  points = std::vector<cv::Point2f>(pointsMatrix.rows);
  pointsMatrix.copyTo(points);
}

cw::cloud2d::cloud2d(const std::string& nameForLoad)
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

cw::cloud2d::~cloud2d()
{
}

void cw::cloud2d::dump(const std::string& name) const
{
  std::ofstream stream;
  stream.open("data\\cloud2d_" + name + ".pl2");
  this->dump(stream);
  stream.close();
}

void cw::cloud2d::dump(std::ostream& out) const
{
  for (auto& point : points)
  {
    out << point.x << " " << point.y << std::endl;
  }
}

void cw::cloud2d::addPoint(cv::Point2f point)
{
  points.push_back(point);
}

cv::Mat cw::cloud2d::epipolarFilter(std::pair<cw::cloud2d, cw::cloud2d>& clouds, int method)
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
    //cloud2d::filterWithFundamentalMatrix(clouds, fundamentalMat);
  }
  return fundamentalMat;
}

std::vector<double> cw::cloud2d::epipolarRestrictionValues(const std::pair<const cloud2d&, const cloud2d&>& clouds, const cv::Mat& fundamentalMat)
{
  std::vector<double> distances;
  cv::Mat hPoints1, hPoints2;
  cv::convertPointsToHomogeneous(clouds.first.points, hPoints1);
  cv::convertPointsToHomogeneous(clouds.second.points, hPoints2);
  hPoints1 = hPoints1.reshape(1);
  hPoints2 = hPoints2.reshape(1).t();

  for (size_t i = 0; i < hPoints1.rows; i++)
  {
    cv::Mat row = hPoints1.row(i);
    cv::Mat col = hPoints2.col(i);
    row.convertTo(row, CV_64FC1);
    col.convertTo(col, CV_64FC1);
    cv::Mat r = row * fundamentalMat * col;
    distances.push_back(abs(r.at<double>(0, 0)));
  }
  return distances;
}

std::pair<cw::cloud2d, cw::cloud2d> cw::cloud2d::filterWithFundamentalMatrix(const std::pair<const cloud2d&, const cloud2d&>& clouds, const cv::Mat& fundamentalMat)
{
  std::pair<cw::cloud2d, cw::cloud2d> result;

  std::vector<double> distances(epipolarRestrictionValues(clouds, fundamentalMat));
  std::vector<double> distancesCopy(distances.begin(), distances.end());

  std::sort(distancesCopy.begin(), distancesCopy.end());
  double trashhold = distancesCopy[distancesCopy.size() / 10];

  for (size_t i = 0; i < distances.size(); i++)
  {
    if (distances[i] <= trashhold)
    {
      result.first.addPoint(clouds.first.points[i]);
      result.second.addPoint(clouds.second.points[i]);
    }
  }

  return result;
}

void cw::cloud2d::drawMatches(std::pair<cw::cloud2d, cw::cloud2d> &pair, const cv::Mat& image1, const cv::Mat& image2)
{
  srand(enviroment::randomSeed);
  cv::Mat m1(image1.clone());
  cv::Mat m2(image2.clone());
  //cv::Mat m1(cv::Mat::zeros(image2.size(), CV_8UC3));
  //cv::Mat m2(cv::Mat::zeros(image2.size(), CV_8UC3));
  cv::Mat m3(cv::Mat::zeros(image2.size(), CV_8UC3));
  for (int i = 0; i < pair.first.points.size(); ++i)
  {
    auto c = cv::Scalar(rand() & 255, rand() & 255, rand() & 255);
    cv::circle(m1, pair.first.points[i], 6, cw::enviroment::blackColor, -1);
    cv::circle(m2, pair.second.points[i], 6, cw::enviroment::blackColor, -1);

    cv::circle(m1, pair.first.points[i], 5, c, -1);
    cv::circle(m2, pair.second.points[i], 5, c, -1);

    cv::line(m3, pair.first.points[i], pair.second.points[i], cw::enviroment::blackColor, 3);
    cv::line(m3, pair.first.points[i], pair.second.points[i], c, 2);

    cv::circle(m3, pair.first.points[i], 6, cw::enviroment::blackColor, -1);
    cv::circle(m3, pair.second.points[i], 6, cw::enviroment::blackColor, -1);
    cv::circle(m3, pair.first.points[i], 5, c, -1);
    cv::circle(m3, pair.second.points[i], 5, c, -1);
  }
}

cv::Mat cw::cloud2d::drawMatches(const cw::cloud2d& cloud1, const cw::cloud2d& cloud2, bool drawLine)
{
  cv::Mat backgroud(cv::Mat::zeros(501, 750, CV_8UC3));
  return drawMatches(cloud1, cloud2, backgroud, drawLine);
}

cv::Mat cw::cloud2d::drawMatches(const cw::cloud2d& cloud1, const cw::cloud2d& cloud2, const cv::Mat& backgroud, bool drawLine)
{
  srand(enviroment::randomSeed);
  if (cloud1.points.size() != cloud2.points.size())
  {
    throw std::runtime_error("Different size of clouds");
  }
  cv::Mat result = backgroud.clone();
  for (int i = 0; i < cloud1.points.size(); ++i)
  {
    auto c = cv::Scalar(rand() & 255, rand() & 255, rand() & 255);
    cv::circle(result, cloud1.points[i], 6, cw::enviroment::blackColor, -1);
    cv::circle(result, cloud2.points[i], 6, cw::enviroment::blackColor, -1);
    if (drawLine)
    {
      cv::line(result, cloud1.points[i], cloud2.points[i], cw::enviroment::blackColor, 3);
    }
    cv::circle(result, cloud1.points[i], 5, c, -1);
    cv::circle(result, cloud2.points[i], 5, c, -1);
    if (drawLine)
    {
      cv::line(result, cloud1.points[i], cloud2.points[i], c, 2);
    }
  }
  return result;
}

void cw::cloud2d::drawPointsAndEpipolarLines(std::pair<cw::cloud2d, cw::cloud2d>& pair, const cv::Mat& fundamental, const cv::Mat& image1, const cv::Mat& image2)
{
  srand(enviroment::randomSeed);
  std::vector<cv::Vec3f> lines1;
  std::vector<cv::Vec3f> lines2;
  computeCorrespondEpilines(cv::Mat(pair.first.points), 2, fundamental, lines1);
  computeCorrespondEpilines(cv::Mat(pair.second.points), 1, fundamental, lines2);
  cv::Mat m1(image1.clone()), m2(image2.clone());
  for (int i = 0; i < pair.first.points.size(); i++)
  {
    auto c = cv::Scalar(rand() & 255, rand() & 255, rand() & 255);
    cv::Vec3f *it = &lines1[i];
    double dist1 = cw::enviroment::distance(lines1[i], pair.first.points[i]);
    cv::line(m1, cv::Point(0, -(*it)[2] / (*it)[1]), cv::Point(image1.cols, -((*it)[2] + (*it)[0] * image2.cols) / (*it)[1]), c);
    cv::circle(m1, pair.first.points[i], 6, cw::enviroment::blackColor, -1);
    cv::circle(m1, pair.first.points[i], 5, c, -1);
    it = &lines2[i];
    double dist2 = cw::enviroment::distance(lines2[i], pair.first.points[i]);
    cv::line(m2, cv::Point(0, -(*it)[2] / (*it)[1]), cv::Point(image1.cols, -((*it)[2] + (*it)[0] * image2.cols) / (*it)[1]), c);
    cv::circle(m2, pair.second.points[i], 6, cw::enviroment::blackColor, -1);
    cv::circle(m2, pair.second.points[i], 5, c, -1);
  }
}

cw::cloud2d cw::cloud2d::shiftAll(cv::Point2f offset) const
{
  cw::cloud2d result(*this);
  for (cv::Point2f& point : result.points)
  {
    point += offset;
  }
  return result;
}

std::pair<cw::cloud2d, cw::cloud2d> cw::cloud2d::filterByDistance(const std::pair<const cw::cloud2d, const cloud2d>& pair, double distance)
{
  std::pair<cloud2d, cloud2d> result;
  for (int i = 0; i < pair.first.points.size(); i++)
  {
    if (enviroment::distance(pair.first.points[i], pair.second.points[i]) < distance)
    {
      result.first.addPoint(pair.first.points[i]);
      result.second.addPoint(pair.second.points[i]);
    }
  }
  return result;
}

cw::cloud2d cw::cloud2d::center(cv::Size imageSize) const
{
  return shiftAll(cv::Point2f(-imageSize.width / 2.0, -imageSize.height / 2.0));
}

cv::Mat cw::cloud2d::drawPoints(const cv::Mat& backgroundImage) const
{
  srand(enviroment::randomSeed);
  cv::Mat resultImage = backgroundImage.clone();
  for (const cv::Point2f& currentPoint : points)
  {
    auto color = cv::Scalar(rand() & 255, rand() & 255, rand() & 255);
    cv::circle(resultImage, currentPoint, 6, enviroment::blackColor, -1);
    cv::circle(resultImage, currentPoint, 5, color, -1);
  }
  return resultImage;
}

cv::Mat cw::cloud2d::drawPoints() const
{
  cv::Mat background(cv::Mat::zeros(501, 750, CV_8UC3));
  return drawPoints(background);
}

double cw::cloud2d::errorOfMatches(const cloud2d& anotherCloud2d) const
{
  double result(0);
  for (size_t i = 0; i < points.size(); i++)
  {
    double distance = enviroment::distance(points[i], anotherCloud2d.points[i]);
    result += distance * distance;
  }
  return result;
}
