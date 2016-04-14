#include "keypoints.h"
#include "enviroment.h"

cv::SIFT keypoints::detector;
cv::FlannBasedMatcher keypoints::matcher;

std::vector<cv::DMatch> keypoints::getAllFirstBestMatches(cv::Mat& descriptor1, cv::Mat& descriptor2)
{
  std::vector<cv::DMatch> matches;
  matcher.match(descriptor1, descriptor2, matches);
  return matches;
}

std::vector<cv::DMatch> keypoints::getAllKFirstBestMatches(cv::Mat& descriptor1, cv::Mat& descriptor2, int k)
{
  std::vector<std::vector<cv::DMatch>> matchesOfPoints;
  matcher.knnMatch(descriptor1, descriptor2, matchesOfPoints, k);
  std::vector<cv::DMatch> matches;
  for (auto && v : matchesOfPoints) {
    matches.insert(matches.end(), v.begin(), v.end());
  }
  return matches;
}

std::vector<cv::DMatch> keypoints::getGoodMatches(cv::Mat& descriptor1, cv::Mat& descriptor2)
{
  std::vector<cv::DMatch> matches(getAllFirstBestMatches(descriptor1, descriptor2));

  double max_dist = 0; double min_dist = 100;
  for (auto i = 0; i < descriptor1.rows; i++)
  {
    double dist = matches[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }

  std::vector<cv::DMatch > good_matches;

  for (auto i = 0; i < descriptor1.rows; i++)
  {
    if (matches[i].distance < 3 * min_dist)
    {
      good_matches.push_back(matches[i]);
    }
  }
  return good_matches;
}

keypoints::keypoints(cv::Mat &image)
{
  source_image = image;
  detector.detect(image, key_points);
  detector.compute(image, key_points, descriptor);
  std::cout << "Finded " << key_points.size() << " keypoint for image" << std::endl;
}

cv::Mat keypoints::drawKeypoints(bool withBackground) const
{
  cloud2d cloud = this->toCloud2d();
  if (withBackground)
  {
    return cloud.drawPoints(this->source_image);
  }
  else
  {
    return cloud.drawPoints();
  }
}

cloud2d keypoints::toCloud2d() const
{
  cloud2d result;
  for (auto& keypoint : key_points)
  {
    result.addPoint(keypoint.pt);
  }
  return result;
}

std::vector<keypoints> keypoints::createKeypoints(std::vector<cv::Mat> &images)
{
  std::vector<keypoints> result;
  for (auto& image : images)
  {
    result.push_back(keypoints(image));
  }
  return result;
}

std::vector<cv::Mat> keypoints::drawKeypoints(const std::vector<keypoints>& keypointsList, bool withBackground)
{
  cv::vector<cv::Mat> result;
  result.reserve(keypointsList.size());
  for (const keypoints& keypoints : keypointsList)
  {
    result.push_back(keypoints.drawKeypoints(withBackground));
  }
  return result;
}

std::vector<std::pair<cloud2d, cloud2d>> keypoints::descriptorFilter(std::vector<keypoints>& keypointses)
{
  std::vector<std::vector<cv::DMatch>> matches;
  for (int i = 0; i < keypointses.size() - 1; ++i)
  {
    matches.push_back(getAllKFirstBestMatches(keypointses[i].descriptor, keypointses[i + 1].descriptor, 10));
  }

  std::vector<std::pair<cloud2d, cloud2d>> result;
  for (auto k = 0; k < matches.size(); ++k)
  {
    result.push_back(std::make_pair(cloud2d(), cloud2d()));
    for (auto i = 0; i < matches[k].size(); ++i)
    {
      cv::Point2f point1(keypointses[k].key_points[matches[k][i].queryIdx].pt);
      cv::Point2f point2(keypointses[k + 1].key_points[matches[k][i].trainIdx].pt);

      if (enviroment::distance(point1, point2) < 50)
      {
        result.back().first.addPoint(point1);
        result.back().second.addPoint(point2);
      }
    }
  }
  return result;
}
