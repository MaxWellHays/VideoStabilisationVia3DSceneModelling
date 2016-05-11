#include "keypoints.h"
#include "enviroment.h"

cv::SIFT cw::keypoints::detector;
cv::FlannBasedMatcher cw::keypoints::matcher;

std::vector<cv::DMatch> cw::keypoints::getAllFirstBestMatches(const cv::Mat& descriptor1, const cv::Mat& descriptor2)
{
  std::vector<cv::DMatch> matches;
  matcher.match(descriptor1, descriptor2, matches);
  return matches;
}

std::vector<cv::DMatch> cw::keypoints::getAllKFirstBestMatches(const cv::Mat& descriptor1, const cv::Mat& descriptor2, int k)
{
  std::vector<std::vector<cv::DMatch>> matchesOfPoints;
  matcher.knnMatch(descriptor1, descriptor2, matchesOfPoints, k);
  std::vector<cv::DMatch> matches;
  for (auto && v : matchesOfPoints) {
    matches.insert(matches.end(), v.begin(), v.end());
  }
  return matches;
}

std::vector<cv::DMatch> cw::keypoints::getGoodMatches(const cv::Mat& descriptor1, const cv::Mat& descriptor2)
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

cw::keypoints::keypoints(cv::Mat &image)
{
  source_image = image;
  detector.detect(image, key_points);
  detector.compute(image, key_points, descriptor);
  std::cout << "Finded " << key_points.size() << " keypoint for image" << std::endl;
}

cv::Mat cw::keypoints::drawKeypoints(bool withBackground) const
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

cw::cloud2d cw::keypoints::toCloud2d() const
{
  cloud2d result;
  for (auto& keypoint : key_points)
  {
    result.addPoint(keypoint.pt);
  }
  return result;
}

std::vector<cw::keypoints> cw::keypoints::createKeypoints(std::vector<cv::Mat> &images)
{
  std::vector<keypoints> result;
  for (auto& image : images)
  {
    result.push_back(keypoints(image));
  }
  return result;
}

std::vector<cv::Mat> cw::keypoints::drawKeypoints(const std::vector<keypoints>& keypointsList, bool withBackground)
{
  cv::vector<cv::Mat> result;
  result.reserve(keypointsList.size());
  for (const keypoints& keypoints : keypointsList)
  {
    result.push_back(keypoints.drawKeypoints(withBackground));
  }
  return result;
}

std::vector<std::pair<cw::cloud2d, cw::cloud2d>> cw::keypoints::descriptorFilter(const std::vector<keypoints>& keypointses)
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

      result.back().first.addPoint(point1);
      result.back().second.addPoint(point2);
    }
  }
  return result;
}

std::vector<std::pair<cw::cloud2d, cw::cloud2d>> cw::keypoints::oldGoodDescriptorFilter(const std::vector<keypoints>& keypointses)
{
  std::vector<std::vector<cv::DMatch>> matches;
  for (int i = 0; i < keypointses.size() - 1; ++i)
  {
    matches.push_back(getGoodMatches(keypointses[i].descriptor, keypointses[i + 1].descriptor));
  }

  std::vector<std::pair<cloud2d, cloud2d>> result;
  for (auto k = 0; k < matches.size(); ++k)
  {
    result.push_back(std::make_pair(cloud2d(), cloud2d()));
    for (auto i = 0; i < matches[k].size(); ++i)
    {
      result.back().first.addPoint(keypointses[k].key_points[matches[k][i].queryIdx].pt);
      result.back().second.addPoint(keypointses[k + 1].key_points[matches[k][i].trainIdx].pt);
    }
  }
  return result;
}

std::vector<std::pair<cw::cloud2d, cw::cloud2d>> cw::keypoints::oldFirstDescriptorFilter(const std::vector<keypoints>& keypointses)
{
  std::vector<std::vector<cv::DMatch>> matches;
  for (int i = 0; i < keypointses.size() - 1; ++i)
  {
    matches.push_back(getAllFirstBestMatches(keypointses[i].descriptor, keypointses[i + 1].descriptor));
  }

  std::vector<std::pair<cloud2d, cloud2d>> result;
  for (auto k = 0; k < matches.size(); ++k)
  {
    result.push_back(std::make_pair(cloud2d(), cloud2d()));
    for (auto i = 0; i < matches[k].size(); ++i)
    {
      result.back().first.addPoint(keypointses[k].key_points[matches[k][i].queryIdx].pt);
      result.back().second.addPoint(keypointses[k + 1].key_points[matches[k][i].trainIdx].pt);
    }
  }
  return result;
}

std::pair<cw::cloud2d, cw::cloud2d> cw::keypoints::descriptorFilter(const std::pair<const keypoints&, const keypoints&>& keypointsPair)
{
  std::vector<cv::DMatch> matches(getAllKFirstBestMatches(keypointsPair.first.descriptor, keypointsPair.second.descriptor, 10));
  std::pair<cloud2d, cloud2d> result;
  for (auto i = 0; i < matches.size(); ++i)
  {
    cv::Point2f point1(keypointsPair.first.key_points[matches[i].queryIdx].pt);
    cv::Point2f point2(keypointsPair.second.key_points[matches[i].trainIdx].pt);

    result.first.addPoint(point1);
    result.second.addPoint(point2);
  }
  return result;
}

std::pair<cw::cloud2d, cw::cloud2d> cw::keypoints::smartFilter(const std::pair<const keypoints&, const keypoints&>& keypointsPair, cv::Mat& fundamentalMatrix)
{
  std::pair<cloud2d, cloud2d> allMatches(descriptorFilter(keypointsPair));
  std::pair<cloud2d, cloud2d> filteredBydistance1(cloud2d::filterByDistance(allMatches));
  fundamentalMatrix = cloud2d::epipolarFilter(filteredBydistance1);

  //std::pair<cloud2d, cloud2d> filteredBydistance2(cloud2d::filterByDistance(allMatches, 100));
  //std::pair<cloud2d, cloud2d> result(cloud2d::filterWithFundamentalMatrix(filteredBydistance2, fundamentalMatrix));
  return filteredBydistance1;
}
