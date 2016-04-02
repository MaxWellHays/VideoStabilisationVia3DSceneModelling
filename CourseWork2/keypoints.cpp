#include "keypoints.h"

cv::SIFT keypoints::detector;
cv::FlannBasedMatcher keypoints::matcher;

std::vector<cv::DMatch> keypoints::getAllMatches(cv::Mat& descriptor1, cv::Mat& descriptor2)
{
	std::vector<cv::DMatch> matches;
	matcher.match(descriptor1, descriptor2, matches);
	return matches;
}

std::vector<cv::DMatch> keypoints::getGoodMatches(cv::Mat& descriptor1, cv::Mat& descriptor2)
{
	std::vector<cv::DMatch> matches(getAllMatches(descriptor1, descriptor2));

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

std::vector<keypoints> keypoints::createKeypoints(std::vector<cv::Mat> &images)
{
	std::vector<keypoints> result;
	for (auto& image : images)
	{
		result.push_back(keypoints(image));
	}
	return result;
}

std::vector<std::pair<cloud2d, cloud2d>> keypoints::descriptorFilter(std::vector<keypoints>& keypointses)
{
	std::vector<std::vector<cv::DMatch>> matches;
	for (int i = 0; i < keypointses.size() - 1; ++i)
	{
		matches.push_back(getAllMatches(keypointses[i].descriptor, keypointses[i + 1].descriptor));
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