#include "cloud2d.h"
#include <basetsd.h>

cloud2d::cloud2d()
{
}

cloud2d::~cloud2d()
{
}

void cloud2d::addPoint(cv::Point2f point)
{
	points.push_back(point);
}

cv::Mat cloud2d::epipolarFilter(std::pair<cloud2d, cloud2d>& clouds)
{
	cv::Mat mask;
	auto fundamentalMat = findFundamentalMat(cv::Mat(clouds.first.points), cv::Mat(clouds.second.points), mask, cv::FM_RANSAC);

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
	return fundamentalMat;
}

void cloud2d::drawPoints(std::pair<cloud2d, cloud2d> &pair, cv::Mat& image1, cv::Mat& image2)
{
	for (int i = 0; i < pair.first.points.size(); ++i)
	{
		auto c = cv::Scalar(rand() & 255, rand() & 255, rand() & 255);
		cv::circle(image1, pair.first.points[i], 5, c, -1);
		cv::circle(image2, pair.second.points[i], 5, c, -1);
	}
}

void cloud2d::drawPointsAndEpipolarLines(std::pair<cloud2d, cloud2d>& pair, cv::Mat fundamental, cv::Mat& image1, cv::Mat& image2)
{
	std::vector<cv::Vec3f> lines1;
	std::vector<cv::Vec3f> lines2;
	computeCorrespondEpilines(cv::Mat(pair.first.points), 2, fundamental, lines1);
	computeCorrespondEpilines(cv::Mat(pair.second.points), 1, fundamental, lines2);
	for (int i = 0; i < pair.first.points.size(); ++i)
	{
		auto c = cv::Scalar(rand() & 255, rand() & 255, rand() & 255);
		cv::Vec3f *it = &lines1[i];
		cv::line(image1, cv::Point(0, -(*it)[2] / (*it)[1]), cv::Point(image1.cols, -((*it)[2] + (*it)[0] * image2.cols) / (*it)[1]), c);
		cv::circle(image1, pair.first.points[i], 5, c, -1);
		it = &lines2[i];
		cv::line(image2, cv::Point(0, -(*it)[2] / (*it)[1]), cv::Point(image1.cols, -((*it)[2] + (*it)[0] * image2.cols) / (*it)[1]), c);
		cv::circle(image2, pair.second.points[i], 5, c, -1);
	}
}

void cloud2d::shiftAll(cv::Point2f offset)
{
	for (auto& point : points)
	{
		point += offset;
	}
}

void cloud2d::center(cv::Size imageSize)
{
	shiftAll(cv::Point2f(-imageSize.width / 2.0, -imageSize.height / 2.0));
}