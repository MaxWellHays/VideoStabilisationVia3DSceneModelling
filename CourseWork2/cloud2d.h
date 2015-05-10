#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class cloud2d
{
public:
	cloud2d();
	~cloud2d();
	std::vector<cv::Point2f> points;

	void addPoint(cv::Point2f point);

	static cv::Mat epipolarFilter(std::pair<cloud2d, cloud2d>& clouds);

	static void drawPoints(std::pair<cloud2d, cloud2d> &pair, cv::Mat& image1, cv::Mat& image2);

	void shiftAll(cv::Point2f offset);

	void center(cv::Size imageSize);
};

