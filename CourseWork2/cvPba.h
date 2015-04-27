#pragma once
#include <opencv2/core/core.hpp>
#include <pba.h>
class cvPba
{
	ParallelBA::DeviceT device;
	static Point2D cvPba::ConvertCvPoint(cv::Point2f &point);
public:
	cvPba();
	~cvPba();
	vector<Point3D_<float>> generateRough3dPoints(vector<cv::Point2f>& points, int width, int height);
	void RunBundleAdjustment(vector<vector<cv::Point2f>> &imagePoints);
};