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
	void RunBundleAdjustment(vector<vector<cv::Point2f>> &imagePoints);
};