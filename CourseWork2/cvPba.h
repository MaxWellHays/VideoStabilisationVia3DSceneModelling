#pragma once
#include <opencv2/core/core.hpp>
#include <pba.h>
#include "cloud2d.h"

class cvPba
{
	static Point2D cvPba::ConvertCvPoint(cv::Point2f &point);
	vector<Point3D_<float>> generateRough3dPoints(vector<cv::Point2f>& points, int width, int height);
public:
	cvPba();
	~cvPba();
	void RunBundleAdjustment(std::pair<cloud2d, cloud2d> &imagePoints, cv::Mat R, cv::Mat T);
	static float getLockedMask(bool lockFocal, bool lockPosition = false, bool lockRotation = false, bool lockDistortion = false);
};