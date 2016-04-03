#include "cvPba.h"
#include <iostream>
#include <fstream>
#include "cloud3d.h"

Point2D cvPba::ConvertCvPoint(const cv::Point2f &point)
{
	return Point2D(point.x, point.y);
}

cv::Point3d cvPba::ConvertCvPoint(const Point3D &point)
{
  return cv::Point3f(point.xyz[0], point.xyz[1], point.xyz[2]);
}

vector<cv::Point3d> cvPba::ConvertCvPoint(const vector<Point3D> &points)
{
  vector<cv::Point3d> result;
  result.reserve(points.size());
  for (auto& point : points)
  {
    result.push_back(ConvertCvPoint(point));
  }
  return result;
}

vector<Point3D_<float>> cvPba::generateRough3dPoints(const vector<cv::Point2f>& points, int width, int height)
{
	vector<Point3D_<float>> result;
	result.reserve(points.size());

	for (auto& point : points)
	{
		result.push_back(Point3D());
		result.back().SetPoint(point.x * 2 / width, point.y * 2 / height, 10.0f);
	}
	return result;
}

float cvPba::getLockedMask(bool lockFocal, bool lockPosition, bool lockRotation, bool lockDistortion)
{
	return lockFocal*LOCK_FOCAL + lockPosition*LOCK_POSITION + lockRotation*LOCK_ROTATION + lockDistortion*LOCK_DISTORTION;
}

cloud3d cvPba::RunBundleAdjustment(const std::pair<cloud2d, cloud2d> &imagePoints, cv::Mat &R, cv::Mat &T)
{
	vector<CameraT>        camera_data;    //camera (input/ouput)
	vector<Point3D>        point_data;     //3D point(iput/output)
	vector<Point2D>        measurements;   //measurment/projection vector
	vector<int>            camidx, ptidx;  //index of camera/point for each projection

  auto firstCenteredCloud = imagePoints.first.center(cv::Size(750, 501));
  auto secondCenteredCloud = imagePoints.second.center(cv::Size(750, 501));

	for (int i = 0; i < firstCenteredCloud.points.size(); ++i)
	{
		measurements.push_back(ConvertCvPoint(firstCenteredCloud.points[i]));
		camidx.push_back(0);
		ptidx.push_back(i);
		
		measurements.push_back(ConvertCvPoint(secondCenteredCloud.points[i]));
		camidx.push_back(1);
		ptidx.push_back(i);
	}

	ParallelBA pba;

	camera_data.resize(2);

	camera_data[0].SetFocalLength(900);
	camera_data[0].constant_camera = getLockedMask(true, true, true);

	camera_data[1] = CameraT(camera_data[0]);
	camera_data[1].constant_camera = getLockedMask(true);

	for (int i = 0; i < R.size().height; ++i)
	{
		for (int j = 0; j < R.size().width; ++j)
		{
			camera_data[0].m[i][j] = 0;
			camera_data[1].m[i][j] = R.at<double>(i, j);
		}
		camera_data[0].m[i][i] = 1;
		camera_data[0].t[i] = 0;
		camera_data[1].t[i] = T.at<double>(i);
	}
	point_data = generateRough3dPoints(firstCenteredCloud.points, 750, 501);

	pba.SetCameraData(camera_data.size(), camera_data.data());
	pba.SetPointData(point_data.size(), &point_data[0]);
	pba.SetProjection(measurements.size(), measurements.data(), ptidx.data(), camidx.data());

	while (pba.RunBundleAdjustment() > 30);

	for (int i = 0; i < R.size().height; ++i)
	{
		for (int j = 0; j < R.size().width; ++j)
		{
			R.at<double>(i, j) = camera_data[1].m[i][j];
		}
		T.at<double>(i) = camera_data[1].t[i];
	}
  return cloud3d(ConvertCvPoint(point_data));
}