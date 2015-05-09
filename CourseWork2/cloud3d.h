#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;

class cloud3d
{
public:
	cloud3d();
	~cloud3d();
	vector<Point3d> vertexes;
	Scalar currentColor;

	int addPoint(Point3d point);
	void dumpPLY(std::ostream &out);
};

