#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include "dirent.h"
#include <regex>
#include <iostream>

using namespace std;
using namespace cv;

vector<Mat> getImagesFromFolder(string folderPath, regex nameFilter)
{
	vector<Mat> images;
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir(folderPath.c_str())) != nullptr) {
		while ((ent = readdir(dir)) != nullptr) {
			auto fileName(ent->d_name);
			if (regex_match(fileName, nameFilter))
			{
				auto patchToImageFile = folderPath + string(fileName);
				auto image = imread(patchToImageFile);
				images.push_back(image);
				printf("%s\n", patchToImageFile.c_str());
			}
		}
		closedir(dir);
	}
	else {
		throw;
	}
	return images;
}

void showImages(const vector<Mat>& images)
{
	auto selectedImage(0);
	imshow("Image", images[selectedImage]);
	for (;;)
	{
		auto keyCode = waitKey();
		if (keyCode == 2424832 && selectedImage > 0) selectedImage--; //Left arrow
		else if (keyCode == 2555904 && selectedImage < images.size() - 1) selectedImage++; //Right arrow
		else if (keyCode == 27) break; //Esc
		else continue;
		imshow("Image", images[selectedImage]);
	}
}

void getKeypoints(const vector<Mat>& images, vector<vector<KeyPoint>>& keypoints, vector<Mat>& descriptors)
{
	keypoints.clear();
	descriptors.clear();

	SurfFeatureDetector detector;
	detector.detect(images, keypoints);

	SurfDescriptorExtractor extractor;
	for (size_t i = 0; i < images.size(); i++)
	{
		descriptors.push_back(Mat());
		extractor.compute(images[i], keypoints[i], descriptors[i]);
		printf("compute %i/%i\n", i + 1, images.size());
	}
}

vector<DMatch> get_good_matches(Mat& descriptor1, Mat& descriptor2, FlannBasedMatcher& matcher)
{
	vector< DMatch > matches;
	matcher.match(descriptor1, descriptor2, matches);
	double max_dist = 0; double min_dist = 100;

	for (auto i = 0; i < descriptor1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);

	vector< DMatch > good_matches;

	for (auto i = 0; i < descriptor1.rows; i++)
	{
		if (matches[i].distance < 2 * min_dist)
		{
			good_matches.push_back(matches[i]);
		}
	}
	return good_matches;
}

vector<vector<DMatch>> get_good_matches(vector<Mat>& descriptors)
{
	FlannBasedMatcher matcher;
	vector<vector<DMatch>> matches;
	for (auto i = 0; i < descriptors.size() - 1; i++)
	{
		matches.push_back(get_good_matches(descriptors[i], descriptors[i + 1], matcher));
	}
	return matches;
}

void getPointPairs(vector<DMatch>& mathces, vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2, vector<Point2f>& points1, vector<Point2f>& points2)
{
	points1.clear();
	points2.clear();
	for (auto i = 0; i < mathces.size(); ++i)
	{
		points1.push_back(keypoints1[mathces[i].queryIdx].pt);
		points2.push_back(keypoints2[mathces[i].trainIdx].pt);
	}
}

static Scalar randomColor()
{
	return Scalar(rand() & 255, rand() & 255, rand() & 255);
}

void fiter_points(vector<Point2f>& points1, vector<Point2f>& points2, Mat& mask)
{
	vector<Point2f> filteredPoints1, filteredPoints2;
	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<UINT8>(i))
		{
			filteredPoints1.push_back(points1[i]);
			filteredPoints2.push_back(points2[i]);
		}
	}
	points1 = filteredPoints1;
	points2 = filteredPoints2;
}

int main(int argc, char** argv)
{
	auto folderPath("C:\\Users\\Maxim\\Documents\\coursework\\timelapse1\\");
	auto images(getImagesFromFolder(folderPath, regex(".+\\.jpg", regex_constants::icase)));
	if (images.size())
	{
		vector<vector<KeyPoint>> keypoints;
		vector<Mat> descriptors;

		getKeypoints(images, keypoints, descriptors);
		vector<vector<DMatch>> good_mathces(get_good_matches(descriptors));
		vector<Point2f> points1, points2;
		getPointPairs(good_mathces[0], keypoints[0], keypoints[1], points1, points2);
		Mat mask;
		Mat F = findFundamentalMat(Mat(points1), Mat(points2), mask, FM_RANSAC);

		fiter_points(points1, points2, mask);

		Mat h1, h2;
		stereoRectifyUncalibrated(points1, points2, F, images[0].size(), h1, h2);

		Mat rect1;
		warpPerspective(images[0], rect1, h1, images[0].size());
		Mat rect2;
		warpPerspective(images[1], rect2, h2, images[1].size());

		StereoBM stereo(CV_STEREO_BM_BASIC, 16);
		Mat r;
		cvtColor(rect1, rect1, CV_BGR2GRAY);
		cvtColor(rect2, rect2, CV_BGR2GRAY);
		stereo(rect1, rect2, r);

		images.push_back(r);

		showImages(images);
	}
	return 0;
}