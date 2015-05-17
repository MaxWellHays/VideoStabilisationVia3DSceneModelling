#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include "dirent.h"
#include <regex>
#include "cvPba.h"
#include "keypoints.h"

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

std::pair<Mat, Mat> extractExtrinsicParameters(cv::Mat fundamentalMat, double f, double u0, double v0)
{
	cv::Mat R, T, K(cv::Mat::eye(3, 3, CV_64F)), essentialMat;
	K.at<double>(0, 0) = K.at<double>(1, 1) = f;
	K.at<double>(0, 2) = u0;
	K.at<double>(1, 2) = v0;

	essentialMat = K.t()*fundamentalMat*K;

	Mat w, u, vt;
	SVD::compute(essentialMat, w, u, vt, DECOMP_SVD);
	double wValues[9] = { 0.0, -1.0, 0.0, 1, 0.0, 0.0, 0.0, 0.0, 1.0 };
	Mat W(3, 3, CV_64F, wValues);
	R = u*W*vt;
	T = u.col(2);
	return make_pair(R, T);
}

int main(int argc, char** argv)
{
	auto folderPath("C:\\Users\\Maxim\\Documents\\coursework\\timelapse1\\");
	auto images(getImagesFromFolder(folderPath, regex(".+\\.((jpg)|(png))", regex_constants::icase)));
	if (images.size())
	{
		auto keypoints(keypoints::createKeypoints(images));
		auto cloud2dPairs(keypoints::descriptorFilter(keypoints));
		auto fundamentalMat = cloud2d::epipolarFilter(cloud2dPairs[0]);

		cloud2d::drawPointsAndEpipolarLines(cloud2dPairs[0], fundamentalMat, images[0], images[1]);
		showImages(images);

		auto pair = extractExtrinsicParameters(fundamentalMat, 900, images[0].size().width / 2.0, images[0].size().height / 2.0);
		cout << "R: " << endl << pair.first << endl;
		cout << "T: " << endl << pair.second << endl;

		cloud2dPairs[0].first.center(images[0].size());
		cloud2dPairs[0].second.center(images[1].size());

		cvPba pba;
		pba.RunBundleAdjustment(cloud2dPairs[0], pair.first, pair.second);
	}
	return 0;
}