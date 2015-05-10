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

int main(int argc, char** argv)
{
	auto folderPath("C:\\Users\\Maxim\\Documents\\coursework\\timelapse1\\");
	auto images(getImagesFromFolder(folderPath, regex(".+\\.((jpg)|(png))", regex_constants::icase)));
	if (images.size())
	{
		auto keypoints(keypoints::createKeypoints(images));
		auto cloud2dPairs(keypoints::descriptorFilter(keypoints));
		cloud2d::epipolarFilter(cloud2dPairs[0]);
		cloud2d::drawPoints(cloud2dPairs[0], images[0], images[1]);
		showImages(images);

		cloud2dPairs[0].first.center(images[0].size());
		cloud2dPairs[0].second.center(images[0].size());

		cvPba pba;
		pba.RunBundleAdjustment(cloud2dPairs[0]);

	}
	return 0;
}