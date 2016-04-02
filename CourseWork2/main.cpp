#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <dirent.h>
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

void extractExtrinsicParameters(cv::Mat fundamentalMat, double f, double u0, double v0, cv::Mat &T, cv::Mat &R)
{
  cv::Mat K(cv::Mat::eye(3, 3, CV_64F)), essentialMat;
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
}

int main(int argc, char** argv)
{
  auto folderPath("C:\\Users\\UX32VD\\Documents\\coursework\\timelapse1\\2\\");
  auto images(getImagesFromFolder(folderPath, regex(".+\\.((jpg)|(png))", regex_constants::icase)));
  if (images.size())
  {
    auto keypoints(keypoints::createKeypoints(images));
    auto cloud2dPairs(keypoints::descriptorFilter(keypoints));

    for (auto& cloud2DPair : cloud2dPairs)
    {
      cv::Mat fundamentalMat = cloud2d::epipolarFilter(cloud2dPairs[0]);

      #ifdef DEBUG
      cloud2d::drawPoints(cloud2dPairs[0], images[0], images[1]);
      showImages(images);
      #endif

      Mat R, T;
      extractExtrinsicParameters(fundamentalMat, 900,
        images[0].size().width / 2.0,
        images[0].size().height / 2.0,
        T, R);

      cloud2dPairs[0].first.center(images[0].size());
      cloud2dPairs[0].second.center(images[1].size());

      Mat r1 = R.clone();
      Mat t1 = T.clone();
      cvPba pba;
      pba.RunBundleAdjustment(cloud2dPairs[0], R, T);
    }
  }
  return 0;
}