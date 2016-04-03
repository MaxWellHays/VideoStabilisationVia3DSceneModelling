#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <dirent.h>
#include <regex>
#include "cvPba.h"
#include "keypoints.h"

#define DEBUG

vector<cv::Mat> getImagesFromFolder(std::string folderPath, std::regex nameFilter)
{
  vector<cv::Mat> images;
  DIR *dir;
  struct dirent *ent;
  if ((dir = opendir(folderPath.c_str())) != nullptr) {
    while ((ent = readdir(dir)) != nullptr) {
      auto fileName(ent->d_name);
      if (std::regex_match(fileName, nameFilter))
      {
        auto patchToImageFile = folderPath + std::string(fileName);
        auto image = cv::imread(patchToImageFile);
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

void showImages(const vector<cv::Mat>& images)
{
  auto selectedImage(0);
  cv::imshow("Image", images[selectedImage]);
  for (;;)
  {
    auto keyCode = cv::waitKey();
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

  cv::Mat w, u, vt;
  cv::SVD::compute(essentialMat, w, u, vt, cv::DECOMP_SVD);
  double wValues[9] = { 0.0, -1.0, 0.0, 1, 0.0, 0.0, 0.0, 0.0, 1.0 };
  cv::Mat W(3, 3, CV_64F, wValues);
  R = u*W*vt;
  T = u.col(2);
}



int main(int argc, char** argv)
{
  auto folderPath("C:\\Users\\UX32VD\\Documents\\coursework\\timelapse1\\2\\");
  auto images(getImagesFromFolder(folderPath, std::regex(".+\\.((jpg)|(png))", std::regex_constants::icase)));
  if (images.size())
  {
    auto keypoints(keypoints::createKeypoints(images));
    auto cloud2dPairs(keypoints::descriptorFilter(keypoints));

    for (int i = 0; i < cloud2dPairs.size(); i++)
    {
      std::pair<cloud2d, cloud2d>& cloud2DPair = cloud2dPairs[i];

      cv::Mat fundamentalMat = cloud2d::epipolarFilter(cloud2DPair);
      
      #ifdef DEBUG
      //cloud2d::drawMatches(cloud2DPair, images[i], images[i + 1]);
      #endif

      double defaultF = 900;

      cv::Mat R, T;
      extractExtrinsicParameters(fundamentalMat, defaultF,
        images[0].size().width / 2.0,
        images[0].size().height / 2.0,
        T, R);

      cv::Mat r1 = R.clone();
      cv::Mat t1 = T.clone();
      cloud3d cloud3d = cvPba::RunBundleAdjustment(cloud2DPair, R, T);

      #ifdef DEBUG
      cloud3d.dumpPLY("C:\\Users\\UX32VD\\Documents\\coursework\\test.ply");
      cloud2d projectPoints = cloud3d.projectPoints(defaultF, R, T);
      auto projectPointsImage = projectPoints.drawPoints();
      auto instantPointsImage1 = cloud2DPair.first.drawPoints();
      auto instantPointsImage2 = cloud2DPair.second.drawPoints();
      #endif
    }
  }
  return 0;
}