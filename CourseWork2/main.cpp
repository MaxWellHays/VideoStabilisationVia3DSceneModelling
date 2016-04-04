#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <dirent.h>
#include <regex>
#include "cvPba.h"
#include "keypoints.h"
#include "enviroment.h"

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

vector<cv::Mat> LoadImages()
{
  auto folderPath("C:\\Users\\UX32VD\\Documents\\coursework\\timelapse1\\2\\");
  return getImagesFromFolder(folderPath, std::regex(".+\\.((jpg)|(png))", std::regex_constants::icase));
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

void saveImage(const cv::Mat& image, const std::string& name)
{
  vector<int> compression_params;
  compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(9);
  cv::imwrite(name + ".png", image, compression_params);
}

void experementFunction()
{
  auto folderPath("C:\\Users\\UX32VD\\Documents\\coursework\\timelapse1\\2\\");
  vector<cv::Mat> images(getImagesFromFolder(folderPath, std::regex(".+\\.((jpg)|(png))", std::regex_constants::icase)));
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

      cv::Mat R, T;
      extractExtrinsicParameters(fundamentalMat, enviroment::defaultF,
        images[0].size().width / 2.0,
        images[0].size().height / 2.0,
        T, R);

      cloud3d cloud3d = cvPba::RunBundleAdjustment(cloud2DPair, R, T);

#ifdef DEBUG
      cloud2DPair.first.dump("pair1");
      cloud2DPair.second.dump("pair2");
      cloud3d.dump("bundleAdjustmentResult");
      enviroment::dumpMat(R, "R");
      enviroment::dumpMat(T, "T");
      cloud2d projectPoints = cloud3d.projectPoints(enviroment::defaultF, R, T);
      auto projectPointsImage = projectPoints.drawPoints();
      auto instantPointsImage1 = cloud2DPair.first.drawPoints();
      auto instantPointsImage2 = cloud2DPair.second.drawPoints();
#endif
    }
  }
}

std::vector<cv::Mat> generateIntermediateFrames(const cv::Mat& startFrame, const cv::Mat& finishFrame, int frameCount)
{
  int size = max(startFrame.cols, startFrame.rows);
  std::vector<cv::Mat> result(frameCount);
  for (size_t i = 0; i < frameCount; i++)
  {
    result[i] = (startFrame * (1 - i * 1.0 / frameCount)) + finishFrame * (i * 1.0 / frameCount);
  }
  return result;
}

int main(int argc, char** argv)
{
  //experementFunction();
  cloud2d points1("pair1");
  cloud2d points2("pair2");
  cv::Mat R(enviroment::loadMat("R"));
  cv::Mat T(enviroment::loadMat("T"));
  cloud3d spacePoints("bundleAdjustmentResult");

  cloud2d projectPoints1 = spacePoints.projectPoints(enviroment::defaultF);
  cloud2d projectPoints2 = spacePoints.projectPoints(enviroment::defaultF, R, T);

  int frameCount = 10;
  std::vector<cv::Mat> rotations(generateIntermediateFrames(cv::Mat::eye(3, 3, CV_64F), R, frameCount));
  std::vector<cv::Mat> transformations(generateIntermediateFrames(cv::Mat::zeros(3, 1, CV_64F), T, frameCount));

  std::vector<cv::Scalar> colors(spacePoints.vertexes.size());
  for (size_t i = 0; i < colors.size(); i++)
  {
    colors[i] = cv::Scalar(rand() & 255, rand() & 255, rand() & 255);
  }
  cv::Scalar blackColor(0, 0, 0);
  std::vector<cv::Mat> images;
  images.reserve(frameCount);
  for (size_t i = 0; i < frameCount; i++)
  {
    cv::Mat resultImage(cv::Mat::zeros(501, 750, CV_8UC3));
    cloud2d points(spacePoints.projectPoints(enviroment::defaultF, rotations[i], transformations[i]));
    for (size_t j = 0; j < points.points.size(); j++)
    {
      cv::circle(resultImage, points.points[j], 6, blackColor, -1);
      cv::circle(resultImage, points.points[j], 5, colors[j], -1);
    }
    images.push_back(resultImage);
    saveImage(resultImage, std::to_string(i));
  }
  showImages(images);

  return 0;
}