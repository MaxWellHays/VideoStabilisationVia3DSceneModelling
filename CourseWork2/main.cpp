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

void correctCameraMatrixCoeff(cw::cloud3d cloud3d, cw::cloud2d initialPoints, cv::Mat& R, cv::Mat& T)
{
  vector<double> errorSize(4, 0);
  for (size_t i = 0; i < 4; i++)
  {
    cv::Mat tempR = ((i / 2) * 2 - 1) * R;
    cv::Mat tempT = ((i % 2) * 2 - 1) * T;
    cw::cloud2d projectedPoints = cloud3d.projectPoints(cw::enviroment::defaultF, tempR, tempT);
    errorSize[i] = projectedPoints.errorOfMatches(initialPoints);
  }

  vector<double>::iterator minElement = std::min_element(errorSize.begin(), errorSize.end());
  int minElementIndex = std::distance(errorSize.begin(), minElement);

  R = ((minElementIndex / 2) * 2 - 1) * R;
  T = ((minElementIndex % 2) * 2 - 1) * T;
}

void experementFunction()
{
  auto folderPath("C:\\Users\\UX32VD\\Documents\\coursework\\timelapse1\\test2\\");
  vector<cv::Mat> images(getImagesFromFolder(folderPath, std::regex(".+\\.((jpg)|(png))", std::regex_constants::icase)));
  if (images.size())
  {
    std::vector<cw::keypoints> keypointsList(cw::keypoints::createKeypoints(images));

#ifdef DEBUG
    std::vector<cv::Mat> keypointsImages(cw::keypoints::drawKeypoints(keypointsList, true));
#endif

    for (int i = 0; i < keypointsList.size() - 1; i++)
    {
      std::pair<cw::keypoints, cw::keypoints> currentKeypointsPair(std::make_pair(keypointsList[i], keypointsList[i + 1]));

      cv::Mat fundamentalMat;
      std::pair<cw::cloud2d, cw::cloud2d>& cloud2DPair = cw::keypoints::smartFilter(currentKeypointsPair, fundamentalMat);

#ifdef DEBUG
      cw::cloud2d::drawMatches(cloud2DPair, images[i], images[i + 1]);
      cw::cloud2d::drawPointsAndEpipolarLines(cloud2DPair, fundamentalMat, images[i], images[i + 1]);
#endif

      cv::Mat R, T;
      extractExtrinsicParameters(fundamentalMat, cw::enviroment::defaultF,
        images[0].size().width / 2.0,
        images[0].size().height / 2.0,
        T, R);

      cw::cloud3d cloud3d = cw::cvPba::RunBundleAdjustment(cloud2DPair, R, T);

#ifdef DEBUG
      correctCameraMatrixCoeff(cloud3d, cloud2DPair.second, R, T);

      cw::cloud2d projectPoints1 = cloud3d.projectPoints(cw::enviroment::defaultF);
      cw::cloud2d projectPoints2 = cloud3d.projectPoints(cw::enviroment::defaultF, R, T);

      auto instancePoints1 = cloud2DPair.first.drawPoints();
      auto instancePoints2 = cloud2DPair.second.drawPoints();
      auto projectedPoints1 = projectPoints1.drawPoints();
      auto projectedPoints2 = projectPoints2.drawPoints();
      auto matchesImage1 = cw::cloud2d::drawMatches(cloud2DPair.first, projectPoints1, true);
      auto matchesImage2 = cw::cloud2d::drawMatches(cloud2DPair.second, projectPoints2, true);

      cloud2DPair.first.dump("pair1");
      cloud2DPair.second.dump("pair2");
      cloud3d.dump("bundleAdjustmentResult");
      cw::enviroment::dumpMat(R, "R");
      cw::enviroment::dumpMat(T, "T");
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

cv::Vec3d getEulerAngles(cv::Mat &rotCamerMatrix) {

  cv::Mat cameraMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ;
  cv::Vec3d eulerAngles;
  double* _r = rotCamerMatrix.ptr<double>();
  double projMatrix[12] = { _r[0],_r[1],_r[2],0,
    _r[3],_r[4],_r[5],0,
    _r[6],_r[7],_r[8],0 };

  cv::decomposeProjectionMatrix(cv::Mat(3, 4, CV_64FC1, projMatrix),
    cameraMatrix,
    rotMatrix,
    transVect,
    rotMatrixX,
    rotMatrixY,
    rotMatrixZ,
    eulerAngles);
  return eulerAngles;
}

void comparePointsZDepth()
{
  cw::cloud2d points1("pair1");
  cw::cloud2d points2("pair2");
  cv::Mat R(cw::enviroment::loadMat("R"));
  cv::Mat T(cw::enviroment::loadMat("T"));
  cw::cloud3d spacePoints("bundleAdjustmentResult");

  cw::cloud2d projectPoints1 = spacePoints.projectPoints(cw::enviroment::defaultF);
  cw::cloud2d projectPoints2 = spacePoints.projectPoints(cw::enviroment::defaultF, R, T);

  cv::Mat depthImage1(cv::imread("C:\\Users\\UX32VD\\Documents\\coursework\\testImages\\depth1.png", CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_COLOR));

  vector<cv::Point2d> result;

  for (size_t i = 0; i < projectPoints1.points.size(); i++)
  {
    int x = min(projectPoints1.points[i].x, depthImage1.cols - 1);
    int y = min(projectPoints1.points[i].y, depthImage1.rows - 1);
    
    auto vec = depthImage1.at<cv::Vec3b>(y, x);
    result.push_back(cv::Point2d());
    result[i].x = vec[0];
    result[i].y = cv::norm(spacePoints.vertexes[i]);
    std::cout << i << std::endl;
  }

  cv::Mat depthCompasionMap(result);
  cw::enviroment::dumpMat(depthCompasionMap, "DepthCompasion");
}

void loadDumpData()
{
  cw::cloud2d points1("pair1");
  cw::cloud2d points2("pair2");
  cv::Mat R(cw::enviroment::loadMat("R"));
  cv::Mat T(cw::enviroment::loadMat("T"));
  cw::cloud3d spacePoints("bundleAdjustmentResult");

  cw::cloud2d projectPoints1 = spacePoints.projectPoints(cw::enviroment::defaultF);
  cw::cloud2d projectPoints2 = spacePoints.projectPoints(cw::enviroment::defaultF, R, T);

  cv::Mat m1(cv::Mat::zeros(501, 750, CV_8UC4));
  cv::Mat m2(cv::Mat::zeros(501, 750, CV_8UC4));
  cv::Mat m3(cv::Mat::zeros(501, 750, CV_8UC4));

  cv::Mat t1(cv::Mat::zeros(501, 750, CV_8UC4));
  cv::Mat t2(cv::Mat::zeros(501, 750, CV_8UC4));
  cv::Mat t3(cv::Mat::zeros(501, 750, CV_8UC4));

  auto black = cv::Scalar(0, 0, 0, 255);
  for (size_t i = 0; i < points1.points.size(); i++)
  {
    auto color = cv::Scalar(rand() & 255, rand() & 255, rand() & 255, 255);
    cv::circle(m1, points1.points[i], 6, black, -1);
    cv::circle(m1, points1.points[i], 5, color, -1);
    cv::circle(m2, projectPoints1.points[i], 6, black, -1);
    cv::circle(m2, projectPoints1.points[i], 5, color, -1);

    cv::line(m3, points1.points[i], projectPoints1.points[i], black, 3);
    cv::circle(m3, points1.points[i], 6, black, -1);
    cv::circle(m3, points1.points[i], 5, color, -1);
    cv::circle(m3, projectPoints1.points[i], 6, black, -1);
    cv::circle(m3, projectPoints1.points[i], 5, color, -1);
    cv::line(m3, points1.points[i], projectPoints1.points[i], color, 2);

    cv::circle(t1, points2.points[i], 6, black, -1);
    cv::circle(t1, points2.points[i], 5, color, -1);
    cv::circle(t2, projectPoints2.points[i], 6, black, -1);
    cv::circle(t2, projectPoints2.points[i], 5, color, -1);

    cv::line(t3, points2.points[i], projectPoints2.points[i], black, 3);
    cv::circle(t3, points2.points[i], 6, black, -1);
    cv::circle(t3, projectPoints2.points[i], 6, black, -1);
    cv::circle(t3, points2.points[i], 5, color, -1);
    cv::circle(t3, projectPoints2.points[i], 5, color, -1);
    cv::line(t3, points2.points[i], projectPoints2.points[i], color, 2);
    cv::line(t3, points2.points[i], projectPoints2.points[i], color, 2);
  }

  saveImage(m1, "1");
  saveImage(m2, "2");
  saveImage(m3, "3");
  saveImage(t1, "4");
  saveImage(t2, "5");
  saveImage(t3, "6");
}

int main(int argc, char** argv)
{
  //experementFunction();
  //loadDumpData();
  comparePointsZDepth();
  return 0;
}