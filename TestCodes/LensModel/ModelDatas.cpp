#include <iostream>
#include "Rotation.h"
#include "CameraModel.h"
#include "ModelData.h"

using namespace FishEye;


void randomImagePointPairs(std::shared_ptr<CameraModel> &cam, const Rotation &rot, int num, 
						   std::vector<cv::Point2d> &vImgPt1, std::vector<cv::Point2d> &vImgPt2,
						   std::vector<cv::Point3d> &vSpherePt1, std::vector<cv::Point3d> &vSpherePt2)
{
	if (!vImgPt1.empty())vImgPt1.clear();
	if (!vImgPt2.empty())vImgPt2.clear();
	if (!vSpherePt1.empty())vSpherePt1.clear();
	if (!vSpherePt2.empty())vSpherePt2.clear();

	int count = 0;
	while (count < num)
	{
		double phi = cam->fov * (rand() / double(RAND_MAX)) * 0.5;
		double theta = CV_2PI * (rand() / double(RAND_MAX));

		cv::Point3d spherePt(sin(phi)*cos(theta), sin(phi)*sin(theta), cos(phi));
		cv::Point3d spherePtByRot = RotatePoint(spherePt, rot);

		double phiByRot = atan2(sqrt(spherePtByRot.x*spherePtByRot.x + 
									 spherePtByRot.y*spherePtByRot.y), spherePtByRot.z);
		if (phiByRot * 2 > cam->fov) continue;

		cv::Point2d imgPt, imgPtByRot;
		cam->mapS2I(spherePt, imgPt);
		cam->mapS2I(spherePtByRot, imgPtByRot);
		vSpherePt1.push_back(spherePt);
		vSpherePt2.push_back(spherePtByRot);
		vImgPt1.push_back(imgPt);
		vImgPt2.push_back(imgPtByRot);
		count++;
	}
}

void TestInversePolynomial()
{
	FishEye::PTGUIFish ptgui;
	double a, b, c;
	cv::Mat result(1000, 1000, CV_8UC3, cv::Scalar(0));
	ptgui.invertABCBySample(0.3, 0.1, 0.5, a, b, c);
	ptgui.drawWarpLine(0.3, 0.1, 0.5, result, cv::Scalar(0, 0, 255), false);
	ptgui.drawWarpLine(a, b, c, result, cv::Scalar(0, 255, 255), true);
	double a2, b2, c2;
	ptgui.invertABCBySample(a, b, c, a2, b2, c2);
	ptgui.drawWarpLine(a2, b2, c2, result, cv::Scalar(0, 255, 0), false);
	cv::imwrite("result.jpg", result);
}

void ProduceDataShow() 
{
	Rotation rotation(CV_PI * 0.4, CV_PI * 0.5);
	std::shared_ptr<FishEye::Equidistant> pEquisolid = std::make_shared<FishEye::Equidistant>();
	pEquisolid->fov = CV_PI*1.3;
	pEquisolid->fx = pEquisolid->fy = 400;
	pEquisolid->u0 = pEquisolid->fx * pEquisolid->fov * 0.5 + 50;
	pEquisolid->v0 = pEquisolid->fy * pEquisolid->fov * 0.5 + 50;
	int numPt = 1000;
	std::vector<cv::Point2d> vImgPt1, vImgPt2;
	std::vector<cv::Point3d> vSpherePt1, vSpherePt2;
	randomImagePointPairs(std::static_pointer_cast<CameraModel>(pEquisolid),
						  rotation, numPt, vImgPt1, vImgPt2, vSpherePt1, vSpherePt2);

	cv::Mat showResult1(pEquisolid->u0 * 2 + 2, pEquisolid->v0 * 2 + 2, CV_8UC3, cv::Scalar(250, 250, 250));
	cv::Mat showResult2(pEquisolid->u0 * 2 + 2, pEquisolid->v0 * 2 + 2, CV_8UC3, cv::Scalar(250, 250, 250));

	for (size_t i = 0; i < numPt; i++)
	{
		cv::Scalar randomColor = RandomColor();
		cv::circle(showResult1, vImgPt1[i], 4, randomColor, -1);
		cv::circle(showResult2, vImgPt2[i], 4, randomColor, -1);
	}

	cv::imwrite("showResult1.jpg", showResult1);
	cv::imwrite("showResult2.jpg", showResult2);

	std::fstream fs("spherePts.txt", std::ios::out);
	for (size_t i = 0; i < vSpherePt1.size(); i++)
	{
		fs << vSpherePt1[i].x << " ";
		fs << vSpherePt1[i].y << " ";
		fs << vSpherePt1[i].z << " ";
		fs << vSpherePt2[i].x << " ";
		fs << vSpherePt2[i].y << " ";
		fs << vSpherePt2[i].z << " ";
		fs << std::endl;
	}
	fs.close();
}

int main(int argc, char *argv[])
{
	
	int pairNum = 300, trialNum = 2;
	double minFocal = 400, maxFocal = 600;
	double minFov = CV_PI * (160 / 180.0), maxFov = CV_PI * (200 / 180.0);
	double minAngle = CV_PI * (70 / 180.0), maxAngle = CV_PI * (110 / 180.0);
	
	std::ofstream fs("SyntheticData.txt", std::ios::out);
	ModelDataProducer producer;
	for (size_t i = 0; i < trialNum; i++)
	{
		std::shared_ptr<Equidistant> pEquisolid = std::make_shared<Equidistant>();
		std::shared_ptr<Rotation> pRotation = std::make_shared<Rotation>(minAngle, maxAngle);


		pEquisolid->fov = RandomInRange(minFov, maxFov);
		pEquisolid->fx = pEquisolid->fy = RandomInRange(minFocal, maxFocal);
		producer.produce(std::static_pointer_cast<CameraModel>(pEquisolid), pRotation, pairNum);
		producer.writeToFile(fs);
	}
	fs.close();
	return 0;
}