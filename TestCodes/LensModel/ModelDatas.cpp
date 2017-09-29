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

int main(int argc, char *argv[])
{

	
	int pairNum = 300, trialNum = 500;
	double sigma = 0.0, translateLen = 0.0;
	
	std::string classicModelName[3] = { "Equidistant", "Equisolid", "Stereographic" };
	double minFocal = 400, maxFocal = 600;
	double minFov = CV_PI * (160 / 180.0), maxFov = CV_PI * (200 / 180.0);
	double minAngle = CV_PI * (70 / 180.0), maxAngle = CV_PI * (110 / 180.0);
	
	std::ofstream fs("SyntheticData.txt", std::ios::out);
	fs << trialNum << std::endl;
	ModelDataProducer producer;
	for (size_t i = 0; i < trialNum; i++)
	{
		double fov = RandomInRange(minFov, maxFov);
		double f = RandomInRange(minFocal, maxFocal);
		int typeIdx = RandomInRange(0, 3);
		
		std::shared_ptr<CameraModel> pModel = createCameraModel(classicModelName[typeIdx], 0, 0, f, fov, 0);/*std::make_shared<Stereographic>(0, 0, f, fov)*/;
		std::shared_ptr<Rotation> pRotation = std::make_shared<Rotation>(minAngle, maxAngle);
		producer.produce(pModel, pRotation, pairNum);
		producer.writeToFile(fs);
	}
	fs.close();
	return 0;
}