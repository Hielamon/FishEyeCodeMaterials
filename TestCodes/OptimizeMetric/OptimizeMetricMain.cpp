#include <OpencvCommon.h>
#include "precomp.hpp"
#include <random>
#include <map>
#include "../LensModel/ModelData.h"

using namespace cv;
using namespace FishEye;

void CalculateRotation(const std::shared_ptr<ModelDataProducer> &pModelData,
					   const std::shared_ptr<CameraModel> &pModel,
					   const std::shared_ptr<Rotation> &pRot)
{
	assert(pModelData.use_count() != 0 && pModel.use_count() != 0 && pRot.use_count() != 0);

	double s[9] = { 0 };
	cv::Mat S(3, 3, CV_64FC1, s);
	for (size_t i = 0; i < pModelData->mcount; i++)
	{
		cv::Point2d &imgPt1 = pModelData->mvImgPt1[i], &imgPt2 = pModelData->mvImgPt2[i];
		cv::Point3d spherePt1, spherePt2;
		pModel->mapI2S(imgPt1, spherePt1);
		pModel->mapI2S(imgPt2, spherePt2);

		s[0] += (spherePt1.x * spherePt2.x);
		s[1] += (spherePt1.x * spherePt2.y);
		s[2] += (spherePt1.x * spherePt2.z);
		s[3] += (spherePt1.y * spherePt2.x);
		s[4] += (spherePt1.y * spherePt2.y);
		s[5] += (spherePt1.y * spherePt2.z);
		s[6] += (spherePt1.z * spherePt2.x);
		s[7] += (spherePt1.z * spherePt2.y);
		s[8] += (spherePt1.z * spherePt2.z);
	}

	cv::Mat w, u, vt;
	cv::SVD::compute(S, w, u, vt);
	cv::Mat I = cv::Mat::eye(3, 3, CV_64FC1);
	I.at<double>(2, 2) = cv::determinant(vt.t() * u.t());
	cv::Mat R = vt.t() * I * u.t();
	pRot->updataRotation(R);
}


class FishModelRefineCallback : public LMSolver::Callback
{
public:
	FishModelRefineCallback(const std::shared_ptr<ModelDataProducer> &pModelData,
							const std::shared_ptr<CameraModel> &pModel,
							const std::shared_ptr<Rotation> &pRot,
							const std::vector<uchar> &vMask)
	{
		assert(pModel.use_count() != 0 && pModel.use_count() != 0 && pRot.use_count() != 0);
		mpModelData = pModelData;
		mpModel = pModel;
		mpRot = pRot;
		
		assert(vMask.size() == pModel->vpParameter.size());

		for (size_t i = 0; i < vMask.size(); i++)
		{
			if (vMask[i] != 0)
			{
				mvpParameter.push_back(pModel->vpParameter[i]);
				mvRotMask.push_back(false);
			}
		}

		for (size_t i = 0; i < 3; i++)
		{
			mvpParameter.push_back(&(mpRot->axisAngle[i]));
			mvRotMask.push_back(true);
		}
	}

	bool compute(InputArray _param, OutputArray _err, OutputArray _Jac) const
	{
		Mat param = _param.getMat();
		
		for (size_t i = 0; i < mvpParameter.size(); i++)
		{
			*(mvpParameter[i]) = param.at<double>(i, 0);
		}

		int pairNum = mpModelData->mcount;
		//err.create(pairNum * 3, 1, CV_64F);
		
		_err.create(pairNum * 3, 1, CV_64F);
		cv::Mat err = _err.getMat();
		_calcError(err);

		cv::Mat err2 = _err.getMat();

		if (_Jac.needed())
		{
			_Jac.create(pairNum * 3, mvpParameter.size(), CV_64F);
			cv::Mat J = _Jac.getMat();
			_calcJacobian(J);
		}

		/*std::cout << "average error = " << norm(err) << std::endl;
		std::cout << param.at<double>(0, 0) << std::endl;
		std::cout << param.at<double>(1, 0) << " " << param.at<double>(2, 0) << std::endl;
		std::cout << param.at<double>(3, 0) << " " << param.at<double>(4, 0) << " " << param.at<double>(5, 0) << std::endl;*/
		return true;
	}

private:
	void _calcDeriv(const cv::Mat &err1, const cv::Mat &err2, double h, cv::Mat &res) const
	{
		for (int i = 0; i < err1.rows; ++i)
			res.at<double>(i, 0) = (err2.at<double>(i, 0) - err1.at<double>(i, 0)) / h;
	}

	void _calcError(cv::Mat &err) const
	{
		int pairNum = mpModelData->mcount;
		//err.create(pairNum * 3, 1, CV_64F);

		for (size_t i = 0, idx = 0; i < pairNum; i++, idx += 3)
		{
			cv::Point2d &imgPt1 = mpModelData->mvImgPt1[i], &imgPt2 = mpModelData->mvImgPt2[i];
			//cv::Point3d &spherePt1 = mpModelData->mvSpherePt1[i], &spherePt2 = mpModelData->mvSpherePt2[i];
			cv::Point3d spherePt1, spherePt2;
			mpModel->mapI2S(imgPt1, spherePt1);
			mpModel->mapI2S(imgPt2, spherePt2);

			cv::Point3d spherePt1ByRot = RotatePoint(spherePt1, *(mpRot));
			err.at<double>(idx, 0) = spherePt1ByRot.x - spherePt2.x;
			err.at<double>(idx + 1, 0) = spherePt1ByRot.y - spherePt2.y;
			err.at<double>(idx + 2, 0) = spherePt1ByRot.z - spherePt2.z;
		}
	}

	void _calcJacobian(cv::Mat &jac) const
	{
		int pairNum = mpModelData->mcount;

		//jac.create(pairNum * 3, activeParamNum, CV_64F);
		jac.setTo(0);

		const double step = 1e-6;
		cv::Mat err1, err2;
		err1.create(pairNum * 3, 1, CV_64F);
		err2.create(pairNum * 3, 1, CV_64F);

		for (size_t i = 0; i < mvpParameter.size(); i++)
		{
			double originValue = *(mvpParameter[i]);

			*(mvpParameter[i]) = originValue - step;
			_updateParameters(i);

			_calcError(err1);

			*(mvpParameter[i]) = originValue + step;
			_updateParameters(i);

			_calcError(err2);

			_calcDeriv(err1, err2, 2 * step, jac.col(i));

			*(mvpParameter[i]) = originValue;
			_updateParameters(i);
		}
	}

	void _updateParameters(const int& idx) const
	{
		if (mvRotMask[idx])
		{
			mpRot->updataRotation(mpRot->axisAngle);
		}
		else
		{
			mpModel->updateFov();
		}
	}

	std::shared_ptr<ModelDataProducer> mpModelData;
	std::shared_ptr<CameraModel> mpModel;
	std::shared_ptr<Rotation> mpRot;
	std::vector<double *> mvpParameter;
	std::vector<bool> mvRotMask;

};

int main(int argc, char *argv[])
{
	std::shared_ptr<Equidistant> baseModel = std::make_shared<Equidistant>(0, 0, 1, CV_PI);
	std::map<std::string, cv::Vec2d> generalModelInfo;
	generalModelInfo["PolynomialAngle"] = cv::Vec2d(1.000000, 0.000000);
	generalModelInfo["PolynomialRadius"] = cv::Vec2d(1.038552, -0.407288);
	generalModelInfo["GeyerModel"] = cv::Vec2d(0.976517, 1.743803);

	std::ifstream fs("../LensModel/SyntheticData.txt", std::ios::in);
	int trialNum;
	fs >> trialNum;
	for (size_t i = 0; i < trialNum; i++)
	{
		std::cout << "\n\n\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << std::endl;
		std::cout << "data trial No. = " << i << std::endl;

		std::shared_ptr<ModelDataProducer> pModelData = std::make_shared<ModelDataProducer>();
		std::string typeName = pModelData->readFromFile(fs);
		std::cout << "The prodecer model Info: " << typeName << std::endl;
		std::cout << "fov : " << pModelData->mpCam->fov << "  f = " << pModelData->mpCam->f << std::endl;
		std::cout << "rotateAngle : " << norm(pModelData->mpRot->axisAngle);
		std::cout << "  rotateAxis : " << normalize(pModelData->mpRot->axisAngle) << std::endl;

		double maxRadius = pModelData->mpCam->maxRadius;
		double f = maxRadius / baseModel->maxRadius;

		std::map<std::string, cv::Vec2d>::iterator iter = generalModelInfo.begin();
		for (; iter != generalModelInfo.end(); iter++)
		{
			//std::cout << "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv" << std::endl;
			//std::cout << "Model -------> " << iter->first << std::endl;
			std::shared_ptr<CameraModel> pModel = std::static_pointer_cast<CameraModel>(
				createCameraModel(iter->first, 0, 0, f, 0, maxRadius, iter->second[0], iter->second[1]));

			std::shared_ptr<Rotation> pRot = std::make_shared<Rotation>(CV_PI*0.5, CV_PI*0.5);
			CalculateRotation(pModelData, pModel, pRot);
			std::vector<uchar> vMask(pModel->vpParameter.size(), 1);
			vMask[0] = vMask[1] = 0;

			std::string logFileName = iter->first + "_error.txt";
			//std::string logFileName;
			Ptr<FishModelRefineCallback> cb = makePtr<FishModelRefineCallback>(pModelData, pModel, pRot, vMask);
			Ptr<LMSolver> levmarpPtr = customCreateLMSolver(cb,	200, FLT_EPSILON, FLT_EPSILON, logFileName);

			//get the initial parameters in param;
			//we use fov = 180бу and known circle radius to init the focol length and relative parameters
			cv::Mat param(6, 1, CV_64FC1);
			{
				param.at<double>(0, 0) = *(pModel->vpParameter[2]);
				param.at<double>(1, 0) = *(pModel->vpParameter[3]);
				param.at<double>(2, 0) = *(pModel->vpParameter[4]);
				param.at<double>(3, 0) = pRot->axisAngle[0];
				param.at<double>(4, 0) = pRot->axisAngle[1];
				param.at<double>(5, 0) = pRot->axisAngle[2];
			}

			levmarpPtr->run(param);
		}

		system("py -3 ../DrawErrorCurve/DrawErrorCurve.py");
	}
	return 0;
}