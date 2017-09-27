#include <OpencvCommon.h>
#include "precomp.hpp"
#include <random>
#include "../LensModel/ModelData.h"

using namespace cv;

class SinRefineCallback : public LMSolver::Callback
{
public:

	SinRefineCallback(InputArray _src, InputArray _dst) 
	{
		src = _src.getMat();
		dst = _dst.getMat();
	}

	bool compute(InputArray _param, OutputArray _err, OutputArray _Jac) const
	{
		int dimPt = 1;
		int i, count = src.checkVector(dimPt);
		Mat param = _param.getMat();
		_err.create(count, 1, CV_64F);
		Mat err = _err.getMat(), J;
		if (_Jac.needed())
		{
			_Jac.create(count * dimPt, param.rows, CV_64F);
			J = _Jac.getMat();
			CV_Assert(J.isContinuous() && J.cols == 1);
		}

		const float* M = src.ptr<float>();
		const float* m = dst.ptr<float>();
		const double* a = param.ptr<double>();
		double* errptr = err.ptr<double>();
		double* Jptr = J.data ? J.ptr<double>() : 0;

		for (i = 0; i < count; i++)
		{
			float x = M[i], y = m[i];
			float yi = std::sin(a[0] * x);
			errptr[i] = yi - y;

			if (Jptr)
			{
				Jptr[0] = x * std::cos(a[0] * x);
				Jptr += 1;
			}
		}

		return true;
	}

	Mat src, dst;
};

void TestSinRefineCallback()
{
	int numPt = 400;
	cv::Mat xMat(numPt, 1, CV_32FC1);
	cv::Mat yMat(numPt, 1, CV_32FC1);
	float *xPtr = xMat.ptr<float>();
	float *yPtr = yMat.ptr<float>();
	float a = 4;
	std::normal_distribution<float> distribution(0, 0.00001);
	for (size_t i = 0; i < numPt; i++)
	{
		xPtr[i] = (i / float(numPt - 1)) * 4;
		yPtr[i] = std::sin(a * xPtr[i]) + distribution(std::default_random_engine());
	}
	float startA = -10, endA = 10;
	int numSplit = 1000;
	cv::Mat showCurve(numSplit, numSplit, CV_8UC3, cv::Scalar(0));
	std::vector<float> vError(400);
	std::fstream fs("sinRefineData.txt", std::ios::in);
	for (size_t i = 0; i < numSplit; i++)
	{
		float tmpA = startA + (i / float(numSplit - 1)) * (endA - startA);
		float error = 0;
		for (size_t j = 0; j < numPt; j++)
		{
			float dy = std::sin(tmpA * xPtr[j]) - yPtr[j];
			error += dy * dy;
		}
		cv::circle(showCurve, cv::Point(i, numSplit - 1 - error), 1, cv::Scalar(0, 255, 0));
		fs << i << " " << error << std::endl;
	}
	fs.close();

	cv::Mat param(1, 1, CV_64FC1);
	param.at<double>(0, 0) = 2;

	Ptr<LMSolver> levmarpPtr = createLMSolver(makePtr<SinRefineCallback>(xMat, yMat), 20);
	int iter = levmarpPtr->run(param);
	std::cout << "result is : a = " << param.at<double>(0, 0) << "; ground truth : a = " << a << std::endl;
	std::cout << "Optimization iteration is " << iter << " times" << std::endl;
	a = param.at<double>(0, 0);
	float error = 0;
	for (size_t j = 0; j < numPt; j++)
	{
		float dy = std::sin(a * xPtr[j]) - yPtr[j];
		error += dy * dy;
	}
	std::cout << "Optimized error is " << error << std::endl;
	cv::circle(showCurve, cv::Point(((a - startA) / (endA - startA)) * numSplit, numSplit - 1 - error), 5, cv::Scalar(0, 255, 255));
	cv::imshow("showCurve", showCurve);
	cv::waitKey(0);
	cv::imwrite("SinRefineCurve.jpg", showCurve);
}

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

	double error = 0;
	for (size_t i = 0; i < pModelData->mcount; i++)
	{
		cv::Point2d &imgPt1 = pModelData->mvImgPt1[i], &imgPt2 = pModelData->mvImgPt2[i];
		cv::Point3d spherePt1, spherePt2;
		bool valid = true;
		valid = valid && pModel->mapI2S(imgPt1, spherePt1);
		valid = valid && pModel->mapI2S(imgPt2, spherePt2);
		cv::Point3d spherePt1ByRot = RotatePoint(spherePt1, *(pRot));
		error += (spherePt2.x - spherePt1ByRot.x)*(spherePt2.x - spherePt1ByRot.x);
		error += (spherePt2.y - spherePt1ByRot.y)*(spherePt2.y - spherePt1ByRot.y);
		error += (spherePt2.z - spherePt1ByRot.z)*(spherePt2.z - spherePt1ByRot.z);
	}

	error = std::sqrt(error);
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

		if (_Jac.needed())
		{
			_Jac.create(pairNum * 3, mvpParameter.size(), CV_64F);
			cv::Mat J = _Jac.getMat();
			_calcJacobian(J);
		}

		std::cout << "average error = " << norm(err) << std::endl;
		std::cout << param.at<double>(0, 0) << " " << param.at<double>(1, 0) << std::endl;
		std::cout << param.at<double>(2, 0) << " " << param.at<double>(3, 0) << std::endl;
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

		const double step = 1e-5;
		cv::Mat err1, err2;
		err1.create(pairNum * 3, 1, CV_64F);
		err2.create(pairNum * 3, 1, CV_64F);

		for (size_t i = 0; i < mvpParameter.size(); i++)
		{
			double originValue = *(mvpParameter[i]);

			*(mvpParameter[i]) = originValue - step;
			if (mvRotMask[i])
			{
				mpRot->updataRotation(mpRot->axisAngle);
			}

			_calcError(err1);

			*(mvpParameter[i]) = originValue + step;
			if (mvRotMask[i])
			{
				mpRot->updataRotation(mpRot->axisAngle);
			}

			_calcError(err2);

			_calcDeriv(err1, err2, 2 * step, jac.col(i));

			*(mvpParameter[i]) = originValue;
			if (mvRotMask[i])
			{
				mpRot->updataRotation(mpRot->axisAngle);
			}
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
	std::ifstream fs("../LensModel/SyntheticData.txt", std::ios::in);
	int trialNum;
	fs >> trialNum;
	for (size_t i = 0; i < trialNum; i++)
	{
		std::shared_ptr<ModelDataProducer> pModelData = std::make_shared<ModelDataProducer>();
		pModelData->readFromFile(fs);
		
		std::shared_ptr<FishEye::GeyerModel> pGeyer =  std::make_shared<FishEye::GeyerModel>();
		cv::Point3d edge3DPt(1, 0, 0);
		cv::Point2d edge2DPt;
		pModelData->mpCam->mapS2I(edge3DPt, edge2DPt);
		double circleRadius = std::abs(edge2DPt.x);
		double radio = circleRadius / (CV_PI * 0.5);
		pGeyer->fx = pGeyer->fy = radio;
		//pGeyer->fov = CV_PI * 1.5;
		pGeyer->l = 1.743803;
		pGeyer->m = 0.976517;
		std::shared_ptr<CameraModel> pModel = std::static_pointer_cast<CameraModel>(pGeyer);

		std::shared_ptr<Rotation> pRot = std::make_shared<Rotation>(CV_PI*0.5, CV_PI*0.5);
		CalculateRotation(pModelData, pModel, pRot);
		std::vector<uchar> vMask(pGeyer->vpParameter.size(), 1);
		vMask[0] = vMask[1] = 0;

		Ptr<LMSolver> levmarpPtr = createLMSolver(makePtr<FishModelRefineCallback>(pModelData, pModel, pRot, vMask), 20);

		//get the initial parameters in param;
		//we use fov = 180бу and known circle radius to init the focol length and relative parameters
		cv::Mat param(7, 1, CV_64FC1);
		{
			param.at<double>(0, 0) = pGeyer->fx;
			param.at<double>(1, 0) = pGeyer->fy;
			param.at<double>(2, 0) = pGeyer->l;
			param.at<double>(3, 0) = pGeyer->m;
			param.at<double>(4, 0) = pRot->axisAngle[0];
			param.at<double>(5, 0) = pRot->axisAngle[1];
			param.at<double>(6, 0) = pRot->axisAngle[2];
		}
		
		levmarpPtr->run(param);
	}
	return 0;
}