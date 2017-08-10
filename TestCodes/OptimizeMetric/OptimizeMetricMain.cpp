#include <OpencvCommon.h>
#include "precomp.hpp"
#include <random>

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

int main(int argc, char *argv[])
{
	TestSinRefineCallback();

	/*std::string filename = "test.py";
	std::string command = "python ";
	command += filename;
	system(command.c_str());*/
	return 0;
}