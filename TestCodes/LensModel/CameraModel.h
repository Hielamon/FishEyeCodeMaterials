#pragma once
#include <OpencvCommon.h>

class CameraModel
{
public:
	CameraModel() {}
	~CameraModel() {}

	//mapping the image coordinate to the unit sphere coordinate	
	virtual bool mapI2S(const cv::Point2d &imgPt, cv::Point3d &spherePt)  = 0;

	//mapping the unit sphere coordinate to the image coordinate	
	virtual bool mapS2I(const cv::Point3d &spherePt, cv::Point2d &imgPt)  = 0;

	double fov;
	double u0, v0;
	double f;

};

class PTGUIFishEye : public CameraModel
{
public:
	PTGUIFishEye() {};
	~PTGUIFishEye() {};

	//mapping the image coordinate to the unit sphere coordinate	
	virtual bool mapI2S(const cv::Point2d &imgPt, cv::Point3d &spherePt) 
	{
		double x = imgPt.x - u0;
		double y = -imgPt.y + v0;
		double r_dist = sqrt(x*x + y*y);
		if (r_dist > radius)return false;

		double r_dist_n = r_dist / radius;
		double d = 1 - a - b - c;
		double r_udis_n = solvequartic(a, b, c, d, -r_dist_n);
		double theta = atan2(y, x);

		double phi = r_udis_n*fov*0.5;

		spherePt.x = sin(phi)*cos(theta);
		spherePt.y = sin(phi)*sin(theta);
		spherePt.z = cos(phi);

		return true;
	}

	//mapping the unit sphere coordinate to the image coordinate	
	virtual bool mapS2I(const cv::Point3d &spherePt, cv::Point2d &imgPt) 
	{
		double theta = atan2(spherePt.y, spherePt.x);
		double phi = atan2(sqrt(spherePt.x*spherePt.x + spherePt.y*spherePt.y), spherePt.z);
		if (phi * 2 > fov)return false;

		double r_udis_n = std::abs(phi * 2 / fov);
		double d = 1 - a - b - c;
		double r_dist_n = (((a*r_udis_n + b)*r_udis_n + c)*r_udis_n + d)*r_udis_n;
		double r_dist = radius*r_dist_n;

		imgPt.x = r_dist*cos(theta) + u0;
		imgPt.y = -r_dist*sin(theta) + v0;

		return true;
	}

	//use the least-squared algorithm to invert the ploynomial function,
	//the value region is [0,1]
	//polynoial function form: y = ax^4 + bx^3 + cx^2 + (1 - a - b - c)x
	void invertABCBySample(const double &_a, const double &_b, const double &_c, 
						   double &_aInv, double &_bInv, double &_cInv )
	{
		int numSamples = 1000;
		std::vector<double> vA(numSamples * 3);
		std::vector<double> vY(numSamples);
		for (size_t i = 0, ia = 0; i < numSamples; i++, ia += 3)
		{
			double x = i / double(numSamples - 1);
			double x2 = x*x, x3 = x2*x, x4 = x3*x;
			
			double y = _a*x4 + _b*x3 + _c*x2 + (1 - _a - _b - _c) * x;
			vY[i] = x - y;

			double y2 = y*y, y3 = y*y2, y4 = y3*y;
			vA[ia] = y4 - y;
			vA[ia + 1] = y3 - y;
			vA[ia + 2] = y2 - y;
		}

		cv::Mat ATA(3, 3, CV_64FC1, cv::Scalar(0));

		//cv::Mat A(numSamples, 4, CV_64FC1, &vA[0]);
		//cv::Mat ATACheck = A.t() * A;

		for (size_t i = 0; i < 3; i++)
		{
			double *pATARow = reinterpret_cast<double *>(ATA.ptr(i));
			for (size_t j = 0; j < 3; j++)
			{
				if (j < i)
				{
					pATARow[j] = ATA.at<double>(j, i);
				}
				else
				{
					size_t ibase = i;
					size_t jbase = j;
					for (size_t k = 0; k < numSamples; k++)
					{
						pATARow[j] += (vA[ibase] * vA[jbase]);
						ibase += 3;
						jbase += 3;
					}
				}
			}
			
		}

		
		cv::Mat A(numSamples, 3, CV_64FC1, &vA[0]);
		cv::Mat b(numSamples, 1, CV_64FC1, &vY[0]);


		cv::Mat result = ATA.inv() * A.t() * b;
		//std::cout << "result = " << result << std::endl;

		_aInv = result.at<double>(0, 0);
		_bInv = result.at<double>(1, 0);
		_cInv = result.at<double>(2, 0);

		
	}

	void drawWarpLine(const double &_a, const double &_b, const double &_c, cv::Mat &result, cv::Scalar &color, bool transpose = false)
	{
		int width = result.cols;
		for (size_t i = 0, ia = 0; i < width; i++, ia += 3)
		{
			double x = i / double(width - 1);
			double x2 = x*x, x3 = x2*x, x4 = x3*x;

			double y = _a*x4 + _b*x3 + _c*x2 + (1 - _a - _b - _c) * x;
			if (transpose)
			{
				cv::circle(result, cv::Point(y*(width - 1), width - 1 - i), 1, color, -1);
			}
			else
			{
				cv::circle(result, cv::Point(i, width - 1 - y*(width - 1)), 1, color, -1);
			}
		}
	}

	double radius;
	double a, b, c;
	double aInv, bInv, cInv;

private:
	//solve the quartic equation a*x^4+b*x^3+c*x^2+d*x+e=0
	//with the prior knowledge that the value is during 0~1
	double solvequartic(double a, double b, double c, double d, double e)
	{
		double eps = 1e-8;
		double x0 = -e;
		double y0 = (((a*x0 + b)*x0 + c)*x0 + d)*x0 + e;
		if (std::abs(y0) < eps)return x0;

		double k0 = ((4 * a*x0 + 3 * b)*x0 + 2 * c)*x0 + d;
		double init_step = y0*k0 < 0 ? 0.1 : -0.1;

		double x1 = x0 + init_step;
		double y1 = (((a*x1 + b)*x1 + c)*x1 + d)*x1 + e;

		int iter_num = 0;
		while (std::abs(y1) > eps && iter_num < 1000)
		{
			if (y1*y0 > 0)
			{
				x0 = x1;
				y0 = y1;
				x1 += init_step;
			}
			else
			{
				init_step *= 0.5;
				x1 = x0 + init_step;
			}

			y1 = (((a*x1 + b)*x1 + c)*x1 + d)*x1 + e;
			iter_num++;
		}

		return x1;
	}
};

class EquidistantFishEye : public CameraModel
{
public:
	EquidistantFishEye() {}
	~EquidistantFishEye() {}

	//mapping the image coordinate to the unit sphere coordinate	
	virtual bool mapI2S(const cv::Point2d &imgPt, cv::Point3d &spherePt) 
	{
		double x = imgPt.x - u0;
		double y = -imgPt.y + v0;
		double r_dist = sqrt(x*x + y*y);
		double theta = atan2(y, x);

		double phi = r_dist / f;
		if (phi * 2 > fov)return false;

		spherePt.x = sin(phi)*cos(theta);
		spherePt.y = sin(phi)*sin(theta);
		spherePt.z = cos(phi);
		return true;
	}

	//mapping the unit sphere coordinate to the image coordinate	
	virtual bool mapS2I(const cv::Point3d &spherePt, cv::Point2d &imgPt) 
	{
		double theta = atan2(spherePt.y, spherePt.x);
		double phi = atan2(sqrt(spherePt.x*spherePt.x + spherePt.y*spherePt.y), spherePt.z);
		if (phi * 2 > fov)return false;

		double r_dist = f * phi;

		imgPt.x = r_dist*cos(theta) + u0;
		imgPt.y = -r_dist*sin(theta) + v0;
		return true;
	}

};

class EquisolidFishEye : public CameraModel
{
public:
	EquisolidFishEye() {}
	~EquisolidFishEye() {}

	//mapping the image coordinate to the unit sphere coordinate	
	virtual bool mapI2S(const cv::Point2d &imgPt, cv::Point3d &spherePt) 
	{
		double x = imgPt.x - u0;
		double y = -imgPt.y + v0;
		double r_dist = sqrt(x*x + y*y);
		double theta = atan2(y, x);

		double phi = 2 * asin(r_dist / (2 * f));
		if (phi * 2 > fov)return false;

		spherePt.x = sin(phi)*cos(theta);
		spherePt.y = sin(phi)*sin(theta);
		spherePt.z = cos(phi);
		return true;
	}

	//mapping the unit sphere coordinate to the image coordinate	
	virtual bool mapS2I(const cv::Point3d &spherePt, cv::Point2d &imgPt) 
	{
		double theta = atan2(spherePt.y, spherePt.x);
		double phi = atan2(sqrt(spherePt.x*spherePt.x + spherePt.y*spherePt.y), spherePt.z);
		if (phi * 2 > fov)return false;

		double r_dist = 2 * f * sin(phi * 0.5);

		imgPt.x = r_dist*cos(theta) + u0;
		imgPt.y = -r_dist*sin(theta) + v0;
		return true;
	}


};

class StereographicFishEye : public CameraModel
{
public:
	StereographicFishEye() {}
	~StereographicFishEye() {}

	//mapping the image coordinate to the unit sphere coordinate	
	virtual bool mapI2S(const cv::Point2d &imgPt, cv::Point3d &spherePt) 
	{
		double x = imgPt.x - u0;
		double y = -imgPt.y + v0;
		double r_dist = sqrt(x*x + y*y);
		double theta = atan2(y, x);

		double phi = 2 * atan(r_dist / (2 * f));
		if (phi * 2 > fov)return false;

		spherePt.x = sin(phi)*cos(theta);
		spherePt.y = sin(phi)*sin(theta);
		spherePt.z = cos(phi);
		return true;
	}

	//mapping the unit sphere coordinate to the image coordinate	
	virtual bool mapS2I(const cv::Point3d &spherePt, cv::Point2d &imgPt) 
	{
		double theta = atan2(spherePt.y, spherePt.x);
		double phi = atan2(sqrt(spherePt.x*spherePt.x + spherePt.y*spherePt.y), spherePt.z);
		if (phi * 2 > fov)return false;

		double r_dist = 2 * f * tan(phi * 0.5);

		imgPt.x = r_dist*cos(theta) + u0;
		imgPt.y = -r_dist*sin(theta) + v0;
		return true;
	}
};


