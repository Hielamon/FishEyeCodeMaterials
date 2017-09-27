#pragma once
#include <OpencvCommon.h>

class CameraModel
{
public:
	CameraModel() 
	{
		//fov = CV_PI * 0.5;
		u0 = v0 = 0;
		fx = fy = 1;
		//vpParameter.push_back(&fov);
		vpParameter.push_back(&u0);
		vpParameter.push_back(&v0);
		vpParameter.push_back(&fx);
		vpParameter.push_back(&fy);
	}
	~CameraModel() {}

	//mapping the image coordinate to the unit sphere coordinate	
	virtual bool mapI2S(const cv::Point2d &imgPt, cv::Point3d &spherePt)
	{
		double x = (imgPt.x - u0) / fx;
		double y = (-imgPt.y + v0) / fy;
		double r_dist = sqrt(x*x + y*y);

		double theta, phi;
		theta = atan2(y, x);

		if (!inverseProject(r_dist, phi))
		{
			std::cout << "Warning: Invalid mapping in mapI2S" << std::endl;
			return false;
		}

		spherePt.x = sin(phi)*cos(theta);
		spherePt.y = sin(phi)*sin(theta);
		spherePt.z = cos(phi);
		return true;
	}

	//mapping the unit sphere coordinate to the image coordinate	
	virtual bool mapS2I(const cv::Point3d &spherePt, cv::Point2d &imgPt)
	{
		double theta, phi, r_dist;
		theta = atan2(spherePt.y, spherePt.x);
		phi = atan2(sqrt(spherePt.x*spherePt.x + spherePt.y*spherePt.y), spherePt.z);
		if (/*phi * 2 > fov || */!project(phi, r_dist))
		{
			std::cout << "Warning: Invalid mapping in mapS2I" << std::endl;
			return false;
		}

		imgPt.x = r_dist*cos(theta)*fx + u0;
		imgPt.y = -r_dist*sin(theta)*fy + v0;
		return true;
	}

	//projecting the imaging radius to the incident angle
	virtual bool inverseProject(const double& radius, double &angle)
	{
		angle = atan(radius);
		if (angle < 0 /*|| angle > fov*0.5 */|| angle >= CV_PI*0.5)
		{
			return false;
		}
		
		return true;
	}

	//projecting the incident angle to the imaging radius
	virtual bool project(const double& angle, double &radius)
	{
		if (angle < 0 /*|| angle > fov*0.5 */|| angle >= CV_PI*0.5)
		{
			return false;
		}

		radius = tan(angle);
		return true;
	}

	virtual std::string getTypeName()
	{
		return "Default";
	}

	//double fov;
	double u0, v0;
	double fx, fy;
	//double radius;

	//not used actually, when we limit that the fx = fy, the changing of f by setP
	//double f;

	//The array of parameters point, which is arrange as
	//vParameter[0:5] = {&fov, &u0, &v0, &fx, &fy}
	//which can be extended in the derive class
	std::vector<double*> vpParameter;
};

namespace FishEye
{
	//The solver for quadratic equation with one unknown
	//a*x^2 + b*x + c = 0
	inline std::vector<double> solverUnitaryQuadratic(const double &a, const double &b, const double &c)
	{
		std::vector<double> result;
		result.reserve(2);
		if (a == 0)
		{
			if (b != 0)
			{
				result.push_back(-c / b);
			}
		}
		else
		{
			double delta = b*b - 4 * a*c;
			if (delta >= 0)
			{
				if (delta > 0)
				{
					double sqrtDelta = sqrt(delta);
					result.push_back((-b + sqrtDelta) / (2 * a));
					result.push_back((-b - sqrtDelta) / (2 * a));
				}
				else
				{
					result.push_back(-b * 0.5 / a);
				}
			}
		}
		return result;
	}

	//this class is actually not be used in our experiment
	/*(class PTGUIFish : public CameraModel
	{
	public:
		PTGUIFish() {};
		~PTGUIFish() {};

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
							   double &_aInv, double &_bInv, double &_cInv)
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
	};*/

	class Equidistant : public CameraModel
	{
	public:
		Equidistant() {}
		~Equidistant() {}

		//projecting the imaging radius to the incident angle
		virtual bool inverseProject(const double& radius, double &angle)
		{
			angle = radius;
			if (angle < 0 /*|| angle > fov*0.5 */|| angle > CV_PI)
			{
				return false;
			}

			return true;
		}

		//projecting the incident angle to the imaging radius
		virtual bool project(const double& angle, double &radius)
		{
			if (angle < 0 /*|| angle > fov*0.5*/ || angle > CV_PI)
			{
				return false;
			}

			radius = angle;
			return true;
		}

		virtual std::string getTypeName()
		{
			return "Equidistant";
		}
	};

	class Equisolid : public CameraModel
	{
	public:
		Equisolid() {}
		~Equisolid() {}

		//projecting the imaging radius to the incident angle
		virtual bool inverseProject(const double& radius, double &angle)
		{
			angle = asin(radius * 0.5) * 2;
			if (angle < 0 /*|| angle > fov*0.5 */|| angle > CV_PI)
			{
				return false;
			}

			return true;
		}

		//projecting the incident angle to the imaging radius
		virtual bool project(const double& angle, double &radius)
		{
			if (angle < 0 /*|| angle > fov*0.5 */|| angle > CV_PI)
			{
				return false;
			}

			radius = 2 * sin(angle*0.5);
			return true;
		}

		virtual std::string getTypeName()
		{
			return "Equisolid";
		}
	};

	class Stereographic : public CameraModel
	{
	public:
		Stereographic() {}
		~Stereographic() {}

		//projecting the imaging radius to the incident angle
		virtual bool inverseProject(const double& radius, double &angle)
		{
			angle = atan(radius * 0.5) * 2;
			if (angle < 0 /*|| angle > fov*0.5 */|| angle > CV_PI)
			{
				return false;
			}

			return true;
		}

		//projecting the incident angle to the imaging radius
		virtual bool project(const double& angle, double &radius)
		{
			if (angle < 0 /*|| angle > fov*0.5 */|| angle > CV_PI)
			{
				return false;
			}

			radius = 2 * tan(angle*0.5);
			return true;
		}

		virtual std::string getTypeName()
		{
			return "Stereographic";
		}
	};

	//Refer to : A Generic Camera Model and Calibration Method for Conventional, Wide-Angle, and Fish-Eye Lenses
	class PolynomialAngle : public CameraModel
	{
	public:
		PolynomialAngle() 
		{
			//add extra parameters reference
			k1 = 1;
			k2 = 0;
			vpParameter.push_back(&k1);
			vpParameter.push_back(&k2);
		}
		~PolynomialAngle() {}

		//projecting the imaging radius to the incident angle
		//radius = k1 * (angle) + k2 * (angle)^3
		//According to https://zh.wikipedia.org/wiki/%E4%B8%89%E6%AC%A1%E6%96%B9%E7%A8%8B
		virtual bool inverseProject(const double& radius, double &angle)
		{
			double a = k2, c = k1, d = -radius;
			if (a == 0)
			{
				if (c == 0)
				{
					return false;
				}

				angle = -d / c;
				//return true;
			}
			else
			{
				double p = c / a, q = d / a;
				double p3 = p*p*p, q2 = q*q;
				double delta = q2 + 4 * p3 / 27;
				if (delta > 0)
				{
					double u3 = (-q + sqrt(delta))*0.5;
					double v3 = (-q - sqrt(delta))*0.5;
					angle = pow(u3, 1.0 / 3) + pow(v3, 1.0 / 3);
				}
				else if (delta == 0)
				{
					if (p == 0)
					{
						angle = 0;
					}
					else
					{
						angle = pow(q*0.2, 1.0 / 3);
						angle = angle > 0 ? angle : -angle;
					}
				}
				else
				{
					double Q = -p / 3.0, R = -q * 0.5;
					double sqrtQ = sqrt(Q);
					double theta = acos(R / (Q * sqrtQ));
					//I think that the only positive one is x1
					angle = 2 * sqrtQ*cos(theta / 3.0);
				}
			}
			
			if (angle < 0 /*|| angle > fov*0.5*/ || angle > CV_PI)
			{
				return false;
			}

			return true;
		}

		//projecting the incident angle to the imaging radius
		//radius = k1 * (angle) + k2 * (angle)^3
		virtual bool project(const double& angle, double &radius)
		{
			if (angle < 0 /*|| angle > fov*0.5 */|| angle > CV_PI)
			{
				return false;
			}

			radius = k1 * angle + k2 * angle * angle * angle;

			return radius >= 0;
		}

		virtual std::string getTypeName()
		{
			return "PolynomialAngle";
		}

		double k1, k2;
	};

	//Refer to : A Toolbox for Easily Calibrating Omnidirectional
	class PolynomialRadius : public CameraModel
	{
	public:
		PolynomialRadius() 
		{
			a0 = 1;
			a2 = 0;
			vpParameter.push_back(&a0);
			vpParameter.push_back(&a2);
		}
		~PolynomialRadius() {}

		//projecting the imaging radius to the incident angle
		//rd / (a0 + a2*rd^2) = sin(theta) / cos(theta)
		virtual bool inverseProject(const double& radius, double &angle)
		{
			angle = atan2(radius, a0 + a2*radius*radius);

			if (angle < 0 /*|| angle > fov*0.5*/ || angle > CV_PI)
			{
				return false;
			}

			return true;
		}

		//projecting the incident angle to the imaging radius
		//rd / (a0 + a2*rd^2) = sin(theta) / cos(theta)
		//rd^2*a2*sin(theta) - rd*cos(theta) + a0*sin(theta) = 0
		virtual bool project(const double& angle, double &radius)
		{
			if (angle < 0 /*|| angle > fov*0.5*/ || angle > CV_PI)
			{
				return false;
			}

			double a = a2 * sin(angle), b = -cos(angle), c = a0 * sin(angle);
			std::vector<double> root = solverUnitaryQuadratic(a, b, c);
			radius = -1;
			bool unique = false;

			for (size_t i = 0; i < root.size(); i++)
			{
				if (root[i] >= 0)
				{
					radius = root[i];
					unique = !unique;
				}
			}
			
			return unique;
		}

		virtual std::string getTypeName()
		{
			return "PolynomialRadius";
		}

		double a0, a2;
	};

	//Refer to : A unifying theory for central panoramic systems and practical implications
	class GeyerModel : public CameraModel
	{
	public:
		GeyerModel() 
		{
			l = 0;
			m = 1;
			vpParameter.push_back(&l);
			vpParameter.push_back(&m);
		}
		~GeyerModel() {}

		//projecting the imaging radius to the incident angle
		//rd = (m + l)*sin(theta) / ( l + cos(theta))
		//d + e*cos(theta) = sin(theta)
		virtual bool inverseProject(const double& radius, double &angle)
		{
			double d = radius * l / (m + l);
			double e = radius / (m + l);

			//d^2 - 1 + 2*d*e*cos(theta) + (e^2 + 1)*cos(theta)^2 = 0
			double a = e*e + 1;
			double b = 2 * d * e;
			double c = d*d - 1;
			double cosValue = 0;

			std::vector<double> root = solverUnitaryQuadratic(a, b, c);
			int rootNum = root.size();
			angle = -1;
			bool unique = false;

			for (size_t i = 0; i < root.size(); i++)
			{
				double cosValue = root[i];
				if (abs(cosValue) <= 1)
				{
					double tmpAngle = acos(cosValue);
					double left = d + e * cosValue;
					double right = sin(tmpAngle);
					if (left * right >= 0 && tmpAngle >= 0
						/*&& tmpAngle <= fov*0.5 */&&tmpAngle < CV_PI)
					{
						angle = tmpAngle;
						unique = !unique;
					}
				}
			}


			return unique;
		}

		//projecting the incident angle to the imaging radius
		//rd = (m + l)*sin(theta) / ( l + cos(theta))
		virtual bool project(const double& angle, double &radius)
		{
			if (angle < 0 /*|| angle > fov*0.5*/ || angle > CV_PI)
			{
				return false;
			}

			radius = (m + l) * sin(angle) / (l + cos(angle));
			
			return radius >= 0;
		}

		virtual std::string getTypeName()
		{
			return "GeyerModel";
		}

		double l, m;
	};

	
}

inline std::shared_ptr<CameraModel> createCameraModel(const std::string &typeName)
{
	std::shared_ptr<CameraModel> result = std::make_shared<CameraModel>();
	if (typeName == "Equidistant")
	{
		result = std::static_pointer_cast<CameraModel>(std::make_shared<FishEye::Equidistant>());
	}
	else if (typeName == "Equisolid")
	{
		result = std::static_pointer_cast<CameraModel>(std::make_shared<FishEye::Equisolid>());
	}
	else if (typeName == "Stereographic")
	{
		result = std::static_pointer_cast<CameraModel>(std::make_shared<FishEye::Stereographic>());
	}
	else if (typeName == "PolynomialAngle")
	{
		result = std::static_pointer_cast<CameraModel>(std::make_shared<FishEye::PolynomialAngle>());
	}
	else if (typeName == "PolynomialRadius")
	{
		result = std::static_pointer_cast<CameraModel>(std::make_shared<FishEye::PolynomialRadius>());
	}
	else if (typeName == "GeyerModel")
	{
		result = std::static_pointer_cast<CameraModel>(std::make_shared<FishEye::GeyerModel>());
	}

	return result;
}




