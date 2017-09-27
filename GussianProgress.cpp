#include "GaussianProgress.h"



GaussianProgress::GaussianProgress():sigma(0.5),
	error(0.5)
{

}


GaussianProgress::~GaussianProgress()
{
}


cv::Mat GaussianProgress::CalculateW(const Mat& label, const Mat f)
{
	std::size_t num = label.cols;
	cv::Mat temp(1, (int)num, CV_32FC1);
	auto pdata = temp.ptr<float>(0);
	auto pdata1 = f.ptr<float>(0);
	for (std::size_t i = 0; i < num; ++i)
	{
		float pi = 1.0f / (1.0f + std::exp( -pdata1[i]));
		pdata[i] = pi*(1 - pi);
	}
	return cv::Mat::diag(temp);
}


void GaussianProgress::train()
{
	cv::Mat f(1, sample_num, CV_32FC1,cv::Scalar(0));
	cv::Mat a(1, sample_num, CV_32FC1);
	deltaP = CalculateDelataP(LabelMat, f);
	do
	{
		W = CalculateW(LabelMat, f);
		cv::sqrt(W, sqrtW);
		cv::Mat temp = W*CovarianceMat;
		cv::Mat I(1, temp.rows, CV_32FC1, cv::Scalar(1.0));
		I = cv::Mat::diag(I);
/*		cv::PCA pca;
		pca(temp, cv::Mat(), cv::PCA::DATA_AS_ROW, 4);*/
		temp = I + temp;
		cv::Mat temp1 = I + temp;
		//print(temp);
		cv::Mat b = f * W + deltaP;
		L = CalculateL(temp);
		cv::Mat buffer = sqrtW * CovarianceMat * b.t();
		cv::solve(L, buffer, buffer);
		cv::solve(L.t(), buffer, buffer);
		a = (b - (sqrtW*buffer).t());
		print(a);
		std::cout << std::endl;
		f = a*CovarianceMat;
		

		
		/*temp1 = temp1.inv();
		a = temp1*b.t();
		std::cout << std::endl << "a :" << std::endl;
		print(a);
		f = (CovarianceMat * a).t();*/
		
	//	print(f);
		


		deltaP = CalculateDelataP(LabelMat, f);
	} while ((!ifconvergens(a, deltaP)));
}


cv::Mat GaussianProgress::CalculateL( cv::Mat& temp)
{
	int n = temp.cols;
	if (!Cholesky((float*)temp.data, sizeof(float) * n, n))
		throw std::runtime_error("not invertble");
	cv::Mat out(n, n, CV_32FC1, cv::Scalar(0));
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j <= i; ++j)
			out.at<float>(i, j) = temp.at<float>(i, j);
	}
	return out;
}

bool GaussianProgress::ifconvergens(const cv::Mat& p, const cv::Mat&a)
{
	cv::Mat temp;
	cv::absdiff(p, a,temp);
	auto diff = cv::sum(temp)[0] / sample_num;
	std::cout << "diff :" << diff;
	return diff < error;
}

/*
	Test data must be row type
*/

int GaussianProgress::predict(const cv::Mat& x)
{
	cv::Mat k21 = K(x, Sample,sigma),
		k22 = K(x, x,sigma);
	float f_ = ((cv::Mat)(k21 * deltaP.t())).at<float>(0,0);
	cv::Mat v;
	cv::solve(L, sqrtW * k21.t(),v);
	float sigma = ((cv::Mat) (k22 - v.t() * v)).at<float>(0, 0);
	/*
	approximation of Eq[π∗|X, y, x∗] = Z σ(f∗)q(f∗|X, y, x∗) df∗
	*/
	float gama = std::sqrt(1 / (1 + 3.1415926f*sigma / 8));
	float p = 1 / (1 + std::exp(-gama * f_));
	if (p > 0.5)
		return 1;
	else
		return -1;
}

cv::Mat GaussianProgress::CalculateCovariance(const Mat& src)
{
	return K(src, src,sigma);
}

cv::Mat GaussianProgress::CalculateDelataP(const cv::Mat& label, const cv::Mat& f)
{
	cv::Mat out(1, label.cols, CV_32FC1);
	for (int i = 0; i < sample_num; ++i)
		out.at<float>(cv::Point(i, 0)) = (label.at<int>(cv::Point(i, 0)) + 1.0f) / 2
		- 1.0f / (1 + std::exp(-f.at<float>(cv::Point(i, 0))));
	return out;
}


cv::Mat Line(const cv::Mat& src1, const cv::Mat& src2,double a)
{
	cv::Mat temp = src2.t();
	return src1 * temp;
}

cv::Mat Exponential(const cv::Mat& src1, const cv::Mat& src2,double sigma)
{
	int n = src1.rows;
	int m = src2.rows;
	cv::Mat out(n, m, CV_32FC1);
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j <m; ++j)
		{
			cv::Mat u = src1.row(i) - src2.row(j);
			auto l = cv::norm(u, cv::NormTypes::NORM_L2);
			out.at<float>(i, j) = std::exp(-l * 0.5);
		}
	}
	return out;
}


