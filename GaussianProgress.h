#pragma once
#include <opencv2\imgproc.hpp>
#include <opencv2\core.hpp>
#include <opencv\cv.hpp>
#include <functional>
#include <iostream>
#include <tempalte\functor.h>
#include <exception>
#include "cholesky.h"
#include <numeric>
cv::Mat Line(const cv::Mat&, const cv::Mat&,double a);
cv::Mat Exponential(const cv::Mat& src1, const cv::Mat& src2, double sigma = 0.5);
typedef cv::Mat KernelFunc(const cv::Mat&, const cv::Mat&,double );
class GaussianProgress
{
public:
	typedef cv::Mat Mat;
	typedef float  PosteriorFunc(int , float);
	void train();
	int predict(const Mat&);
	GaussianProgress();
	/*
	kernel function must be accept two cv::Mat and return cv::Mat
	*/
	template < typename Func>
	GaussianProgress(const Mat& sample, const Mat& label,Func f):Sample(sample),
		sample_num(sample.rows),
		LabelMat(label),
		K(f),
		sigma(0.5),
		error(0.5){CovarianceMat = CalculateCovariance(Sample);}
	template <typename Func>
	void setKernel(Func f) { K = std::function<KernelFunc>(f); }
	/*
	label map shoud be row type
	*/
	void setTraindata(const Mat& img,const Mat& label) { Sample = img;
	CovarianceMat = CalculateCovariance(Sample);
	sample_num = img.rows;
	LabelMat = label;
	}
	void setSigma(double a)
	{
		sigma = 0.5;
	}
	void setError(double s)
	{
		error = s;
	}
	~GaussianProgress();
private:
	int sample_num;
	std::function<KernelFunc> K;
	Mat CovarianceMat;
	Mat LabelMat;
	Mat deltaP;
	Mat sqrtW;
	Mat L;
	Mat W;
	Mat Sample;

	Mat CalculateW(const Mat& label, const Mat f);
	Mat CalculateL(cv::Mat& temp);
	Mat CalculateCovariance(const cv::Mat& src);
	Mat CalculateDelataP(const cv::Mat& label, const cv::Mat& f);
	bool ifconvergens(const cv::Mat&, const cv::Mat&);
	double sigma;
	double error;
};

