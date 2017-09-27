#pragma once
#include <opencv2\imgproc.hpp>
#include <opencv2\core.hpp>
#include <opencv\cv.hpp>
#include <vector>
#include <numeric>
#include <iostream>
#include <exception>
#define ROW 112
#define COL 92
class Eigenfaces
{
public:
	Eigenfaces();
	void setData(const cv::Mat& m,const cv::Mat& l)
	{
		train_data = m;
		label = l;
	}
	inline void doPCA(const cv::Mat& train,double rateornum);
	void train(double rateornum = 0.99);
	std::vector<cv::Mat> eigenfaces(int row,int col);
	int predict(const cv::Mat& src);
	cv::Mat subtract_mean(const cv::Mat& src);
	void add_one(const cv::Mat& m);
	~Eigenfaces();
private:
	cv::Mat train_data;
	cv::Mat label;
	cv::Mat eigen_map;
	cv::PCA pca;
	int eigen_num;
	int  find_closest(const cv::Mat& train, const cv::Mat& predict);
};

