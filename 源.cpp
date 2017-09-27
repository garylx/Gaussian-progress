#include <dlib\image_processing\frontal_face_detector.h>
#include <dlib\image_processing.h>
#include <dlib\image_io.h>
#include <dlib\image_io.h>
#include <dlib\opencv.h>
#include <dlib\opencv\to_open_cv.h>
#include <opencv2\imgproc.hpp>
#include <opencv2\core.hpp>
#include <opencv\cv.hpp>
#include <vector>
#include <chrono>
#include <functional>
#include <tempalte\functor.h>
#include "Eigenfaces.h"
#include "GaussianProgress.h"
int main()
{
	
	float a[22] = {
		1,0,
		2,5,
		9,10,
		-30,30
		-45,-45,
		5,20,
		6,6
		-8,-9,
		4,-8,
		9,-6,
		-44,-44
	};
	
	int l[11] = { 1,1,1,1,1,1,1,-1,-1,-1,-1 };
	cv::Mat train(11, 2, CV_32FC1, a),
		label(1, 11, CV_32SC1, l);
	GaussianProgress p(train, label, Exponential);
	p.setError(0.00002);
	p.train();
	
	cv::Mat img(100, 100, CV_8UC1);
	for (int i = 0; i < img.rows; ++i)
	{
		auto data = img.ptr<uchar>(i);
		for (int j = 0; j < img.cols; ++j)
		{
			float b[2] = { j - 50, i - 50 };
			cv::Mat test(1, 2, CV_32FC1, b);
			float c = p.predict(test);
			data[j] = static_cast<uchar>(c * 255);
		}
	}
	float b[2] = {20, 21 };
	cv::Mat test(1, 2, CV_32FC1, b);
	std::cout << p.predict(test);
	cv::imshow("ds", img);
	cv::waitKey();
	getchar();
}