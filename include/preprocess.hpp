/*! 
 *  \brief     hpp file for class Preprocess.
 *  \author    Zhi Chai
 *  \date      May 11 2017
 */
#ifndef PREPROCESS_HPP
#define PREPROCESS_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class Preprocess {
public:
	Preprocess() {}
	~Preprocess() {}
	static cv::Mat_<float> filter(const cv::Mat_<float>& img);
	static cv::Mat_<float> findROI(const cv::Mat_<float>& img);
};

#endif