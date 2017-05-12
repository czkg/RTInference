/*! 
 *  \brief     cpp file for class Preprocess.
 *  \author    Zhi Chai
 *  \date      May 11 2017
 */
#include <preprocess.hpp>
#include <iostream>


cv::Mat_<float> Preprocess::filter(const cv::Mat_<float>& img) {
	const cv::Size kernelSize(9, 9);
    cv::Mat_<float> des;
    cv::Mat_<float> delta;

    cv::GaussianBlur(img, des, kernelSize, 0);

    if(img.rows != des.rows || img.cols != des.cols ) {
    	std::cout << "Demension should be the same!" << std::endl;
    }

   	des = img - des;
   	 
   	cv::pow(des, 2, delta);
   	cv::GaussianBlur(delta, delta, kernelSize, 0);
   	cv::sqrt(delta, delta);
   	float c = cv::mean(delta).val[0];
   	delta = cv::max(delta, c);

    des = des / delta;
    
    return des;
}


cv::Mat_<float> Preprocess::normalizeHand(const cv::Mat_<float>& img) {
	float max = -10.0;
    float min = 10.0;
    cv::Mat_<float> des = img;

    for(int i = 0; i < img.rows; i++) {
    	for(int j = 0; j < img.cols; j++) {
    		if(img(i, j) != 1.0) {
    			if(img(i, j) < min) {
    				min = img(i, j);
    			}
    			if(img(i, j) > max) {
    				max = img(i, j);
    			}
    		}
    	}
    }

    for(int i = 0; i < des.rows; i++) {
    	for(int j = 0; j < des.cols; j++) {
    		if(des(i, j) != 1.0) {
    			des(i, j) = (des(i, j) - min) / (max - min);
    		}
    		else {
    			des(i, j) = 2.0;
    		}
    	}
    }

    return des;
}
