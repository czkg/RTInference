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

bool GetSquareImage(cv::Mat& depth, const int& target_width) {
    int width = depth.cols, height = depth.rows;

    int top = 0, bottom = 0, left = 0, right = 0;

    if(width > height) {
        bottom = (width - height) / 2;
        top = width - height - bottom;          
    }
    else if(width < height) {
        left = (height - width) / 2;
        right = height - width - left;   
    }
    const int pad = 4;
    top += pad;
    bottom += pad;
    left += pad;
    right += pad;

    cv::Scalar depth_val(2.0);
    //pad the image
    cv::copyMakeBorder(depth, depth, top, bottom, left, right, cv::BORDER_CONSTANT, depth_val);

    return true;
}

cv::Mat_<float> Preprocess::findROI(const cv::Mat_<float>& img) {
	double maxv = -10.0;
    double minv = 10.0;

    cv::Mat origMask = (img != 1.0);

    if (countNonZero(origMask) < 100) {
    	return img.clone();
    }

    cv::Mat mask = origMask.clone();
    cv::erode(mask, mask, cv::Mat(), cv::Point(-1,-1), 2);
    cv::minMaxIdx(img, &minv, &maxv, 0, 0, mask);

    cv::Mat des;
    cv::Mat((img - minv) / (maxv - minv)).copyTo(des);
    des.setTo(2.0, ~origMask);


    std::vector<std::vector<cv::Point2i> > ctrs;
    int biggest = -1;
    int size = -1;
    cv::findContours(origMask, ctrs, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    for (int c = 0; c < ctrs.size(); c++) {
        if (cv::contourArea(ctrs[c]) > size) {
            biggest = c;
            size = cv::contourArea(ctrs[c]);
        }
    }
    cv::Rect box = cv::boundingRect(ctrs[biggest]);
    cv::Mat des_roi;
    des(box).copyTo(des_roi);

    GetSquareImage(des_roi, 100);

    return des_roi;
}
