#ifndef INFERENCE_HPP
#define INFERENCE_HPP

#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

class Inference {
public:
	Inference();
	~Inference() {}
	std::vector<float> Predict(const cv::Mat& img);
	void WrapInputLayer(std::vector<cv::Mat>* input_channels);
	void Inference::Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);
private:
	std::shared_ptr<caffe.Net<float> > net_;
	cv::Size input_geometry_;
	int num_channels_;
};

#endif

