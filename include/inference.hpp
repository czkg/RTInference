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
	Inference(const std::string& model_file, const std::string& weight_file);
	~Inference() {}
	std::vector<float> Predict(const std::vector<cv::Mat_<float> >& imgs);
	void WrapInputLayer(std::vector<cv::Mat>* input_channels, const int& idx);
	void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels, const int& idx);
	int numInputs();
private:
	std::shared_ptr<caffe::Net<float> > net_;
	std::vector<cv::Size> input_geometry_;
	int num_channels_;
};

#endif

