#include <inference.hpp>

using caffe::Caffe;


Inference::Inference(const std::string& model_file, const std::string& weight_file) {
	//model file: deploy prototxt
	net_.reset(new caffe::Net<float>(model_file, caffe::TEST));
	//weight file: caffe model
	net_->CopyTrainedLayersFrom(weight_file);

	input_geometry_.resize(numInputs());
	caffe::Blob<float>* input_layer;

	for(int idx = 0; idx < numInputs(); idx++) {
		input_layer = net_->input_blobs()[idx];
		input_geometry_[idx] = cv::Size(input_layer->width(), input_layer->height());
	}
	num_channels_ = input_layer->channels();
}

int Inference::numInputs() {
	return net_->input_blobs().size();
}

void Inference::WrapInputLayer(std::vector<cv::Mat>* input_channels, const int& idx) {
  caffe::Blob<float>* input_layer = net_->input_blobs()[idx];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Inference::Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels, const int& idx) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_[idx])
    cv::resize(sample, sample_resized, input_geometry_[idx]);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  sample_resized.convertTo(sample_float, CV_32FC1);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_float, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[idx]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

std::vector<float> Inference::Predict(const std::vector<cv::Mat_<float> >& imgs) {
	for(int idx = 0; idx < numInputs(); idx++) {
		caffe::Blob<float>* input_layer = net_->input_blobs()[idx];
		input_layer->Reshape(1, num_channels_, input_geometry_[idx].height, input_geometry_[idx].width);
	}

	net_->Reshape();

	for(int idx = 0; idx < numInputs(); idx++) {
		std::vector<cv::Mat> input_channels;
		WrapInputLayer(&input_channels, idx);
		Preprocess(imgs[idx], &input_channels, idx);
	}

	net_->Forward();

	/* Copy the output layer to a std::vector */
  	caffe::Blob<float>* output_layer = net_->output_blobs()[0];
  	const float* begin = output_layer->cpu_data();
  	const float* end = begin + output_layer->channels();
  	return std::vector<float>(begin, end);
}

