#include <inference.hpp>


Inference::Inference(const std::string& model_file, const std::string& weight_file) {
	//model file: deploy prototxt
	net_.reset(new caffe.Net<float>(model_file, TEST));
	//weight file: caffe model
	net_->CopyTrainedLayersFrom(weight_file);
	caffe.Blob<float>* input_layer = net_->input_blobs()[0];
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
	num_channels_ = input_layer->channels();
}

void Inference::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Inference::Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  sample_resized.convertTo(sample_float, CV_32FC1);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_float, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

std::vector<float> Inference::Predict(const cv::Mat& img) {
	caffe.Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);

	net_->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);
	Preprocess(img, &input_channels);

	net_->Forward();

	/* Copy the output layer to a std::vector */
  	Blob<float>* output_layer = net_->output_blobs()[0];
  	const float* begin = output_layer->cpu_data();
  	const float* end = begin + output_layer->channels();
  	return std::vector<float>(begin, end);
}

