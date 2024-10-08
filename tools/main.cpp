/*! 
 *  \brief     main cpp file for inference.
 *  \author    Zhi Chai
 *  \date      May 9 2017
 */
#include <iostream>
#include <OpenNI.h>
#include <Viewer.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <utilities.hpp>
#include <preprocess.hpp>
#include <inference.hpp>
#include <vector>
#include <algorithm>
#include <caffe/caffe.hpp>

#define SAMPLE_READ_WAIT_TIMEOUT 2000 //2000ms
#define radius 120   //radius of the ROI
#define input_size 96   //network input size
#define thres 0.6  //threshold to segment the hand from backgrounds
#define heatmap_width 20 //width of the heatmap
#define mul 2  //the multiple of the dispaly window

using namespace openni;
using caffe::Caffe;

int main(int argc, char** argv)
{
	Status rc = OpenNI::initialize();
	if (rc != STATUS_OK)
	{
		printf("Initialize failed\n%s\n", OpenNI::getExtendedError());
		return 1;
	}

	Device device;

    //rc = device.open("/home/cz/RTInference/Captured.oni");
    rc = device.open(ANY_DEVICE);

    if (rc != STATUS_OK)
	{
		printf("Couldn't open device\n%s\n", OpenNI::getExtendedError());
		return 2;
	}

	VideoStream depth;

	if (device.getSensorInfo(SENSOR_DEPTH) != NULL)
	{
		rc = depth.create(device, SENSOR_DEPTH);
		if (rc != STATUS_OK)
		{
			printf("Couldn't create depth stream\n%s\n", OpenNI::getExtendedError());
			return 3;
		}
	}

	rc = depth.start();
	if (rc != STATUS_OK)
	{
		printf("Couldn't start the depth stream\n%s\n", OpenNI::getExtendedError());
		return 4;
	}

	VideoFrameRef frame;

	if(argc != 6) {
		std::cout << "Accept 5 arguments!" << std::endl;
		return 6;
	}

	//arg 1: model_file: deploy prototxt
	std::string model_file = argv[1];
	//arg 2: weight file: caffe model
	std::string weight_file = argv[2];
	//arg 3: finger tips only or full hands
	int isFT = std::atoi(argv[3]);
	//arg 4: GPU or CPU
	int isGPU = std::atoi(argv[4]);
	int device_id = 0;
	//arg 5: uvd or uv
	int isUVD = std::atoi(argv[5]);

	if(isGPU) {
		std::cout << "Use GPU." << std::endl;
		Caffe::SetDevice(device_id);
    	Caffe::set_mode(Caffe::GPU);
	}
	else {
		std::cout << "Use CPU." << std::endl;
		Caffe::set_mode(Caffe::CPU);
	}

	Inference inference(model_file, weight_file);
	int count = 0;

	while (!wasKeyboardHit())
	{
		int changedStreamDummy;
		VideoStream* pStream = &depth;
		rc = OpenNI::waitForAnyStream(&pStream, 1, &changedStreamDummy, SAMPLE_READ_WAIT_TIMEOUT);
		if (rc != STATUS_OK)
		{
			printf("Wait failed! (timeout is %d ms)\n%s\n", SAMPLE_READ_WAIT_TIMEOUT, OpenNI::getExtendedError());
			continue;
		}

		rc = depth.readFrame(&frame);
		if (rc != STATUS_OK)
		{
			printf("Read failed!\n%s\n", OpenNI::getExtendedError());
			continue;
		}

		if (frame.getVideoMode().getPixelFormat() != PIXEL_FORMAT_DEPTH_1_MM && frame.getVideoMode().getPixelFormat() != PIXEL_FORMAT_DEPTH_100_UM)
		{
			printf("Unexpected frame format\n");
			continue;
		}

		DepthPixel* pDepth = (DepthPixel*)frame.getData();

		cv::Mat im(frame.getHeight(), frame.getWidth(), CV_16SC1, pDepth);
		cv::Mat im32f;
		im.convertTo(im32f, CV_32FC1, 1.0/1000.0);
		cv::Mat dis;
		cv::normalize(im32f, dis, 0, 1, cv::NORM_MINMAX);
		cv::cvtColor(dis, dis, CV_GRAY2BGR);

		//define the ROI
		cv::Point center(frame.getWidth() / 2.0, frame.getHeight() / 2.0);
		cv::Point p1(center.x - radius, center.y - radius);
		cv::Point p2(center.x + radius, center.y + radius);
		cv::rectangle(dis, p1, p2, cv::Scalar(0, 0, 255));

		//crop the image to feed into the network
		cv::Mat_<float> cropped = im32f(cv::Range(p1.y, p2.y), cv::Range(p1.x, p2.x));
		//find the hand region
		cropped.setTo(1.0, (cropped > thres) | (cropped <= 0));
		//normalize hand
		cv::Mat_<float> roi = Preprocess::findROI(cropped);
		cv::Mat mm = (roi != 2.0);
		int roi_width = roi.cols;
		cv::Mat roi_dis = roi.clone();
		//if(!isUVD) {
			roi_dis = Preprocess::filter(roi);
		//}
		cv::cvtColor(roi_dis, roi_dis, CV_GRAY2BGR);

		//resize to appropriate size so we can put it into network
		cv::Mat_<float> roi_high, roi_mid, roi_low;
		std::vector<cv::Mat_<float> > imgs;
		if(inference.numInputs() == 1) {
			cv::resize(roi, roi, cv::Size(input_size, input_size), 0, 0, cv::INTER_AREA);
			//cv::Mat_<float> roi_input = isUVD ? roi : Preprocess::filter(roi);
			cv::Mat_<float> roi_input = Preprocess::filter(roi);
			imgs.push_back(roi_input);
		}
		else {
			cv::resize(roi, roi_high, cv::Size(input_size, input_size), 0, 0, cv::INTER_AREA);
			cv::resize(roi, roi_mid, cv::Size(input_size / 2, input_size / 2), 0, 0, cv::INTER_AREA);
			cv::resize(roi, roi_low, cv::Size(input_size / 4, input_size / 4), 0, 0, cv::INTER_AREA);
			//cv::Mat_<float> roi_input_high = isUVD ? roi_high : Preprocess::filter(roi_high);
			//cv::Mat_<float> roi_input_mid = isUVD ? roi_mid : Preprocess::filter(roi_mid);
			//cv::Mat_<float> roi_input_low = isUVD ? roi_low : Preprocess::filter(roi_low);
			cv::Mat_<float> roi_input_high = Preprocess::filter(roi_high);
			cv::Mat_<float> roi_input_mid = Preprocess::filter(roi_mid);
			cv::Mat_<float> roi_input_low = Preprocess::filter(roi_low);
			imgs.push_back(roi_input_high);
			imgs.push_back(roi_input_mid);
			imgs.push_back(roi_input_low);
		}

		//feed the image into the network
		std::vector<float> result = inference.Predict(imgs);
		int heatmap_size = heatmap_width * heatmap_width;
		std::vector<cv::Point> coordinates;
		if(isFT) {
			coordinates.resize(5);
		}
		else {
			coordinates.resize(20);
		}
		
		std::vector<float> xs, ys;

		//store all joints
		if(isFT) {
			for(int i = 0; i < 5; i++) {
				float x, y;
				if(isUVD) {
					x = result[i * 2] /(float)input_size * (float)roi_width;
					y = result[i * 2 + 1] /(float)input_size * (float)roi_width;
				}
				else {
					std::vector<float> current(result.begin() + i * heatmap_size, result.begin() + (i + 1) * heatmap_size);
					auto ite_max = std::max_element(current.begin(), current.end());
					int max = std::distance(current.begin(), ite_max);
					x = (max % heatmap_width) / (float)heatmap_width * (float)roi_width;
					y = (max / heatmap_width) / (float)heatmap_width * (float)roi_width;
				}
				xs.push_back(x);
				ys.push_back(y);
			}
		}
		else {
			for(int i = 0; i < 20; i++) {
				float x, y;
				if(isUVD) {
					x = result[i * 2] /(float)input_size * (float)roi_width;
					y = result[i * 2 + 1] /(float)input_size * (float)roi_width;
				}
				else {
					std::vector<float> current(result.begin() + i * heatmap_size, result.begin() + (i + 1) * heatmap_size);
					auto ite_max = std::max_element(current.begin(), current.end());
					int max = std::distance(current.begin(), ite_max);
					x = (max % heatmap_width) / (float)heatmap_width * (float)roi_width;
					y = (max / heatmap_width) / (float)heatmap_width * (float)roi_width;
				}
				xs.push_back(x);
				ys.push_back(y);
			}
		}
		
		if(isFT) {
			//thumb tip
			cv::circle(roi_dis, cv::Point(xs[0], ys[0]), 2, cv::Scalar(0, 255, 0));

			//index finger tip
			cv::circle(roi_dis, cv::Point(xs[1], ys[1]), 2, cv::Scalar(255, 0, 255));

			//middle finger tip
			cv::circle(roi_dis, cv::Point(xs[2], ys[2]), 2, cv::Scalar(0, 255, 255));

			//ring finger tip
			cv::circle(roi_dis, cv::Point(xs[3], ys[3]), 2, cv::Scalar(255, 0, 0));

			//little finger tip
			cv::circle(roi_dis, cv::Point(xs[4], ys[4]), 2, cv::Scalar(0, 0, 255));
		}
		else {
			cv::circle(roi_dis, cv::Point(xs[0], ys[0]), 2, cv::Scalar(255, 255, 255));

			//thumb
			cv::circle(roi_dis, cv::Point(xs[1], ys[1]), 2, cv::Scalar(0, 255, 0));
			cv::circle(roi_dis, cv::Point(xs[2], ys[2]), 2, cv::Scalar(0, 255, 0));
			cv::circle(roi_dis, cv::Point(xs[3], ys[3]), 2, cv::Scalar(0, 255, 0));
			cv::line(roi_dis, cv::Point(xs[0], ys[0]), cv::Point(xs[1], ys[1]), cv::Scalar(0, 255, 0));
			cv::line(roi_dis, cv::Point(xs[1], ys[1]), cv::Point(xs[2], ys[2]), cv::Scalar(0, 255, 0));
			cv::line(roi_dis, cv::Point(xs[2], ys[2]), cv::Point(xs[3], ys[3]), cv::Scalar(0, 255, 0));

			//index finger
			cv::circle(roi_dis, cv::Point(xs[4], ys[4]), 2, cv::Scalar(255, 0, 255));
			cv::circle(roi_dis, cv::Point(xs[5], ys[5]), 2, cv::Scalar(255, 0, 255));
			cv::circle(roi_dis, cv::Point(xs[6], ys[6]), 2, cv::Scalar(255, 0, 255));
			cv::circle(roi_dis, cv::Point(xs[7], ys[7]), 2, cv::Scalar(255, 0, 255));
			cv::line(roi_dis, cv::Point(xs[0], ys[0]), cv::Point(xs[4], ys[4]), cv::Scalar(255, 0, 255));
			cv::line(roi_dis, cv::Point(xs[4], ys[4]), cv::Point(xs[5], ys[5]), cv::Scalar(255, 0, 255));
			cv::line(roi_dis, cv::Point(xs[5], ys[5]), cv::Point(xs[6], ys[6]), cv::Scalar(255, 0, 255));
			cv::line(roi_dis, cv::Point(xs[6], ys[6]), cv::Point(xs[7], ys[7]), cv::Scalar(255, 0, 255));

			//middle finger
			cv::circle(roi_dis, cv::Point(xs[8], ys[8]), 2, cv::Scalar(0, 255, 255));
			cv::circle(roi_dis, cv::Point(xs[9], ys[9]), 2, cv::Scalar(0, 255, 255));
			cv::circle(roi_dis, cv::Point(xs[10], ys[10]), 2, cv::Scalar(0, 255, 255));
			cv::circle(roi_dis, cv::Point(xs[11], ys[11]), 2, cv::Scalar(0, 255, 255));
			cv::line(roi_dis, cv::Point(xs[0], ys[0]), cv::Point(xs[8], ys[8]), cv::Scalar(0, 255, 255));
			cv::line(roi_dis, cv::Point(xs[8], ys[8]), cv::Point(xs[9], ys[9]), cv::Scalar(0, 255, 255));
			cv::line(roi_dis, cv::Point(xs[9], ys[9]), cv::Point(xs[10], ys[10]), cv::Scalar(0, 255, 255));
			cv::line(roi_dis, cv::Point(xs[10], ys[10]), cv::Point(xs[11], ys[11]), cv::Scalar(0, 255, 255));

			//ring finger
			cv::circle(roi_dis, cv::Point(xs[12], ys[12]), 2, cv::Scalar(255, 0, 0));
			cv::circle(roi_dis, cv::Point(xs[13], ys[13]), 2, cv::Scalar(255, 0, 0));
			cv::circle(roi_dis, cv::Point(xs[14], ys[14]), 2, cv::Scalar(255, 0, 0));
			cv::circle(roi_dis, cv::Point(xs[15], ys[15]), 2, cv::Scalar(255, 0, 0));
			cv::line(roi_dis, cv::Point(xs[0], ys[0]), cv::Point(xs[12], ys[12]), cv::Scalar(255, 0, 0));
			cv::line(roi_dis, cv::Point(xs[12], ys[12]), cv::Point(xs[13], ys[13]), cv::Scalar(255, 0, 0));
			cv::line(roi_dis, cv::Point(xs[13], ys[13]), cv::Point(xs[14], ys[14]), cv::Scalar(255, 0, 0));
			cv::line(roi_dis, cv::Point(xs[14], ys[14]), cv::Point(xs[15], ys[15]), cv::Scalar(255, 0, 0));

			//little finger
			cv::circle(roi_dis, cv::Point(xs[16], ys[16]), 2, cv::Scalar(0, 0, 255));
			cv::circle(roi_dis, cv::Point(xs[17], ys[17]), 2, cv::Scalar(0, 0, 255));
			cv::circle(roi_dis, cv::Point(xs[18], ys[18]), 2, cv::Scalar(0, 0, 255));
			cv::circle(roi_dis, cv::Point(xs[19], ys[19]), 2, cv::Scalar(0, 0, 255));
			cv::line(roi_dis, cv::Point(xs[0], ys[0]), cv::Point(xs[16], ys[16]), cv::Scalar(0, 0, 255));
			cv::line(roi_dis, cv::Point(xs[16], ys[16]), cv::Point(xs[17], ys[17]), cv::Scalar(0, 0, 255));
			cv::line(roi_dis, cv::Point(xs[17], ys[17]), cv::Point(xs[18], ys[18]), cv::Scalar(0, 0, 255));
			cv::line(roi_dis, cv::Point(xs[18], ys[18]), cv::Point(xs[19], ys[19]), cv::Scalar(0, 0, 255));
		}

		//show image
		cv::imshow("frame", dis);
		cv::imshow("results", roi_dis);
		std::string name = "res/" + std::to_string(++count) + ".jpg";
		roi_dis.setTo(cv::Scalar(255, 255, 255), ~mm);
		cv::imwrite(name, dis * 255);
		cv::waitKey(1);

		int middleIndex = (frame.getHeight()+1)*frame.getWidth()/2;

		//printf("[%08llu] %8d\n", (long long)frame.getTimestamp(), pDepth[middleIndex]);
	}

	depth.stop();
	depth.destroy();
	device.close();
	OpenNI::shutdown();

	return 0;
}