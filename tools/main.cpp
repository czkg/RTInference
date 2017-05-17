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

#define SAMPLE_READ_WAIT_TIMEOUT 2000 //2000ms
#define radius 80   //radius of the ROI
#define input_size 96   //network input size
#define thres 0.7  //threshold to segment the hand from backgrounds
#define heatmap_width 20 //width of the heatmap
#define mul 3  //the multiple of the dispaly window

using namespace openni;

int main(int argc, char** argv)
{
	Status rc = OpenNI::initialize();
	if (rc != STATUS_OK)
	{
		printf("Initialize failed\n%s\n", OpenNI::getExtendedError());
		return 1;
	}

	Device device;

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

	std::string model_file = argv[1];
	std::string weight_file = argv[2];
	Inference inference(model_file, weight_file);

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

		//show image
		// cv::imshow("frame", dis);
		// cv::waitKey(1);

		//crop the image to feed into the network
		cv::Mat_<float> cropped = im32f(cv::Range(p1.y, p2.y), cv::Range(p1.x, p2.x));
		//find the hand region
		cropped.setTo(1.0, (cropped > thres) | (cropped <= 0));
		//normalize hand
		cv::Mat_<float> normalized = Preprocess::normalizeHand(cropped);
		//resize to appropriate size so we can put it into network
		cv::Mat_<float> normalized_high, normalized_mid, normalized_low;
		std::vector<cv::Mat_<float> > imgs;
		if(inference.numInputs() == 1) {
			cv::resize(normalized, normalized, cv::Size(input_size, input_size), 0, 0, cv::INTER_AREA);
			imgs.push_back(normalized);
		}
		else {
			cv::resize(normalized, normalized_high, cv::Size(input_size, input_size), 0, 0, cv::INTER_AREA);
			cv::resize(normalized, normalized_mid, cv::Size(input_size / 2, input_size / 2), 0, 0, cv::INTER_AREA);
			cv::resize(normalized, normalized_low, cv::Size(input_size / 4, input_size / 4), 0, 0, cv::INTER_AREA);
			imgs.push_back(normalized_high);
			imgs.push_back(normalized_mid);
			imgs.push_back(normalized_low);
		}
		// cv::Mat_<float> filtered = Preprocess::filter(normalized);
		// filtered = (filtered + 1.0) / 2.0;

		//feed the image into the network
		std::vector<float> result = inference.Predict(imgs);
		int heatmap_size = heatmap_width * heatmap_width;
		std::vector<cv::Point> coordinates;
		coordinates.resize(20);
		//cv::cvtColor(filtered, filtered, CV_GRAY2BGR);

		// cv::Mat_<float> nor_dis = normalized;
		// cv::resize(nor_dis, nor_dis, cv::Size(), mul, mul, cv::INTER_AREA);

		for(int i = 0; i < 20; i++) {
			std::vector<float> current(result.begin() + i * heatmap_size, result.begin() + (i + 1) * heatmap_size);
			auto ite_max = std::max_element(current.begin(), current.end());
			int max = std::distance(current.begin(), ite_max);
			// float x = (max % heatmap_width) / (float)heatmap_width * (float)input_size * (float)mul;
			// float y = (max / heatmap_width) / (float)heatmap_width * (float)input_size * (float)mul;
			float x = (max % heatmap_width) / (float)heatmap_width * (float)radius * 2.0;
			float y = (max / heatmap_width) / (float)heatmap_width * (float)radius * 2.0;
			x += p1.x;
			y += p1.y;
			coordinates[i] = cv::Point(x, y);
			cv::circle(dis, coordinates[i], 2, cv::Scalar(0, 0, 255));
		}
		// cv::imshow("normalized", nor_dis);
		// cv::waitKey(1);

		//show image
		cv::imshow("frame", dis);
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