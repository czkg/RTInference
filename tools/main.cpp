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

#define SAMPLE_READ_WAIT_TIMEOUT 2000 //2000ms
#define radius 80   //radius of the ROI
#define size 96   //network input size
#define thres 0.7  //threshold to segment the hand from backgrounds

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

    if (argc < 2)
        rc = device.open(ANY_DEVICE);
    else
        rc = device.open(argv[1]);

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
		cv::imshow("frame", dis);
		cv::waitKey(1);

		//crop the image to feed into the network
		cv::Mat_<float> cropped = im32f(cv::Range(p1.y, p2.y), cv::Range(p1.x, p2.x));
		//resize to appropriate size so we can put it into network
		cv::Mat_<float> resized;
		cv::resize(cropped, resized, cv::Size(size, size), 0, 0, cv::INTER_AREA);
		//find the hand region
		cropped.setTo(1.0, (cropped > thres) | (cropped <= 0));
		//normalize hand
		cv::Mat_<float> normalized = Preprocess::normalizeHand(cropped);
		cv::Mat_<float> filtered = Preprocess::filter(normalized);
		filtered = (filtered + 1.0) / 2.0;
		// cv::imshow("filtered", filtered);
		// cv::waitKey(1);

		//feed the image into the network




		int middleIndex = (frame.getHeight()+1)*frame.getWidth()/2;

		printf("[%08llu] %8d\n", (long long)frame.getTimestamp(), pDepth[middleIndex]);
	}

	depth.stop();
	depth.destroy();
	device.close();
	OpenNI::shutdown();

	return 0;
}