#ifndef Utils_h
#define Utils_h
#include <opencv2/opencv.hpp>
#include <OpenCL/OpenCL.h>
namespace sd {


#define BIAS 0
#define SCALE 1
#define MEAN 2
#define VARIANCE 3
	class Utils {
	public:
		static cv::Mat createfastNormalizeMat(cv::Mat biasScaleMeanVariance) {
			cv::Mat returnMat(cv::Size(biasScaleMeanVariance.cols, 2), biasScaleMeanVariance.type());
			for (int gy = 0; gy < biasScaleMeanVariance.cols; gy++) {
				returnMat.at<float>(cv::Point(gy, 0)) = biasScaleMeanVariance.at<float>(cv::Point(gy, SCALE)) / (sqrt(biasScaleMeanVariance.at<float>(cv::Point(gy, VARIANCE))) + .000001f);
				returnMat.at<float>(cv::Point(gy, 1)) = biasScaleMeanVariance.at<float>(cv::Point(gy, BIAS)) - (biasScaleMeanVariance.at<float>(cv::Point(gy, MEAN)) * returnMat.at<float>(cv::Point(gy, 0)));
			}
			return returnMat;
		}


		template <class T>
		static cv::Mat readFromFile(cv::Size size, int type, FILE* fp) {
			int numbers = size.height * size.width;
			T *file = (T*)malloc(sizeof(T) * numbers);
			fread(file, sizeof(T), numbers, fp);
			cv::Mat mat_ = cv::Mat(size, type);
			mat_.data = (uchar*)file;
			return mat_;
		}

	};

	class Box {
	public:
		float x, y, w, h;
	};

	class ImageBox {
	public:
		int className;
		int x, y, width, height;
	};
	
}
#endif

