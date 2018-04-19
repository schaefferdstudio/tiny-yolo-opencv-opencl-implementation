#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/core.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "ConvolutionalLayer.h"
#include "MaxPooling.h"
#include "DetectionLayer.h"
#include "PreparingLayer.h"
#include <cstdint>
using namespace std;
using namespace cv;
static double secsGPU;
static double secsCPU;
static UMat kernelGPU;
static Mat kernelCPU;


Mat reshapeImage(Mat im) {
	Mat returnMat = Mat(Size(im.size().width * im.size().height, im.channels()), CV_32F);
	for (int i = 0; i < im.size().width; i++) {
		for (int j = 0; j < im.size().height; j++) {
			Point point(i, j);
			Vec3f p = im.at<Vec3f>(Point(i, j));
			for (int x = 0; x < im.channels(); x++) {
				returnMat.at<float>(Point(i + j*im.size().width, x)) = p.val[x];
			}
		}	
	}
	return returnMat;
}

Mat reshapeImageInt(Mat im) {
	Mat returnMat = Mat(Size(im.size().width * im.size().height, im.channels()), CV_8UC1);
	for (int i = 0; i < im.size().width; i++) {
		for (int j = 0; j < im.size().height; j++) {
			Point point(i, j);
			Vec3b p = im.at<Vec3b>(Point(i, j));
			for (int x = 0; x < im.channels(); x++) {
				returnMat.at<uchar>(Point(i + j*im.size().width, x)) = p.val[x];
			}
		}
	}
	return returnMat;
}




template <class T>
void set_pixel(Mat* m, int x, int y, int c, T val)
{
	T* pointer = (T*)(m->data);
	pointer[x*m->channels() + c + y * m->size().width*m->channels()]= val;
}

template <class T>
float get_pixel(Mat* m, int x, int y, int c)
{
	T* pointer = (T*)(m->data);
	return pointer[x*m->channels() + c + y * m->size().width*m->channels()];
}

template <class T>
void add_pixel(Mat* m, int x, int y, int c, T val)
{
	T* pointer = (T*)(m->data);
	pointer[x*m->channels() + c + y * m->size().width*m->channels()] += val;
}

template <class T>
Mat resize_image(Mat im, int w, int h, int channels) {
	Mat resized(Size(w, h), im.type());
	Mat part(im.size(), im.type());
	int r, c, k;
	float w_scale = (float)(im.size().width - 1) / (w - 1);
	float h_scale = (float)(im.size().height - 1) / (h - 1);
	for (k = 0; k < channels; ++k) {
		for (r = 0; r < im.size().height; ++r) {
			for (c = 0; c < w; ++c) {
				T val = 0;
				if (c == w - 1 || im.size().width == 1) {
					val = get_pixel<T>(&im, im.size().width - 1, r, k);
				}
				else {
					float sx = c*w_scale;
					int ix = (int)sx;
					T dx = sx - ix;
					val = (1 - dx) * get_pixel<T>(&im, ix, r, k) + dx * get_pixel<T>(&im, ix + 1, r, k);
				}
				set_pixel<T>(&part, c, r, k, val);
			}
		}
	}

	for (k = 0; k < channels; ++k) {
		for (r = 0; r < h; ++r) {
			float sy = r*h_scale;
			int iy = (int)sy;
			float dy = sy - iy;
			for (c = 0; c < w; ++c) {
				T val = (1 - dy) * get_pixel<T>(&part, c, iy, k);
				set_pixel<T>(&resized, c, r, k, val);
			}
			if (r == h - 1 || im.size().height == 1) continue;
			for (c = 0; c < w; ++c) {
				T val = dy * get_pixel<T>(&part, c, iy + 1, k);
				add_pixel<T>(&resized, c, r, k, val);
			}
		}
	}
	return resized;
}

int max_index(float *a, int n)
{
	if (n <= 0) return -1;
	int i, max_i = 0;
	float max = a[0];
	for (i = 1; i < n; ++i) {
		if (a[i] > max) {
			max = a[i];
			max_i = i;
		}
	}
	return max_i;
}

vector<sd::ImageBox> get_detections(Mat im, int num, float thresh, std::vector<sd::Box> boxes, Mat probs, int classes)
{
	int i;
	int first = 1;
	vector<sd::ImageBox> returnVec;
	for (i = 0; i < num; ++i) {

		float* p = probs.ptr<float>(i*classes);
		int act_class = max_index(p, classes);
		float prob = probs.at<float>(Point(i, act_class));
		if (prob > thresh) {

			int width = im.rows * .012;

			if (0) {
				width = pow(prob, 1. / 2.) * 10 + 1;
				
			}
			//printf("%s: %.0f%%\n", , prob * 100);
			sd::Box b = boxes.at(i);

			int left = (b.x - b.w / 2.)*im.cols;
			int right = (b.x + b.w / 2.)*im.cols;
			int top = (b.y - b.h / 2.)*im.rows;
			int bot = (b.y + b.h / 2.)*im.rows;

			if (left < 0) left = 0;
			if (right > im.cols - 1) right = im.cols - 1;
			if (top < 0) top = 0;
			if (bot > im.rows - 1) bot = im.rows - 1;

			// ------------------------------ own code ------------------------------------------
			sd::ImageBox boxImg;
			boxImg.x = left;
			boxImg.y = top;
			boxImg.width = right - left;
			boxImg.height = bot - top;
			boxImg.className = act_class;
			returnVec.push_back(boxImg);
			// ------------------------------------------------------------------------


		}
	}
	return returnVec;
}

int main() {
	//cv::ocl::setUseOpenCL(false);
	FILE *fp = fopen("yolov2-tiny-voc.weights", "rb");

	Mat major_la = sd::Utils::readFromFile<int>(Size(4, 1), CV_32SC1, fp);

	vector<sd::Layer*> layers;

	//im.copyTo(imCL);
	//programSource = initGPU(imCL);
	int factor = 1;
    float thresh = 0.5;
	sd::DetectionLayer* detect = new sd::DetectionLayer(13, 30, 30, thresh);
	sd::PreparingLayer* prepare = new sd::PreparingLayer(416, 3, 3);
	layers.push_back(new sd::ConvolutionalLayer(416, 16, 3, 3));
	layers.push_back(new sd::MaxPooling(416, 16, 16, 2));
	layers.push_back(new sd::ConvolutionalLayer(208, 32, 16, 3));
	layers.push_back(new sd::MaxPooling(208, 32, 32, 2));
	layers.push_back(new sd::ConvolutionalLayer(104, 64, 32, 3));
	layers.push_back(new sd::MaxPooling(104, 64, 64, 2));
	layers.push_back(new sd::ConvolutionalLayer(52, 128, 64, 3));
	layers.push_back(new sd::MaxPooling(52, 128, 128, 2));
	layers.push_back(new sd::ConvolutionalLayer(26, 256, 128, 3));
	layers.push_back(new sd::MaxPooling(26, 256, 256, 2));
	layers.push_back(new sd::ConvolutionalLayer(13, 512, 256, 3));
	layers.push_back(new sd::MaxPooling(13, 512, 512, 1));
	layers.push_back(new sd::ConvolutionalLayer(13, 1024, 512, 3));
	layers.push_back(new sd::ConvolutionalLayer(13, 1024, 1024, 3));
	layers.push_back(new sd::ConvolutionalLayer(13, 30, 1024, 1, ACTIVATION_LINEAR));
	layers.push_back(detect);

    detect->init(fp);
	prepare->init(fp);
	for (int i = 0; i < layers.size() -1 ; i++) {
		layers.at(i)->init(fp);
	}
	Mat* before;
	Mat dstTmp, dst;
    

	
	//int64 t0 = cv::getTickCount();
	//for (int j = 0; j < factor; j++) {
	//	before = &im;
	//	for (int i = 0; i < layers.size(); i++) {
	//		dst = layers.at(i)->runOnCPU(before);
	//		before = &dst;
	//	}
	//}
	//int64 t1 = cv::getTickCount();
	//secsCPU = (t1 - t0) / cv::getTickFrequency() / (float)factor;
	//cout << "Cpu needs: " << secsCPU << endl;

	VideoCapture cap(0); // open the default camera
	if (!cap.isOpened())  // check if we succeeded
		return 1;

	Mat source;
	Mat sourceclone;
	Mat res, im;
	Mat dstGPU;
	UMat* beforeGP;
	UMat dstGPTmp;
	vector<sd::ImageBox> boxes; Mat probs;
	//Mat dstGPU = Mat
    UMat testres = UMat(Size(169, 30), CV_32FC1);
	while (true) {
		
		cap >> source;
		//source = imread("453.jpg");
		
		
        resize(source, source, Size(source.size().width / 2, source.size().height / 2));
        sourceclone = source.clone();
		cvtColor(source, source, CV_BGR2RGB);

		////source.convertTo(source, CV_32FC3);
		////source = source / 255.0;
		////resize(source, res, Size(416, 416));
		////im = reshapeImage(res);
		
		resize(source, source, Size(416, 416));
		
		res = reshapeImageInt(source);
        UMat imCL;
        res.copyTo(imCL);
        imCL.convertTo(imCL, CV_32FC1);
        imCL = prepare->runOnGPU(&imCL);
        UMat dstGP;
        UMat res;
        UMat matref;
        
        int64 t2 = cv::getTickCount();
        beforeGP = &imCL;
        for (int i = 0; i < layers.size(); i++) {
            dstGP = layers.at(i)->runOnGPU(beforeGP);
            beforeGP = &dstGP;
        }
        int64 t3 = cv::getTickCount();
        
        secsGPU = (t3 - t2) / cv::getTickFrequency();
        cout << "Neuralnet needs: " << secsGPU << endl;

        int64 t0 = cv::getTickCount();
        Mat test = detect->gpuDst.getMat(ACCESS_READ);
        int64 t1 = cv::getTickCount();
        secsGPU = (t1 - t0) / cv::getTickFrequency() / (float)factor;
        cout << "Download from GPU needs:" << secsGPU << endl;
		probs = detect->afterTranspone(&test);
		boxes = get_detections(sourceclone, 845, thresh, detect->boxes, probs, 1);

		for (sd::ImageBox box : boxes) {
			rectangle(sourceclone, Rect(box.x, box.y, box.width, box.height), Scalar(255, 0, 0), 3);
		}
       
		//dstGP.copyTo(dstGPU);
	
		//
		imshow("HelloWorld", sourceclone);
		waitKey(1);
	}
    return 1;
}


