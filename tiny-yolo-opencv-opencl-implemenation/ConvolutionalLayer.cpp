#include "ConvolutionalLayer.h"

using namespace sd;
using namespace cv;

ConvolutionalLayer::ConvolutionalLayer(int imageSize, int filterDepth, int prevImageDepth , int strideSize) :Layer(imageSize, filterDepth, prevImageDepth)
{
	this->strideSize = strideSize;
}

ConvolutionalLayer::ConvolutionalLayer(int imageSize, int filterDepth, int prevImageDepth, int activation, int strideSize) : Layer(imageSize, filterDepth, prevImageDepth)
{
	this->activation = activation;
	this->strideSize = strideSize;
}


ConvolutionalLayer::~ConvolutionalLayer()
{
}


int ConvolutionalLayer::init(FILE *fp) {
	int returnCode = 0;
	strideFromMiddle = floor(strideSize / 2);
	
	if (activation != 1) {
		biasScaleMeanVariance = Utils::readFromFile<float>(cv::Size(filterDepth, 4), CV_32FC1, fp);
		fastBiasScaleMeanVariance = Utils::createfastNormalizeMat(biasScaleMeanVariance);
	}
	else {
		biasScaleMeanVariance = Utils::readFromFile<float>(cv::Size(filterDepth, 1), CV_32FC1, fp);
		
	}
	kernel = Utils::readFromFile<float>(cv::Size(strideSize*strideSize*prevImageDepth, filterDepth), CV_32FC1, fp);
	initGPU();

	return returnCode;
}


int ConvolutionalLayer::initGPU() {
	// GPU
	kernel.copyTo(kernelGPU);
    //kernelGPU = kernel.getUMat(ACCESS_RW);
	std::ifstream ifs("conv.cl");
	std::string kernelSource((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
	cv::ocl::ProgramSource programSource(kernelSource);
	cv::ocl::Device dev = cv::ocl::Device::getDefault();
	cv::UMat inImage = cv::Mat(cv::Size(imageSize*imageSize*prevImageDepth, filterDepth), CV_32FC1).getUMat(ACCESS_RW);
	type = inImage.type();
	depth = CV_MAT_DEPTH(type);
	cn = CV_MAT_CN(type);
	kercn = cv::ocl::predictOptimalVectorWidth(inImage, inImage, inImage);
	ktype = CV_MAKE_TYPE(depth, kercn);
	cv::ocl::Kernel ke("conv", programSource,
		cv::format("-D %s -D T=%s -D T1=%s -D STRIDE_SIZE=%d%s", "THRESH_BINARY",
			cv::ocl::typeToStr(ktype), cv::ocl::typeToStr(depth), 1,
			""));
	k = ke;

	const double min_vals[] = { 0, CHAR_MIN, 0, SHRT_MIN, INT_MIN, -FLT_MAX, -DBL_MAX, 0 };
	min_val = min_vals[CV_32F];

	gpuDst = UMat(Size(imageSize*imageSize, filterDepth), CV_32F);

	if (activation != 1) {
		fastBiasScaleMeanVariance.copyTo(fastBiasScaleMeanVarianceGPU);

		std::ifstream ifsNorm("fastNormalize.cl");
		std::string kernelSourceNorm((std::istreambuf_iterator<char>(ifsNorm)), std::istreambuf_iterator<char>());
		cv::ocl::ProgramSource programSourceNorm(kernelSourceNorm);
		cv::ocl::Kernel keNorm("fastNorm", programSourceNorm,
			cv::format("-D T=%s -D T1=%s -D STRIDE_SIZE=%d%s",
				cv::ocl::typeToStr(ktype), cv::ocl::typeToStr(depth), 1,
				""));

		kernelNormalize = keNorm;
	}
	else {
		biasScaleMeanVariance.copyTo(fastBiasScaleMeanVarianceGPU);

		std::ifstream ifsNorm("biasActivation.cl");
		std::string kernelSourceNorm((std::istreambuf_iterator<char>(ifsNorm)), std::istreambuf_iterator<char>());
		cv::ocl::ProgramSource programSourceNorm(kernelSourceNorm);
		cv::ocl::Kernel keNorm("biasact", programSourceNorm,
			cv::format("-D T=%s -D T1=%s -D STRIDE_SIZE=%d%s",
				cv::ocl::typeToStr(ktype), cv::ocl::typeToStr(depth), 1,
				""));

		kernelNormalize = keNorm;
	}
	return 1;
}


void  ConvolutionalLayer::normalizeBiasCPU(cv::Mat* img) {
	for (int gy = 0; gy < img->rows; gy++) {
		for (int gx = 0; gx < img->cols; gx++) {
			cv::Point p(gx, gy);
			img->at<float>(p) = (img->at<float>(p) - biasScaleMeanVariance.at<float>(cv::Point(gy, MEAN))) / (sqrt(biasScaleMeanVariance.at<float>(cv::Point(gy, VARIANCE))) + .000001f);
			img->at<float>(p) *= biasScaleMeanVariance.at<float>(cv::Point(gy, SCALE));
			img->at<float>(p) += biasScaleMeanVariance.at<float>(cv::Point(gy, BIAS));
		}
	}
}


void  ConvolutionalLayer::fastNormalizeBiasCPU(cv::Mat* img) {
	for (int gy = 0; gy < img->rows; gy++) {
		for (int gx = 0; gx < img->cols; gx++) {
			img->at<float>(gy, gx) *= fastBiasScaleMeanVariance.at<float>(cv::Point(gy, BIAS));
			//std::cout << img->at<float>(gy, gx) << std::endl;
			img->at<float>(gy, gx) += fastBiasScaleMeanVariance.at<float>(cv::Point(gy, SCALE));
			if (activation == 0)
				img->at<float>(gy, gx) = img->at<float>(gy, gx) > 0 ? img->at<float>(gy, gx) : 0.1*img->at<float>(gy, gx);
			//std::cout << img->at<float>(gy, gx) << std::endl;
		}
	}
}

void ConvolutionalLayer::fastNormalizeBiasGPU(cv::UMat* img) {
	kernelNormalize.args(ocl::KernelArg::ReadOnlyNoSize(fastBiasScaleMeanVarianceGPU), ocl::KernelArg::ReadWrite(*img, cn, kercn));
	size_t globalsize[2] = { (size_t)img->cols * cn / kercn, (size_t)img->rows };
	globalsize[1] = (globalsize[1] + 1 - 1) / 1;
	bool returnVal = kernelNormalize.run(2, globalsize, NULL, false);
}


Mat ConvolutionalLayer::runOnCPU(Mat* im) {
	Mat dst = Mat(Size(im->size().width, kernel.size().height), CV_32F);
	int widthSingleRow = std::sqrt(im->size().width);
	for (int gy = 0; gy < dst.rows; gy++) {
		for (int gx = 0; gx < dst.cols; gx++) {
			int actualY = floor(gx / widthSingleRow);
			int actualX = gx - actualY*widthSingleRow;
			float sum = 0.0f;
			for (int ly = -strideFromMiddle; ly < strideFromMiddle+1; ly++) {
				if (actualY + ly > -1 && actualY + ly < widthSingleRow) {
					for (int lx = -strideFromMiddle; lx < strideFromMiddle+1; lx++) {
						if (actualX + lx > -1 && actualX + lx < widthSingleRow) {
							for (int actChannel = 0; actChannel < im->rows; actChannel++) {
								Point p((lx + strideFromMiddle) + (ly + strideFromMiddle) * strideSize + (actChannel * strideSize*strideSize), gy);
								float layerVal = kernel.at<float>(p);
								float imVal = im->at<float>(Point(actualX + lx + (actualY + ly) * widthSingleRow, actChannel));
								sum += (layerVal * imVal);
							}
						}
					}
				}
			}
			dst.at<float>(Point(gx, gy)) = sum;
		}
	}
	if (activation != 1) {
		fastNormalizeBiasCPU(&dst);
	}
	else {
		for (int gy = 0; gy < dst.rows; gy++) {
			for (int gx = 0; gx < dst.cols; gx++) {
				cv::Point p(gx, gy);
				dst.at<float>(p) += biasScaleMeanVariance.at<float>(cv::Point(gy, BIAS));
			}
		}
	}
	return dst;
}



UMat ConvolutionalLayer::runOnGPU(UMat* im) {
	k.args(ocl::KernelArg::ReadOnlyNoSize(*im), ocl::KernelArg::ReadOnlyNoSize(kernelGPU), ocl::KernelArg::WriteOnly(gpuDst, cn, kercn),
		ocl::KernelArg::Constant(Mat(1, 1, depth, Scalar::all(prevImageDepth))),
		ocl::KernelArg::Constant(Mat(1, 1, depth, Scalar::all(imageSize))),
		ocl::KernelArg::Constant(Mat(1, 1, depth, Scalar::all(strideSize))),
		ocl::KernelArg::Constant(Mat(1, 1, depth, Scalar::all(strideFromMiddle))));

	size_t globalsize[2] = { (size_t)gpuDst.cols * cn / kercn, (size_t)gpuDst.rows };
	globalsize[1] = (globalsize[1] + 1 - 1) / 1;
	bool returnVal = k.run(2, globalsize, NULL, false);
	fastNormalizeBiasGPU(&gpuDst);
    
	return gpuDst;
}
