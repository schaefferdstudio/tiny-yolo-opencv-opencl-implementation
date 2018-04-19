#include "MaxPooling.h"

using namespace cv;
using namespace sd;
MaxPooling::MaxPooling(int imageSize, int filterDepth, int prevImageDepth, int stride) :Layer(imageSize, filterDepth, prevImageDepth)
{
	this->stride = stride;
}


MaxPooling::~MaxPooling()
{
}


int MaxPooling::init(FILE *fp) {
	int returnCode = 0;
	initGPU();
	return returnCode;
}



int MaxPooling::initGPU() {
	// GPU
	std::ifstream ifs("max.cl");
	std::string kernelSource((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
	cv::ocl::ProgramSource programSource(kernelSource);
	cv::UMat inImage = cv::UMat(cv::Size(imageSize*imageSize*prevImageDepth, filterDepth), CV_32FC1);
	type = inImage.type();
	depth = CV_MAT_DEPTH(type);
	cn = CV_MAT_CN(type);
	kercn = cv::ocl::predictOptimalVectorWidth(inImage, inImage, inImage);
	ktype = CV_MAKE_TYPE(depth, kercn);
	cv::ocl::Kernel ke("maxPool", programSource,
		cv::format("-D %s -D T=%s -D T1=%s -D STRIDE_SIZE=%d%s", "THRESH_BINARY",
			cv::ocl::typeToStr(ktype), cv::ocl::typeToStr(depth), 1,
			""));
	k = ke;

	const double min_vals[] = { 0, CHAR_MIN, 0, SHRT_MIN, INT_MIN, -FLT_MAX, -DBL_MAX, 0 };
	min_val = min_vals[CV_32F];

	gpuDst = UMat(Size(inImage.cols / (stride * stride), filterDepth), CV_32F);
	return 1;
}

cv::Mat MaxPooling::runOnCPU(cv::Mat* im) {
	Mat dst = Mat(Size(im->size().width / (stride*stride), im->size().height), CV_32F);
	int widthSingleRow = std::sqrt(dst.size().width);
	for (int gy = 0; gy < dst.rows; gy++) {
		for (int gx = 0; gx < dst.cols; gx++) {
			int actualY = floor(gx / widthSingleRow);
			int actualX = (gx - actualY*widthSingleRow);
			Point pFirstRow((actualX * stride) + (actualY * stride) * widthSingleRow*stride, gy);
			float maxVal = im->at<float>(pFirstRow);
			pFirstRow.x += 1;
			if (actualX + 1 < widthSingleRow*stride) {
				maxVal = maxVal >  im->at<float>(pFirstRow) ? maxVal : im->at<float>(pFirstRow);
			}
			
			if (actualY + 1 < widthSingleRow*stride) {
				Point pSecondRow((actualX * stride) + ((actualY * stride) + 1) * widthSingleRow * stride, gy);
				maxVal = maxVal > im->at<float>(pSecondRow) ? maxVal : im->at<float>(pSecondRow);
				pSecondRow.x += 1;
				if (actualX + 1 < widthSingleRow*stride) {
					maxVal = maxVal > im->at<float>(pSecondRow) ? maxVal : im->at<float>(pSecondRow);
				}
			}
			dst.at<float>(Point(gx, gy)) = maxVal;
		}
	}
	return dst;
}

cv::UMat MaxPooling::runOnGPU(cv::UMat* im) {
	k.args(ocl::KernelArg::ReadOnlyNoSize(*im), ocl::KernelArg::WriteOnly(gpuDst, cn, kercn), 
		ocl::KernelArg::Constant(Mat(1, 1, depth, Scalar::all((float)imageSize/ stride))),
		ocl::KernelArg::Constant(Mat(1, 1, depth, Scalar::all(stride))));
	size_t globalsize[2] = { (size_t)gpuDst.cols * cn / kercn, (size_t)gpuDst.rows };
	globalsize[1] = (globalsize[1] + 1 - 1) / 1;
	bool returnVal = k.run(2, globalsize, NULL, false);
	return gpuDst;
}
