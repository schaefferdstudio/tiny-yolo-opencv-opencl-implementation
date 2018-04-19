#include "PreparingLayer.h"

using namespace cv;
using namespace sd;
PreparingLayer::PreparingLayer(int imageSize, int filterDepth, int prevImageDepth) :Layer(imageSize, filterDepth, prevImageDepth)
{
	
}


PreparingLayer::~PreparingLayer()
{
}


int PreparingLayer::init(FILE *fp) {
	int returnCode = 0;
	initGPU();
	return returnCode;
}



int PreparingLayer::initGPU() {
	// GPU
	std::ifstream ifs("prepare.cl");
	std::string kernelSource((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
	cv::ocl::ProgramSource programSource(kernelSource);
	cv::UMat inImage = cv::UMat(cv::Size(imageSize*imageSize, prevImageDepth), CV_32FC1);
	type = inImage.type();
	depth = CV_MAT_DEPTH(type);
	cn = CV_MAT_CN(type);
	kercn = cv::ocl::predictOptimalVectorWidth(inImage, inImage, inImage);
	ktype = CV_MAKE_TYPE(depth, kercn);
	cv::ocl::Kernel ke("prepare", programSource,
		cv::format("-D %s -D T=%s -D T1=%s -D STRIDE_SIZE=%d%s", "THRESH_BINARY",
			cv::ocl::typeToStr(ktype), cv::ocl::typeToStr(depth), 1,
			""));
	k = ke;

	const double min_vals[] = { 0, CHAR_MIN, 0, SHRT_MIN, INT_MIN, -FLT_MAX, -DBL_MAX, 0 };
	min_val = min_vals[CV_32F];

	gpuDst = UMat(cv::Size(imageSize*imageSize, prevImageDepth), CV_32F);
	return 1;
}

cv::Mat PreparingLayer::runOnCPU(cv::Mat* im) {
	Mat dst;
	im->copyTo(dst);
	dst /= 255.0;
	return dst;
}

cv::UMat PreparingLayer::runOnGPU(cv::UMat* im) {
	k.args(ocl::KernelArg::ReadWrite(*im, cn, kercn));
	size_t globalsize[2] = { (size_t)im->cols * cn / kercn, (size_t)im->rows };
	globalsize[1] = (globalsize[1] + 1 - 1) / 1;
	bool returnVal = k.run(2, globalsize, NULL, false);
	return *im;
}