#include "DetectionLayer.h"

using namespace cv;
using namespace sd;
DetectionLayer::DetectionLayer(int imageSize, int filterDepth, int prevImageDepth, float thresh) :Layer(imageSize, filterDepth, prevImageDepth)
{
    this->thresh = thresh;
}


DetectionLayer::~DetectionLayer()
{
}


int DetectionLayer::init(FILE *fp) {
	int returnCode = 0;
	biasDetection = Mat(Size(10, 1), CV_32FC1);
	biasDetection.at<float>(0, 0) = 1.08;
	biasDetection.at<float>(0, 1) = 1.19;
	biasDetection.at<float>(0, 2) = 3.42;
	biasDetection.at<float>(0, 3) = 4.41;
	biasDetection.at<float>(0, 4) = 6.63;
	biasDetection.at<float>(0, 5) = 11.38;
	biasDetection.at<float>(0, 6) = 9.42;
	biasDetection.at<float>(0, 7) = 5.11;
	biasDetection.at<float>(0, 8) = 16.62;
	biasDetection.at<float>(0, 9) = 10.52;
	initGPU();
	return returnCode;
}



int DetectionLayer::initGPU() {
	// GPU
	//std::ifstream ifs("max.cl");
	//std::string kernelSource((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
	//cv::ocl::ProgramSource programSource(kernelSource);
	cv::UMat inImage = cv::UMat(cv::Size(imageSize*imageSize*prevImageDepth, filterDepth), CV_32FC1);
	//type = inImage.type();
	//depth = CV_MAT_DEPTH(type);
	//cn = CV_MAT_CN(type);
	//kercn = cv::ocl::predictOptimalVectorWidth(inImage, inImage, inImage);
	//ktype = CV_MAKE_TYPE(depth, kercn);
	//cv::ocl::Kernel ke("maxPool", programSource,
	//	cv::format("-D %s -D T=%s -D T1=%s -D STRIDE_SIZE=%d%s", "THRESH_BINARY",
	//		cv::ocl::typeToStr(ktype), cv::ocl::typeToStr(depth), 1,
	//		""));
	//k = ke;

	//const double min_vals[] = { 0, CHAR_MIN, 0, SHRT_MIN, INT_MIN, -FLT_MAX, -DBL_MAX, 0 };
	//min_val = min_vals[CV_32F];
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1,
                         &device_id, &ret_num_devices);
    
    // Create an OpenCL context
    cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
    
    c_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                      imageSize*imageSize* filterDepth* sizeof(float), NULL, &ret);
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    C = (float*)malloc(imageSize*imageSize* filterDepth* sizeof(float));
    //ocl::convertFromBuffer(c_mem, inImage.step[0], filterDepth, imageSize*imageSize, CV_32F, gpuDst);
    gpuDst = UMat(Size(filterDepth, imageSize*imageSize), CV_32F);
	return 1;
}

Box DetectionLayer::get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h)
{
	Box b;
	b.x = (i + 1. / (1. + exp(-(x[index + 0])))) / w;
	b.y = (j + 1. / (1. + exp(-(x[index + 1])))) / h;
	b.w = exp(x[index + 2]) * biases[2 * n];
	b.h = exp(x[index + 3]) * biases[2 * n + 1];
	//if (DOABS) {
	b.w = exp(x[index + 2]) * biases[2 * n] / w;
	b.h = exp(x[index + 3]) * biases[2 * n + 1] / h;
	//}
	return b;
}


std::vector<Box>  DetectionLayer::get_region_boxes(float* predictions, int w, int h, int nLayer, int classesLayer, float thresh, cv::Mat* probs)
{
	std::vector<Box> boxes;
	int i, j, n;
	float* bias = biasDetection.ptr<float>();
	for (i = 0; i < w*h; ++i) {
		int row = i / w;
		int col = i % w;
		for (n = 0; n < nLayer; ++n) {
			int index = i*nLayer + n;
			int p_index = index * (classesLayer + 5) + 4;
			float scale = predictions[p_index];
			//if (l.classfix == -1 && scale < .5) scale = 0;
			int box_index = index * (classesLayer + 5);
			boxes.push_back(get_region_box(predictions, bias, n, box_index, col, row, w, h));

			int class_index = index * (classesLayer + 5) + 5;
		
			for (j = 0; j < classesLayer; ++j) {
				float prob = scale*predictions[class_index + j];
				probs->at<float>(Point(j, index)) = (prob > thresh) ? prob : 0;
			}
		}
	}
	return boxes;
}


void DetectionLayer::softmax(float *inout, int n) {
	int i;
	float sum = 0;
	float largest = -FLT_MAX;
	for (i = 0; i < n; ++i) {
		if (inout[i] > largest) largest = inout[i];
	}
	for (i = 0; i < n; ++i) {
		float e = exp(inout[i] - largest);
		sum += e;
		inout[i] = e;
	}
	for (i = 0; i < n; ++i) {
		inout[i] /= sum;
	}
}

cv::Mat DetectionLayer::afterTranspone(cv::Mat* im) {
	float* p = im->ptr<float>();
	for (int i = 0; i < 845; i++) {
		int index = 6 * i;
		p[index + 4] = 1. / (1. + exp(-p[index + 4]));
		float* poin = p + index + 5;
		softmax(poin, 1);
	}

	Mat probs = Mat(Size(1, 845), CV_32F);
	boxes = get_region_boxes(p, 13, 13, 5, 1, thresh, &probs);
	return probs;
}


cv::Mat DetectionLayer::runOnCPU(cv::Mat* im) {
	Mat dst = Mat(Size(im->size().height, im->size().width), CV_32F);
	cv::transpose(*im, dst);
	
	Mat probs = afterTranspone(&dst);
	return dst;
}

cv::UMat DetectionLayer::runOnGPU(cv::UMat* im) {
	/*k.args(ocl::KernelArg::ReadOnlyNoSize(*im), ocl::KernelArg::WriteOnly(gpuDst, cn, kercn),
		ocl::KernelArg::Constant(Mat(1, 1, depth, Scalar::all((float)imageSize / stride))),
		ocl::KernelArg::Constant(Mat(1, 1, depth, Scalar::all(stride))));
	size_t globalsize[2] = { (size_t)gpuDst.cols * cn / kercn, (size_t)gpuDst.rows };
	globalsize[1] = (globalsize[1] + 1 - 1) / 1;
	bool returnVal = k.run(2, globalsize, NULL, false);*/
	//cv::transpose(*im, gpuDst);
    //gpuDst = *im;
    //gpuDst = Mat(Size(filterDepth, imageSize*imageSize), CV_32F).getUMat(ACC);
    
    
    //ret = clEnqueueReadBuffer(command_queue, c_mem, CL_TRUE, 0,
    //                          imageSize*imageSize* filterDepth* sizeof(float), C, 0, NULL, NULL);
    //et = clEnqueueReadBuffer(ocl::Context::getDefault(), gpuDst.data, )
    //ret = clEnqueueReadBuffer(Queue, c_mem_obj, CL_TRUE, 0,
    //                          sizeof(float)*gpuDst.cols * gpuDst.rows, C, 0, NULL, NULL);
	return gpuDst;
}
