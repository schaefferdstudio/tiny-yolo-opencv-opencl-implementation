#ifndef Layer_h
#define Layer_h
#include "opencv2/core/ocl.hpp"
#include <cstdint>
#include "Utils.h"
namespace sd {
	class Layer
	{
	public:
		Layer(int imageSize, int filterDepth, int prevImageDepth) {
			this->filterDepth = filterDepth;
			this->imageSize = imageSize;
			this->prevImageDepth = prevImageDepth;
		};
		virtual ~Layer() = default;
		virtual cv::Mat runOnCPU(cv::Mat* im) = 0;
		virtual cv::UMat runOnGPU(cv::UMat* im) = 0;
		virtual int init(FILE *fp) = 0;
	protected:
		virtual int initGPU() = 0;
		int imageSize;
		int filterDepth;
		int prevImageDepth;
		cv::ocl::Kernel k;
		int type;
		int depth;
		int cn;
		int kercn;
		int ktype;
		bool firstInit;
		double min_val;
		cv::UMat gpuDst;	

	};
}
#endif
