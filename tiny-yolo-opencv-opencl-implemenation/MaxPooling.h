#ifndef MaxPooling_h
#define MaxPooling_h
#include "Layer.h"
namespace sd {
	class MaxPooling : public Layer
	{
	public:
		MaxPooling(int imageSize, int filterDepth, int prevImageDepth, int stride);
		~MaxPooling();
		virtual cv::Mat runOnCPU(cv::Mat* im) override;
		virtual cv::UMat runOnGPU(cv::UMat* im) override;
		virtual int init(FILE *fp) override;
	private:
		// GPU
		bool firstInit;
		double min_val;
		int stride;
		cv::UMat gpuDst;
		int initGPU();
	};
}
#endif
