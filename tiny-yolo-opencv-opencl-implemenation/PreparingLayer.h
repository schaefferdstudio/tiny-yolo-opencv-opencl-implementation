#ifndef PreparingLayer_h
#define PreparingLayer_h
#include "Layer.h"
namespace sd {
    // This Layer prepares the image 
	class PreparingLayer : public Layer
	{
	public:
		PreparingLayer(int imageSize, int filterDepth, int prevImageDepth);
		~PreparingLayer();
		virtual cv::Mat runOnCPU(cv::Mat* im) override;
		virtual cv::UMat runOnGPU(cv::UMat* im) override;
		virtual int init(FILE *fp) override;
	private:
		// GPU
		bool firstInit;
		double min_val;
		cv::UMat gpuDst;
		int initGPU();
	};
}
#endif

