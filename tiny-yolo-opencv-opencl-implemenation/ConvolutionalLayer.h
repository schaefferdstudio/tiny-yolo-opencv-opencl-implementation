#ifndef ConvolutionalLayer_h
#define ConvolutionalLayer_h

#include "Layer.h"
#define ACTIVATION_LINEAR 1
namespace sd {
	class ConvolutionalLayer : public Layer
	{
	public:
		


		ConvolutionalLayer(int imageSize, int filterDepth, int prevImageDepth, int strideSize);
		ConvolutionalLayer(int imageSize, int filterDepth, int prevImageDepth, int strideSize,  int activation);
		~ConvolutionalLayer();
		virtual cv::Mat runOnCPU(cv::Mat* im) override;
		virtual cv::UMat runOnGPU(cv::UMat* im) override;
		virtual int init(FILE *fp) override;

	private:
		void  fastNormalizeBiasCPU(cv::Mat* img);
		void  normalizeBiasCPU(cv::Mat* img);
		cv::Mat biasScaleMeanVariance;
		cv::Mat fastBiasScaleMeanVariance;
		cv::Mat kernel;
		int activation = 0;
		int strideSize;
		int strideFromMiddle;
		// GPU
		int initGPU();
		cv::UMat kernelGPU;
		cv::UMat gpuDst;
		cv::UMat fastBiasScaleMeanVarianceGPU;
		void  fastNormalizeBiasGPU(cv::UMat* img);
		cv::ocl::Kernel kernelNormalize;
	};
}

#endif

