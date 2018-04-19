#ifndef DetectionLayer_h
#define DetectionLayer_h

#include "Layer.h"
#include <vector>
namespace sd {
	class DetectionLayer : public Layer
	{
	public:
		DetectionLayer(int imageSize, int filterDepth, int prevImageDepth, float thresh);
		~DetectionLayer();
		virtual cv::Mat runOnCPU(cv::Mat* im) override;
		virtual cv::UMat runOnGPU(cv::UMat* im) override;
		virtual int init(FILE *fp) override;
	
		Box get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h);
		std::vector<Box>  get_region_boxes(float* predictions, int w, int h, int nLayer, int classesLayer, float thresh, cv::Mat* probs);
		std::vector<Box> boxes;
        cv::Mat afterTranspone(cv::Mat* im);
        cv::UMat gpuDst;
	private:
		// GPU
        cl_int ret;
        float *C;
        cl_command_queue command_queue;
        cl_mem c_mem;
		bool firstInit;
		double min_val;
		cv::Mat biasDetection;
        float thresh;
		int initGPU();
		void softmax(float *inout, int n);

	};
}

#endif
