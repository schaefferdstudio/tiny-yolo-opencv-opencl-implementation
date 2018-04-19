

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

__kernel void conv(__global const uchar * srcptr, int src_step, int src_offset, __global const uchar* kernelptr, int kernel_step, int kernel_offset,
	__global uchar * dstptr, int dst_step, int dst_offset, int rows, int cols,
	T1 src_rows, T1 widthSingleRow, T1 strideSize, T1 strideFromMiddle)
{
	int gx = get_global_id(0);
	int gy = get_global_id(1) ;

	if (gx < cols)
	{
		if (gy < rows) {


			//int actualY = floor(gx / widthSingleRow);
			//int actualX = gx - actualY*widthSingleRow;
			//float sum = 0.0f;
			//for (int ly = -1; ly < 2; ly++) {
			//	if (actualY + ly > -1 && actualY + ly < widthSingleRow) {
			//		for (int lx = -1; lx < 2; lx++) {
			//			if (actualX + lx > -1 && actualX + lx < widthSingleRow) {
			//				for (int actChannel = 0; actChannel < src_rows; actChannel++) {
			//					int kernel_index = mad24(gy, kernel_step, mad24((lx + 1) + (ly + 1) * 3 + (actChannel * 9), (int)sizeof(T), kernel_offset));
			//					int src_index = mad24(actChannel, src_step, mad24(actualX + lx + (actualY + ly) * widthSingleRow, (int)sizeof(T), src_offset));
			//					T skernel = *(__global const T *)(kernelptr + kernel_index);
			//					T src = *(__global const T *)(srcptr + src_index);
			//					sum += skernel * src;
			//				}
			//			}
			//		}
			//	}
			//}
		


			int actualY = floor((gx + 0.01f) / widthSingleRow);
			int actualX = gx - actualY*widthSingleRow;
			float sum = 0.0f;
			for (int ly = -strideFromMiddle; ly < strideFromMiddle+1; ly++) {
				if (actualY + ly > -1 && actualY + ly < widthSingleRow) {
					for (int lx = -strideFromMiddle; lx < strideFromMiddle+1; lx++) {
						if (actualX + lx > -1 && actualX + lx < widthSingleRow) {
							for (int actChannel = 0; actChannel < src_rows; actChannel++) {
								int kernel_index = mad24(gy, kernel_step, mad24((lx + strideFromMiddle) + (ly + strideFromMiddle) * strideSize + (actChannel * strideSize*strideSize), (int)sizeof(T), kernel_offset));
								T skernel = *(__global const T *)(kernelptr + kernel_index);
								int src_index = mad24(actChannel, src_step, mad24(actualX + lx + (actualY + ly) * widthSingleRow, (int)sizeof(T), src_offset));
								T src = *(__global const T *)(srcptr + src_index);
								sum += skernel * src;
							}
						}
					}
				}
			}
			int dst_index = mad24(gy, dst_step, mad24(gx, (int)sizeof(T), dst_offset));
			__global float * dst = (__global float *)(dstptr + dst_index);
			dst[0] = sum;
		}
	}
}
