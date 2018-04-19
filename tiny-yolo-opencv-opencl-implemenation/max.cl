


#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

__kernel void maxPool(__global const uchar * srcptr, int src_step, int src_offset,
	__global uchar * dstptr, int dst_step, int dst_offset, int rows, int cols, T1 widthSingleRow, T1 stride )
{
	int gx = get_global_id(0);
	int gy = get_global_id(1);

	if (gx < cols)
	{
		if (gy < rows) {
			int actualY = floor((gx + 0.01f) / widthSingleRow);
			int actualX = gx - actualY*widthSingleRow;

			int src_index = mad24(gy, src_step, mad24((actualX * stride) + (actualY * stride) * widthSingleRow * stride, (int)sizeof(T), src_offset));
			__global float * s_kernel = (__global const T *)(srcptr + src_index);

			float maxVal = s_kernel[0];
			if (actualX + 1 < widthSingleRow*stride)
				maxVal = maxVal > s_kernel[1] ? maxVal : s_kernel[1];
			if (actualY + 1 < widthSingleRow*stride) {
				src_index = mad24(gy, src_step, mad24((actualX * stride) + ((actualY * stride) + 1) * widthSingleRow * stride, (int)sizeof(T), src_offset));
				s_kernel = (__global const T *)(srcptr + src_index);
				maxVal = maxVal > s_kernel[0] ? maxVal : s_kernel[0];
				if (actualX + 1 < widthSingleRow*stride)
					maxVal = maxVal > s_kernel[1] ? maxVal : s_kernel[1];
			}
			//src_index = mad24(gy, src_step, mad24(sFirstGx + 1, (int)sizeof(T), src_offset));
			//__global float * sTest_kernel = (__global const T *)(srcptr + src_index);
			//maxVal = maxVal >  sTest_kernel[0] ? maxVal : sTest_kernel[0];

			int dst_index = mad24(gy, dst_step, mad24(gx, (int)sizeof(T), dst_offset));
			__global float * dst = (__global float *)(dstptr + dst_index);
			dst[0] = maxVal;
		}
	}
}
