

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif


__kernel void biasact(__global const uchar* kernelptr, int kernel_step, int kernel_offset,
	__global uchar * dstptr, int dst_step, int dst_offset, int rows, int cols)
{
	int gx = get_global_id(0);
	int gy = get_global_id(1);

	if (gx < cols)
	{
		if (gy < rows) {
			int kernel_index = mad24(0, kernel_step, mad24(gy, (int)sizeof(T), kernel_offset));
			T skernel = *(__global const T *)(kernelptr + kernel_index);
			int dst_index = mad24(gy, dst_step, mad24(gx, (int)sizeof(T), dst_offset));
			__global float * dst = (__global float *)(dstptr + dst_index);
			dst[0] += skernel;
		}
	}
}
