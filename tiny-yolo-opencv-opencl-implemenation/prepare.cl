
// TODO: Add OpenCL kernel code here.


#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

__kernel void prepare(__global uchar * dstptr, int dst_step, int dst_offset, int rows, int cols)
{
	int gx = get_global_id(0);
	int gy = get_global_id(1);

	if (gx < cols)
	{
		if (gy < rows) {

			int dst_index = mad24(gy, dst_step, mad24(gx, (int)sizeof(T), dst_offset));
			__global float * dst = (__global float *)(dstptr + dst_index);
			dst[0] = dst[0] / 255.0;
		}
	}
}
