#pragma once
#include <cuda.h>
#include <cuda_runtime_api.h>

__global__ void kernelApply3x3MatrixOnImage(unsigned int* input, unsigned int* output, unsigned int* mat, unsigned int width, unsigned int height)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ unsigned int s_mat[9];
	if (x >= width || y >= height)
		return;
	if (threadIdx.x == 0 && threadIdx.y == 0){
		s_mat[0] = mat[0];
		s_mat[1] = mat[1];
		s_mat[2] = mat[2];
		s_mat[3] = mat[3];
		s_mat[4] = mat[4];
		s_mat[5] = mat[5];
		s_mat[6] = mat[6];
		s_mat[7] = mat[7];
		s_mat[8] = mat[8];
	}
	unsigned int pos = x + y*width;
	unsigned int val = input[pos];
	__syncthreads();
	//numbers dictate the center of the matrix in coords relative to midpoint. each thread writes "his value" multiplied with factor in the realtive centerpoint of the matrix
	
	if (y > 0)
	{	
		//-1 -1
		if (x > 0)
			atomicAdd(&output[pos - width - 1], val * s_mat[8]);
		//0 -1
		atomicAdd(&output[pos - width], val * s_mat[7]);
		//1 -1
		if (x < width - 1)
			atomicAdd(&output[pos - width + 1], val * s_mat[6]);
	}
	//-1 0
	if (x > 0)
		atomicAdd(&output[pos - 1], val * s_mat[5]);
	//0 0
	atomicAdd(&output[pos], val * s_mat[4]);
	//1 0
	if (x < width - 1)
		atomicAdd(&output[pos + 1], val * s_mat[3]);

	if (y < height - 1)
	{
		//-1 1
		if (x > 0)
			atomicAdd(&output[pos + width - 1], val * s_mat[2]);
		//0 1
		atomicAdd(&output[pos + width], val * s_mat[1]);
		//1 1
		if (x < width - 1)
			atomicAdd(&output[pos + width + 1], val * s_mat[0]);
	}
		
}