#pragma once
#include "cuda.h"
#include "device_functions.h"
#include "stdio.h"
#include "defines.h"

__constant__ float pi = 3.14159265359f;
__constant__ float pi2 = 3.14159265359f/2;
__constant__ float pi4 = 3.14159265359f / 4;
__constant__ float pi34 = 3* 3.14159265359f / 4;
__constant__ float gaussianMatrix[49] = {	0.00000067, 0.00002292, 0.00019117, 0.00038771, 0.00019117, 0.00002292, 0.00000067,
											0.00002292, 0.00078634, 0.00655965, 0.01330373, 0.00655965, 0.00078633, 0.00002292,
											0.00019117, 0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965, 0.00019117,
											0.00038771, 0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373, 0.00038771,
											0.00019117, 0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965, 0.00019117,
											0.00002292, 0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633, 0.00002292,
											0.00000067, 0.00002292, 0.00019117, 0.00038771, 0.00019117, 0.00002292, 0.00000067 };

__global__ void kernelApply3x3MatrixOnImage(int* input, int* output, float* mat, unsigned int width, unsigned int height)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ float s_mat[9];
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
#ifdef DEBUG_OUTPUT
	if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0){
		printf("Applied Matrix:\n");
		printf("%g, ", mat[0]);
		printf("%g, ", mat[1]);
		printf("%g\n", mat[2]);
		printf("%g, ", mat[3]);
		printf("%g, ", mat[4]);
		printf("%g\n", mat[5]);
		printf("%g, ", mat[6]);
		printf("%g, ", mat[7]);
		printf("%g\n", mat[8]);
	}
#endif
	unsigned int pos = x + y*width;
	int val = input[pos];
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



//rounds input value to cloest value of pi/4,pi/2,3pi/4,0pi; expects input value between [0,pi]
__device__ void kernelRoundToClosestAngle(float input,float* res) {
	float  val[4] = { 0, pi / 4, pi / 2, 3 * pi / 4 };
	float diff = input; //initialize with diff to 0
	float closest = 0; //initial return value as closest is 0

	for (unsigned int i = 1; i < 4; ++i)
	{
		if (abs(input - val[i]) < diff)
		{
			closest = val[i];
			diff = abs(input - val[i]);
		}
		
	}
	res[0] =  closest;
	
}
//blurs an image input of dim w*h with gaussian blur and returns in output https://en.wikipedia.org/wiki/Gaussian_blur
__global__ void kernelGaussianBlur(float* input, float* output, unsigned int w, unsigned int h)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	__shared__ size_t maxpos;
	if (x >= w || y >= h)
		return;
	float result = 0;
	int pos = 0;
	if (threadIdx.x == 0 && threadIdx.y == 0){
		maxpos = w*h;
		
	}
	/*if (x == 0 && y == 0){
		printf("maxpos: %d\n", maxpos);
		printf("w:%d,h:%d\n", w, h);
	}*/
	__syncthreads();
	for (int j = -3; j < 4; ++j)//y-coord
		{
		for (int i = -3; i < 4; ++i){//x-coord
			pos = (x + i) + (y + j)*w;
			/*if (x == 100 && y == 100){
				printf("i:%d,j:%d\n", i, j);
				printf("pos: %d\n", pos);
			}*/
			if (pos < 0 || pos >= maxpos || x+i < 0 || x+i >= w || y+j <0 || y+j >=h){ //edge of the image
				/*if (x == 100 && y == 100){
					printf("continue\n");
				}*/
				continue;
			}
			/*if (x == 100 && y == 100)
				printf("old res: %f\n", result);*/
			result += gaussianMatrix[(i+3)+(j+3)*7]*input[pos];
			/*if (x == 100 && y == 100)
			{
				printf("MatrixPos:%d\n", (i + 3) + (j + 3) * 7);
				printf("%f * %f = %f\n", gaussianMatrix[(i + 3) + (j + 3) * 7], input[pos], input[pos] * gaussianMatrix[(i + 3) + (j + 3) * 7]);
				printf("new res: %f\n", result);
			}*/
		}
	}
	/*if (x == 100  && y ==100 )
		printf("result: %f\n", result);*/
	output[x + y*w] = result;
	/*if (x == 100 && y ==100)
		printf("output: %f\n", output[x + y*w]);*/

}

__global__ void kernelAtan2FixedAngles(int* input_x, int* input_y, float* output, unsigned int width, unsigned int height) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;
	float angle = atan2((float)input_x[x + y*width], (float)input_y[x + y*width]);
	kernelRoundToClosestAngle(angle, &(output[x + y*width]));
}
/*
__global__ void kernelRoundToClosestAngle(float input, float* res){
	kernelRoundToClosestAngle(input, res);
}*/
/*
	checks on x,y if value is above threshold. if above, we find all points which this point is part of a circle with radius r and add value to the scoring. we repeat this for different radi.
	all points in result array which have big enough score are considered as midpoint of circle
*/
__global__ void findCirclesWithRadius(float* input, unsigned int* midpointSum, float above_threshold, unsigned int r, unsigned int w, unsigned int h){
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= w || y >= h)
		return;

	if (input[x + y*w] < above_threshold){ //return if value is < threshold -> wont participate in any circle
		return;
	}
	//extern __shared__ unsigned int midpointSum[];//shared result array

	//for (unsigned int r = min_radius; r < max_radius; ++r)

		float r2 = r*r;
		for (int xoff = 0; xoff <= r; ++xoff) //go from 0 to r; a ist the x-offset-coord, we calculate the y offset coord with sqrt(r*r-a*a)
		{
			float yoff = sqrt(r2 - xoff*xoff);
			if (x == 20 && y == 20)
			{


			}
			if ((int)x + xoff < w && (int)x + xoff >= 0 && (int)y + yoff < h && (int)y + yoff >= 0){
			    //printf("1\n");
				//printf("1:x:%d; y:%d; xoff:%d; yoff:%d; r:%d adding to %d\n ", x, y, xoff, (int)yoff, r, x + xoff + (y + (int)yoff)*w);
				atomicAdd(&midpointSum[x + xoff + (y + (int)yoff)*w], 1);
			}
			if ((int)x - xoff < w && (int)x - xoff >= 0 && (int)y + yoff < h && (int)y + yoff >= 0){
				//printf("2\n");
				//printf("2:x:%d; y:%d; xoff:%d; yoff:%d; r:%d adding to %d\n ", x, y, xoff, (int)yoff, r, x + xoff + (y + (int)yoff)*w);
				atomicAdd(&midpointSum[x - xoff + (y + (int)yoff)*w], 1);
			}
			if ((int)x + xoff < w && (int)x + xoff >= 0 && (int)y - yoff < h && (int)y - yoff >= 0){
				//printf("3\n");
				//printf("3:x:%d; y:%d; xoff:%d; yoff:%d; r:%d adding to %d\n ", x, y, xoff, (int)yoff, r, x + xoff + (y + (int)yoff)*w);
				atomicAdd(&midpointSum[x + xoff + (y - (int)yoff)*w], 1);
			}
			if ((int)x - xoff < w && (int)x - xoff >= 0 && (int)y - yoff < h && (int)y - yoff >= 0){
				//printf("4\n");
				//printf("4:x:%d; y:%d; xoff:%d; yoff:%d; r:%d adding to %d\n ", x, y, xoff, (int)yoff, r, x + xoff + (y + (int)yoff)*w);
				atomicAdd(&midpointSum[x - xoff + (y - (int)yoff)*w], 1);
			}
		}
		/*
		if (midpointSum[x + y*w]>0)
		{
			rating[x + y*w] = 255;
			printf("(%d|%d)->midpointSum: %d rating:%f \n", x,y, midpointSum[x + y*w], rating[x + y*w]);
		}*/
		//if (midpointSum[x + y*w]>(r + 1)) //if a point is the midpoint for at least a half circle it should be considered as midpoint
			//rating[x + y*w] = midpointSum[x + y*w]*;
		//else
			//rating[x + y*w] = 0;
}

__global__ void circleMidpointAnalysis(float* rating, unsigned int* midpointSum,unsigned int used_radius, unsigned int w, unsigned int h){
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= w || y >= h)
		return;

	//if (midpointSum[x + y*w]>0)
	{
	//	rating[x + y*w] = 255;
	}

	if (midpointSum[x + y*w] > (1.7  * used_radius + 1)) //if a point is the midpoint for at least a half circle it should be considered as midpoint
		rating[x + y*w] = midpointSum[x + y*w];
	else
		rating[x + y*w] = 0;

}

//combines 2 images in image1
__global__ void kernelCombineImagesAndRemoveBelowThreshold(float* image1, float* image2, float threshold, unsigned int w, unsigned int h) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= w || y >= h)
		return;
	image1[x + y*w] += image2[x + y*w];
	if (image1[x + y*w] < threshold)
		image1[x + y*w] = 0;
}

