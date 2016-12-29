#pragma once
#include "EyeTrackerKernelHandler.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "kernels.cu"
#include "defines.h"


EyeTrackerKernelHandler::EyeTrackerKernelHandler()
{
}


EyeTrackerKernelHandler::~EyeTrackerKernelHandler()
{
}




void EyeTrackerKernelHandler::apply3x3MatrixOnImage(std::shared_ptr<Texture<int> > image, float* mat){

	int* hostData = (int*)malloc(image->getH()*image->getW()*sizeof(int));

	int* devInData;
	int* devOutData;
	float* devMat;

	cudaMalloc((void**)&devInData, image->getH()*image->getW()*sizeof(int));
	cudaMalloc((void**)&devOutData, image->getH()*image->getW()*sizeof(int));
	cudaMalloc((void**)&devMat, 9 * sizeof(float));

	
	TextureToArray(image, hostData);
	cudaMemcpy(devInData, hostData, image->getH()*image->getW()*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(devMat, mat, 9 * sizeof(float), cudaMemcpyHostToDevice);
	
	apply3x3MatrixOnImageDev(devInData, devOutData, devMat, image->getW(), image->getH());
	
	cudaMemcpy(hostData, devOutData, image->getH()*image->getW()*sizeof(int), cudaMemcpyDeviceToHost);
	ArrayToTexture(hostData, image);
	
	free(hostData);
	cudaFree(devInData);
	cudaFree(devOutData);
	cudaFree(devMat);
	
}
void EyeTrackerKernelHandler::apply3x3MatrixOnImageDev(int* devImageIn, int* devImageOut, float* devMat, unsigned int w, unsigned int h){
	dim3 blockSize(32, 8);
	dim3 gridSize(w / blockSize.x + (w % blockSize.x == 0 ? 0 : 1), h / blockSize.y + (h % blockSize.y == 0 ? 0 : 1));
	kernelApply3x3MatrixOnImage << < gridSize, blockSize >> >(devImageIn, devImageOut, devMat, w, h);

}

//only write one color from the input Texture since we use gray images R=G=B
void EyeTrackerKernelHandler::TextureToArray(std::shared_ptr<Texture<int> > input, int* out)
{
	for (size_t i = 0; i < input->getH()*input->getW(); ++i)
	{
		out[i] = input->getR(i);
	}
}

void EyeTrackerKernelHandler::TextureToArray(std::shared_ptr<Texture<int> > input, float* out)
{
	for (size_t i = 0; i < input->getH()*input->getW(); ++i)
	{
		out[i] = (float)input->getR(i);
	}
}


//write gray image back
void EyeTrackerKernelHandler::ArrayToTexture(int* in, std::shared_ptr<Texture<int> > out)
{
	for (size_t i = 0; i < out->getH()*out->getW(); ++i)
	{
		out->setR(i, abs(in[i]));
		out->setG(i, abs(in[i]));
		out->setB(i, abs(in[i]));
	}
}

void EyeTrackerKernelHandler::ArrayToTexture(float* in, std::shared_ptr<Texture<float> > out)
{
	for (size_t i = 0; i < out->getH()*out->getW(); ++i)
	{
		out->setR(i, abs(in[i]));
		out->setG(i, abs(in[i]));
		out->setB(i, abs(in[i]));
	}
}

void EyeTrackerKernelHandler::ArrayToTexture(float* in, std::shared_ptr<Texture<int> > out)
{
	for (size_t i = 0; i < out->getH()*out->getW(); ++i)
	{
		out->setR(i, (int)abs(in[i]));
		out->setG(i, (int)abs(in[i]));
		out->setB(i, (int)abs(in[i]));
	}
}


void EyeTrackerKernelHandler::edgeDirection(std::shared_ptr<Texture<int> > img_x, std::shared_ptr<Texture<int> > img_y, std::shared_ptr<Texture<float> > output){
	unsigned int w = img_x->getW();
	unsigned int h = img_x->getH();
	int* hostDataX = (int*)malloc(h*w*sizeof(int));
	int* hostDataY = (int*)malloc(h*w*sizeof(int));
	float* hostResult = (float*)malloc(h*w*sizeof(float));

	int* devInDataX;
	int* devInDataY;
	float* devOutData;

	cudaMalloc((void**)&devInDataX, h*w*sizeof(int));
	cudaMalloc((void**)&devInDataY, h*w*sizeof(int));
	cudaMalloc((void**)&devOutData, h*w*sizeof(float));


	
	TextureToArray(img_x, hostDataX);
	TextureToArray(img_y, hostDataY);
	cudaMemcpy(devInDataX, hostDataX, h*w*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(devInDataY, hostDataY, h*w*sizeof(int), cudaMemcpyHostToDevice);

	edgeDirectionDev(devInDataX, devInDataY, devOutData, w, h);

	cudaMemcpy(hostResult, devOutData, h*w*sizeof(int), cudaMemcpyDeviceToHost);
	ArrayToTexture(hostResult, output);
	
	free(hostDataX);
	free(hostDataY);
	cudaFree(devInDataX);
	cudaFree(devInDataY);
	cudaFree(devOutData);
}

void EyeTrackerKernelHandler::edgeDirectionDev(int* devImage_x, int* devImage_y, float* devOutput, unsigned int w, unsigned int h){
	dim3 blockSize(32, 8);
	dim3 gridSize(w / blockSize.x + (w % blockSize.x == 0 ? 0 : 1), h / blockSize.y + (h % blockSize.y == 0 ? 0 : 1));
	kernelAtan2FixedAngles << < gridSize, blockSize >> > (devImage_x, devImage_y, devOutput, w, h);
}





std::shared_ptr<Texture<int>> EyeTrackerKernelHandler::performCannyEdgeDetection(std::shared_ptr<Texture<int>>& texture){
	auto sobel_ver = std::shared_ptr<Texture<int>>(new Texture<int>(texture));
	auto sobel_hor = std::shared_ptr<Texture<int>>(new Texture<int>(texture));
	float mat_sobel_hor[9] = { -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0 };
	float mat_sobel_ver[9] = { 1.0, 2.0, 1.0, 0.0, 0.0, 0.0, -1.0, -2.0, -1.0 };
	apply3x3MatrixOnImage(sobel_ver, mat_sobel_ver);
	apply3x3MatrixOnImage(sobel_hor, mat_sobel_hor);
	
	//https://de.wikipedia.org/wiki/Canny-Algorithmus
	//Sobel Operator
	//Kantenrichtung a = atan2(gy,gx) (gxgy = Ergebnis Sobel) -> done
	//Kantenstärke G(x,y) = abs(gx(x,y))+ abs(gy(x,y))
	//Kanten entlang gehen, jeweils nur das maximum beibehalten. (rechts und links von pixel vergleichen und das maximum beibehalten). (Non Maximum Supression)
	//Hysterese: 
	std::shared_ptr<Texture<int>> output;
	return output;
}
/*
float EyeTrackerKernelHandler::testRoundKernel(float value)
{
	float* returnvalue;
	float result;
	cudaMalloc((void**)&returnvalue, sizeof(float));

	testRoundToClosestKernel << <1, 1 >> >(value,returnvalue);

	cudaMemcpy(&result, returnvalue, sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(returnvalue);

	return result;
}*/
/*
	take a circle template in different sizes, and move it over the edge image. calculate hit probability, and take set of best hits
	Circles we are searching for are circle shapes with nothing in the middle (pupil). surrounding pixels are not important
*/
//return value is: vector<<<x,y>radius>probability>
//std::vector<std::tuple<std::tuple<std::tuple<int, int, >, float>>, float> EyeTrackerKernelHandler::detectCirclesWithProbability(std::shared_ptr<Texture<int>>& img){

//}

void EyeTrackerKernelHandler::testGaussian(std::shared_ptr<Texture<int> > image) {
	float* hostData = (float*)malloc(image->getH()*image->getW()*sizeof(float));

	float* devInData;
	float* devOutData;
	

	cudaMalloc((void**)&devInData, image->getH()*image->getW()*sizeof(float));
	cudaMalloc((void**)&devOutData, image->getH()*image->getW()*sizeof(float));
	


	TextureToArray(image, hostData);
	cudaMemcpy(devInData, hostData, image->getH()*image->getW()*sizeof(float), cudaMemcpyHostToDevice);
	dim3 blockSize(32, 8);
	dim3 gridSize(image->getW() / blockSize.x + (image->getW() % blockSize.x == 0 ? 0 : 1), image->getH() / blockSize.y + (image->getH() % blockSize.y == 0 ? 0 : 1));


	kernelGaussianBlur << <gridSize, blockSize >> >(devInData, devOutData, image->getW(), image->getH());
	//apply3x3MatrixOnImageDev(devInData, devOutData, devMat, image->getW(), image->getH());

	float* hostData2 = (float*)malloc(image->getH()*image->getW()*sizeof(float));

	cudaMemcpy(hostData2, devOutData, image->getH()*image->getW()*sizeof(float), cudaMemcpyDeviceToHost);
	ArrayToTexture(hostData2, image);

	free(hostData);
	free(hostData2);
	cudaFree(devInData);
	cudaFree(devOutData);
}

void EyeTrackerKernelHandler::testCircles(std::shared_ptr<Texture<int> > image, unsigned int radius) {
	float* hostData = (float*)malloc(image->getH()*image->getW()*sizeof(float));

	float* devInData;
	float* devOutData;
	unsigned int* devMidData;

	cudaMalloc((void**)&devInData, image->getH()*image->getW()*sizeof(float));
	cudaMalloc((void**)&devOutData, image->getH()*image->getW()*sizeof(float));
	cudaMalloc((void**)&devMidData, image->getH()*image->getW()*sizeof(unsigned int));



	TextureToArray(image, hostData);
	cudaMemcpy(devInData, hostData, image->getH()*image->getW()*sizeof(float), cudaMemcpyHostToDevice);
	dim3 blockSize(32, 8);
	dim3 gridSize(image->getW() / blockSize.x + (image->getW() % blockSize.x == 0 ? 0 : 1), image->getH() / blockSize.y + (image->getH() % blockSize.y == 0 ? 0 : 1));
	cudaMemset(devMidData, 0, image->getH()*image->getW()*sizeof(unsigned int));
#ifdef DEBUG_OUTPUT
	std::cerr << "before circle Kernel\n";
	std::cerr << "rad= " << radius << std::endl;
#endif
	//std::cerr << "w: " << image->getW() << " h: " << image->getH() << std::endl;
	findCirclesWithRadius << <gridSize, blockSize >> >(devInData, devMidData, 30, radius, image->getW(), image->getH());
	circleMidpointAnalysis << <gridSize, blockSize >> >(devOutData, devMidData,radius, image->getW(), image->getH());
	//findCirclesWithRadius(float* input, float* rating, float above_threshold, unsigned int r, unsigned int w, unsigned int h)
	//std::cerr << "after circle Kernel\n";
	float* hostData2 = (float*)malloc(image->getH()*image->getW()*sizeof(float));

	cudaMemcpy(hostData2, devOutData, image->getH()*image->getW()*sizeof(float), cudaMemcpyDeviceToHost);
	ArrayToTexture(hostData2, image);

	free(hostData);
	free(hostData2);
	cudaFree(devInData);
	cudaFree(devOutData);
}
//input grey image, output image with circles added to image
void EyeTrackerKernelHandler::performCircleDetection(std::shared_ptr<Texture<int> > input, std::shared_ptr<Texture<int>> output){
	unsigned int w = input->getW();
	unsigned int h = input->getH();
	int* hostInData = (int*)malloc(h*w*sizeof(int));
	int* hostOutData = (int*)malloc(h*w*sizeof(int));
	int* hostDataX = (int*)malloc(h*w*sizeof(int));
	int* hostDataY = (int*)malloc(h*w*sizeof(int));

	int* devInData;
	int* devOutData;


	TextureToArray(input, hostInData);

	ArrayToTexture(hostOutData, output);
	//gauss-blur -> kernelGaussianBlur
	//edge-detection -> apply3x3MatrixOnImage (sobel)
	//combine x-y images -< todo impl for gpu
	//remove below threshold -< todo impl for gpu
	//find midpoints findCirclesWithRadius -> circleMidpointAnalysis




}