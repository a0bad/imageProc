#include "EyeTrackerKernelHandler.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "kernels.cu"



EyeTrackerKernelHandler::EyeTrackerKernelHandler()
{
}


EyeTrackerKernelHandler::~EyeTrackerKernelHandler()
{
}




void EyeTrackerKernelHandler::apply3x3MatrixOnImage(std::shared_ptr<Texture<unsigned int> > image, float* mat){

	unsigned int* hostData = (unsigned int*)malloc(image->getH()*image->getW()*sizeof(unsigned int));

	unsigned int* devInData;
	unsigned int* devOutData;
	unsigned int* devMat;

	cudaMalloc((void**)&devInData, image->getH()*image->getW()*sizeof(unsigned int));
	cudaMalloc((void**)&devOutData, image->getH()*image->getW()*sizeof(unsigned int));
	cudaMalloc((void**)&devMat, 9 * sizeof(float));


	TextureToArray(image, hostData);
	cudaMemcpy(devInData, hostData, image->getH()*image->getW()*sizeof(unsigned int), cudaMemcpyHostToDevice);

	dim3 blockSize(32, 8);
	dim3 gridSize(image->getW() / blockSize.x + (image->getW() % blockSize.x == 0 ? 0 : 1), image->getH() / blockSize.y + (image->getH() % blockSize.y == 0 ? 0 : 1));

	kernelApply3x3MatrixOnImage << < gridSize, blockSize >> >(devInData, devOutData, devMat, image->getW(), image->getH());

	cudaMemcpy(hostData, devOutData, image->getH()*image->getW()*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	ArrayToTexture(hostData, image);

	free(&hostData);
	cudaFree(devInData);
	cudaFree(devOutData);
	cudaFree(devMat);
	//kernelApply3x3MatrixOnImage(T* input, T* output, T* mat, unsigned int width, unsigned int height)
}

//only write one color from the input Texture since we use gray images R=G=B
void EyeTrackerKernelHandler::TextureToArray(std::shared_ptr<Texture<unsigned int> > input, unsigned int* out)
{
	for (size_t i = 0; i < input->getH()*input->getW(); ++i)
	{
		out[i] = input->getR(i);
	}
}


//write gray image back
void EyeTrackerKernelHandler::ArrayToTexture(unsigned int* in, std::shared_ptr<Texture<unsigned int> > out)
{
	for (size_t i = 0; i < out->getH()*out->getW(); ++i)
	{
		out->setR(i, in[i]);
		out->setG(i, in[i]);
		out->setB(i, in[i]);
	}
}