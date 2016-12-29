#ifndef EYETRACKERKERNELHANDLER_H
#define EYETRACKERKERNELHANDLER_H

#include "Texture.h"
#include <memory>



class EyeTrackerKernelHandler
{
public:

	
	EyeTrackerKernelHandler();
	~EyeTrackerKernelHandler();
	
	static void apply3x3MatrixOnImage(std::shared_ptr<Texture<int> > image, float* mat);
	static void apply3x3MatrixOnImageDev(int* devImageIn, int* devImageOut, float* devMat, unsigned int w, unsigned int h);
	static void TextureToArray(std::shared_ptr<Texture<int> > input, int* out);
	static void TextureToArray(std::shared_ptr<Texture<int> > input, float* out);
	static void TextureToArray(std::shared_ptr<Texture<float> > input, float* out);
	static void ArrayToTexture(int* in, std::shared_ptr<Texture<int> > out);
	static void ArrayToTexture(float* in, std::shared_ptr<Texture<int> > out);
	static void ArrayToTexture(float* in, std::shared_ptr<Texture<float> > out);
	static void edgeDirectionDev(int* devImage_x, int* devImage_y, float* devOutput, unsigned int w, unsigned int h);
	static void edgeDirection(std::shared_ptr<Texture<int> > img_x, std::shared_ptr<Texture<int> > img_y, std::shared_ptr<Texture<float> > output);
	static std::shared_ptr<Texture<int>> performCannyEdgeDetection(std::shared_ptr<Texture<int>>& texture);
	static void testGaussian(std::shared_ptr<Texture<int> > img);
	static void testCircles(std::shared_ptr<Texture<int> > image,unsigned int radius);
	static void performCircleDetection(std::shared_ptr<Texture<int> > input, std::shared_ptr<Texture<int>> output);
	//static std::vector<std::tuple<std::tuple<std::tuple<int, int, >, float>>, float> detectCirclesWithProbability(std::shared_ptr<Texture<int>>& img);
	//static float testRoundKernel(float value);
};

#endif