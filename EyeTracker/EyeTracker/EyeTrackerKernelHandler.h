#ifndef EYETRACKERKERNELHANDLER_H
#define EYETRACKERKERNELHANDLER_H

#include "Texture.h"
#include <memory>



class EyeTrackerKernelHandler
{
public:

	
	EyeTrackerKernelHandler();
	~EyeTrackerKernelHandler();
	
	void apply3x3MatrixOnImage(std::shared_ptr<Texture<unsigned int> > image, float* mat);
	void TextureToArray(std::shared_ptr<Texture<unsigned int> > input, unsigned int* out);
	void ArrayToTexture(unsigned int* in, std::shared_ptr<Texture<unsigned int> > out);
};

#endif