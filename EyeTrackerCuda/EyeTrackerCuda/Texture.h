#ifndef TEXTURE_H
#define TEXTURE_H

#include "FreeImage.h"
#include "imageFormat.h"
#include <fstream>
#include <string>
#include <memory>
#include <iostream>
#include <math.h>


const unsigned int COMBINE_ABOVE_THRESHOLD = 220;

template <typename T>
struct rgbT
{
	rgbT(){}
	rgbT(const T& _r,const T& _g,const T& _b) : r(_r), g(_g), b(_b) {}
	rgbT(const rgbT<T>& copy) { r=copy.r; g=copy.g; b = copy.b; }
	T r, g, b;
};


template <typename T>
class Texture 
{
public:
	Texture(const unsigned int& _size) : length(_size) { image_data.resize(length); }
	Texture(const unsigned int& _w, const unsigned int& _h) : length(_w*_h), w(_w), h(_h) { image_data.resize(length); }
	Texture(const std::string& filename);
	Texture(const Texture<T>& image) : length(image.size()), w(image.getW()), h(image.getH()) {
		image_data.resize(length);
		for (size_t i = 0; i < length; ++i)
		{
			image_data[i] = rgbT<T>(image.getR(i), image.getG(i), image.getB(i));
		}
	}

	Texture(const std::shared_ptr<Texture<T>>& image) : length(image->size()), w(image->getW()), h(image->getH()) {
		image_data.resize(length);
		for (size_t i = 0; i < length; ++i)
		{
			image_data[i] = rgbT<T>(image->getR(i), image->getG(i), image->getB(i));
		}
	}

	
	void combineWithImage(std::shared_ptr<Texture<T>>& other);
	void removeBelowThreshold(const T& threshold);
	void printYourselfRaw(const std::string& filename, size_t w, size_t h);
	void printYourselfImage(const std::string& filename, size_t w, size_t h);
	T getR(size_t pos) const {return image_data[pos].r;}
	T getG(size_t pos) const {return image_data[pos].g;}
	T getB(size_t pos) const {return image_data[pos].b;}
	const rgbT<T> getColor(size_t pos) const {return image_data[pos];}
	void setR(size_t pos, T value){image_data[pos].r = value;}
	void setG(size_t pos, T value){image_data[pos].g = value;}
	void setB(size_t pos, T value){image_data[pos].b = value;}
	void setColor(size_t pos, rgbT<T> color){ image_data[pos] = color; }
	void makeGrey();
	void edgeDetectionSobel();
	void applyMxNMatrix(const float* matrix, const size_t& M, const size_t& N);
	void addCircleToPicture(const unsigned int& x, const unsigned int& y, const unsigned int& r, const unsigned int& red, const unsigned int& green, const unsigned int& blue);
	unsigned int size() const { return length; }
	unsigned int getW() const { return w;}
	unsigned int getH() const { return h;}
private:
	unsigned int length;
	unsigned int w, h;
	std::vector<rgbT<T>> image_data;

};

template <typename T>
void Texture<T>::removeBelowThreshold(const T& threshold) {
	for (size_t i = 0; i < length; ++i)
	{
		if (image_data[i].r < threshold)
		{
			image_data[i].r = 0;
			image_data[i].g = 0;
			image_data[i].b = 0;
		}
	}
}

template <typename T>
void Texture<T>::combineWithImage(std::shared_ptr<Texture<T>>& other) {
	for (size_t i = 0; i < length; ++i)
	{
		
		image_data[i].r += other->getR(i);
		if (image_data[i].r <= COMBINE_ABOVE_THRESHOLD){
			image_data[i].r = 0;
			image_data[i].g = 0;
			image_data[i].b = 0;
		}
		else
		{
			image_data[i].g += other->getG(i);
			image_data[i].b += other->getB(i);
		}
		

	}
}

template <typename T>
void Texture<T>::printYourselfRaw(const std::string& filename, size_t w, size_t h)
{
	if (w*h != length)
	{
		std::cerr << "Error. Width*height does not equal size!\n";
		return;
	}

	std::fstream fstr;
	fstr.open(filename.c_str());
	if (!fstr.is_open())
	{
		std::cerr << "Error! Can't open File: "<< filename << "!\n";
	}
	else
	{
		for (size_t i = 0; i < length; ++i)
		{
			fstr << image_data[i].r;
			fstr << image_data[i].g;
			fstr << image_data[i].b;
		}
	}
	std::cout << "Succesfully printed in " << filename << "!\n";
}

template <typename T>
void Texture<T>::printYourselfImage(const std::string& filename, size_t w, size_t h)
{
		std::cout << "Printing as Image\n";
	
			FIBITMAP* fi = FreeImage_Allocate(w,h,24);
			RGBQUAD color;
			//std::cout << "_data size: " << image_data->_data.size() << std::endl;   
			if(fi != 0)
			{
				for(size_t i = 0 ; i < h; ++i)
					for(size_t j = 0 ; j < w; ++j)
						{
							//std::cout << "i*w+j: " << i*w+j << std::endl;
							color.rgbRed = image_data[i*w+j].r;
							color.rgbGreen = image_data[i*w+j].g;
							color.rgbBlue = image_data[i*w+j].b;
							
							//if(i ==256 && j == 256)
							//std::cout << "Texture.cpp: (R,G,B): " << (int)color.rgbRed << "," << (int)color.rgbGreen << "," << (int)color.rgbBlue << std::endl;
							FreeImage_SetPixelColor(fi,j,i,&color);
						}
					
				FreeImage_Save(FIF_TIFF,fi,filename.c_str());
				std::cout << "Saved file: " << filename << std::endl;
			}
}

template <typename T>
Texture<T>::Texture(const std::string& filename) {
	FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(filename.c_str(), 0);
	std::cout << "Trying to read: " << filename.c_str() << "!\n";
	if (fif != FIF_UNKNOWN)
	{
		FIBITMAP* fi = FreeImage_Load(fif, filename.c_str());
		if (fi != 0)
		{
			w = FreeImage_GetWidth(fi);
			h = FreeImage_GetHeight(fi);
			length = w*h;
			if (FreeImage_GetBPP(fi) != 32)
			{
				FIBITMAP* tmp = fi;
				fi = FreeImage_ConvertTo32Bits(tmp);
				FreeImage_Unload(tmp);
			}
			image_data.resize(w*h);
			//std::cout<< "1" << std::endl;

			auto data = FreeImage_GetBits(fi);
			//std::cout<< "2" << std::endl;
			for (size_t i = 0; i < w; ++i)
			for (size_t j = 0; j < h; ++j)
			{
				RGBQUAD color;
				FreeImage_GetPixelColor(fi, i, j, &color);
				//if(i ==256 && j == 256)
				unsigned char val_b = data[(i + j*w) * 4];
				unsigned char val_g = data[(i + j*w) * 4 + 1];
				unsigned char val_r = data[(i + j*w) * 4 + 2];
				image_data[i + j*w].r = val_r;
				image_data[i + j*w].g = val_g;
				image_data[i + j*w].b = val_b;
			}

			FreeImage_Unload(fi);
		}
	}
}


template <typename T>
void Texture<T>::makeGrey() {
	for (size_t i = 0; i < length; ++i)
	{
		unsigned char val = (unsigned char)( 0.299*image_data[i].r + 0.587*image_data[i].g + 0.114*image_data[i].b);
		image_data[i].r = val;
		image_data[i].g = val;
		image_data[i].b = val;
	}
}

template <typename T>
void Texture<T>::addCircleToPicture(const unsigned int& x, const unsigned int& y, const unsigned int& r, const unsigned int& red, const unsigned int& green, const unsigned int& blue){


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
			setR(x + xoff + (y + (int)yoff)*w, red);
			setG(x + xoff + (y + (int)yoff)*w, green);
			setB(x + xoff + (y + (int)yoff)*w, blue);
			
		}
		if ((int)x - xoff < w && (int)x - xoff >= 0 && (int)y + yoff < h && (int)y + yoff >= 0){
			//printf("2\n");
			//printf("2:x:%d; y:%d; xoff:%d; yoff:%d; r:%d adding to %d\n ", x, y, xoff, (int)yoff, r, x + xoff + (y + (int)yoff)*w);
			setR(x - xoff + (y + (int)yoff)*w, red);
			setG(x - xoff + (y + (int)yoff)*w, green);
			setB(x - xoff + (y + (int)yoff)*w, blue);
		}
		if ((int)x + xoff < w && (int)x + xoff >= 0 && (int)y - yoff < h && (int)y - yoff >= 0){
			//printf("3\n");
			//printf("3:x:%d; y:%d; xoff:%d; yoff:%d; r:%d adding to %d\n ", x, y, xoff, (int)yoff, r, x + xoff + (y + (int)yoff)*w);
			setR(x + xoff + (y - (int)yoff)*w, red);
			setG(x + xoff + (y - (int)yoff)*w, green);
			setB(x + xoff + (y - (int)yoff)*w, blue);
		}
		if ((int)x - xoff < w && (int)x - xoff >= 0 && (int)y - yoff < h && (int)y - yoff >= 0){
			//printf("4\n");
			//printf("4:x:%d; y:%d; xoff:%d; yoff:%d; r:%d adding to %d\n ", x, y, xoff, (int)yoff, r, x + xoff + (y + (int)yoff)*w);
			setR(x - xoff + (y - (int)yoff)*w, red);
			setG(x - xoff + (y - (int)yoff)*w, green);
			setB(x - xoff + (y - (int)yoff)*w, blue);
		}
	}
}

/**
apply a pixel wise filter matrix. result is per pixel sum of values multiplied with matrix.
f.e. 3x3 Matrix multiplies each value around a pixel with the corresponding value in the matrix and the result is the sum of all the results.
on the edges the data is expanded by the last value in the image
*/
/*template <typename T>
void Texture<T>::applyMxNMatrix(const float* matrix, const size_t& M, const size_t& N)
{
	std::vector<rgbT<T> > tmp(length);
	bool leftE = false, rightE = false, topE = false, botE = false;
	for (size_t i = 0; i < length; ++i)
	{
		float value = 0;
		for (size_t m = 0; m < M; ++m) //m = x-coord
		{

			if (m == 0)
				leftE = true;
			if (m == w - 1)
				rightE = true;

			for (size_t n = 0; n < N; ++n) //n = y-coord
			{
				if (n == 0)
					topE = true;
				if (n == h - 1)
					botE = true;
				value +=
			}
			leftE = false;
			rightE = false;
			topE = false;
			botE = false;
		}
	}
	imageData = tmp;

}
*/
template <typename T>
void Texture<T>::edgeDetectionSobel(){

}

#endif