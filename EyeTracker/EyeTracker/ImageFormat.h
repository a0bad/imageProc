#ifndef IMAGEFORMAT_H
#define IMAGEFORMAT_H

#include <vector>

struct colorRGB
{
	colorRGB(){}
	colorRGB(int _r,int _g,int _b) : r(_r), g(_g), b(_b){} 
	int r,g,b;
};

struct colorRGBf
{
	colorRGBf(){}
	colorRGBf(float _r, float _g, float _b) : r(_r), g(_g), b(_b){}
	float r,g,b;
};

struct imageRGB
{
	std::vector<colorRGB> _data;
};

struct imageRGBf
{
	std::vector<colorRGBf> _data;
};

#endif