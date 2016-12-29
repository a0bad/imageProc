#ifndef STATICHEADMODEL_H
#define STATICHEADMODEL_H

#include <memory>
#include "Eigen\Dense"
//this model contains all static values for the head model (eye position in face, maybe approx to nose, 
class StaticHeadModel
{
public:
	StaticHeadModel();
	~StaticHeadModel();
private:
	Eigen::Vector3f rightEyeRelativePosition; //relative position of right eye to ...
	Eigen::Vector3f leftEyeRelativePosition;


};

#endif
