#pragma once

#include "vec3.h"

class ray
{
public:
	ray() {}
	ray(const vec3& origin, const vec3& direction) { o = origin; d = direction; }
	vec3 origin() const { return o; }
	vec3 direction() const { return d; }
	vec3 point_at_parameter(const float& t) const { return o + t*d; }

	vec3 o;
	vec3 d;
};

