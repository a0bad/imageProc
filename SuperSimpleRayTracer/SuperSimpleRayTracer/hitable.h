#pragma once

#include "ray.h"


float rng(){
	return (float)rand() / (float)RAND_MAX;
}

vec3 random_in_unit_sphere() {
	vec3 p;
	do {
		p = 2.0*vec3(rng(), rng(), rng()) - vec3(1.0, 1.0, 1.0);
	} while (dot(p, p) >= 1.0);
	return p;
}

class material; 

struct hit_record {
	float t;
	vec3 p;
	vec3 normal;
	material *mat_ptr;
};

class hitable {
	public:
		virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};
