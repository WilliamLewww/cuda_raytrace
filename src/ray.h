#pragma once
#include "tuple.h"

struct Ray {
	Tuple origin;
	Tuple direction;
};

__device__
Tuple project(Ray ray, float t) {
	return ray.origin + (ray.direction * t);
}