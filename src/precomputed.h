#pragma once
#include "tuple.h"

struct Precomputed {
	float intersectionPoint;

	Tuple point;
	Tuple eyeV;
	Tuple normalV;

	Tuple overPoint;

	bool inside;
};