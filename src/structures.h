#pragma once

struct Tuple {
	float x;
	float y;
	float z;
	float w;
};

struct Ray {
	Tuple origin;
	Tuple direction;
};

struct Light {
	Tuple position;
	Tuple intensity;
};

struct Sphere {
	Tuple origin;
};

__device__ Tuple operator+(Tuple tupleA, Tuple tupleB) { return {tupleA.x + tupleB.x, tupleA.y + tupleB.y, tupleA.z + tupleB.z, tupleA.w + tupleB.w}; }
__device__ Tuple operator-(Tuple tupleA, Tuple tupleB) { return {tupleA.x - tupleB.x, tupleA.y - tupleB.y, tupleA.z - tupleB.z, tupleA.w - tupleB.w}; }
__device__ Tuple operator*(Tuple tuple, float scalar) { return {tuple.x * scalar, tuple.y * scalar, tuple.z * scalar, tuple.w * scalar}; }
__device__ Tuple hadamardProduct(Tuple tupleA, Tuple tupleB) { return {tupleA.x * tupleB.x, tupleA.y * tupleB.y, tupleA.z * tupleB.z, tupleA.w * tupleB.w}; }
__device__ float magnitude(Tuple tuple) { return sqrt(pow(tuple.x, 2) + pow(tuple.y, 2) + pow(tuple.z, 2) + pow(tuple.w, 2)); }
__device__ Tuple normalize(Tuple tuple) { return {tuple.x / magnitude(tuple), tuple.y / magnitude(tuple), tuple.z / magnitude(tuple), tuple.w / magnitude(tuple)}; }
__device__ Tuple negate(Tuple tuple) { return {-tuple.x, -tuple.y, -tuple.z, -tuple.w}; }
__device__ Tuple project(Ray ray, float t) { return ray.origin + (ray.direction * t); }
__device__ float dot(Tuple tupleA, Tuple tupleB) { return (tupleA.x * tupleB.x) + (tupleA.y * tupleB.y) + (tupleA.z * tupleB.z) + (tupleA.w * tupleB.w); }