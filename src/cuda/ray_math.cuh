#pragma once
#include "../raytrace_structures.h"

__device__ Tuple operator*(float* matrix, Tuple tuple);
__device__ Tuple operator+(Tuple tupleA, Tuple tupleB);
__device__ Tuple operator-(Tuple tupleA, Tuple tupleB);
__device__ Tuple operator*(Tuple tuple, float scalar);
__device__ Tuple operator*(float scalar, Tuple tuple);
__device__ Ray d_transform(Ray ray, float* matrix);
__device__ Tuple d_hadamardProduct(Tuple tupleA, Tuple tupleB);
__device__ float d_magnitude(Tuple tuple);
__device__ Tuple d_normalize(Tuple tuple);
__device__ Tuple d_negate(Tuple tuple);
__device__ Tuple d_project(Ray ray, float t);
__device__ float d_dot(Tuple tupleA, Tuple tupleB);
__device__ Tuple d_cross(Tuple tupleA, Tuple tupleB);
__device__ Tuple d_reflect(Tuple tuple, Tuple normal);