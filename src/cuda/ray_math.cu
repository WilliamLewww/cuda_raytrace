#include "ray_math.cuh"

__device__ Tuple operator*(float* matrix, Tuple tuple) { 
  return {
    (matrix[0] * tuple.x) + (matrix[1] * tuple.y) + (matrix[2] * tuple.z) + (matrix[3] * tuple.w), 
    (matrix[4] * tuple.x) + (matrix[5] * tuple.y) + (matrix[6] * tuple.z) + (matrix[7] * tuple.w), 
    (matrix[8] * tuple.x) + (matrix[9] * tuple.y) + (matrix[10] * tuple.z) + (matrix[11] * tuple.w), 
    (matrix[12] * tuple.x) + (matrix[13] * tuple.y) + (matrix[14] * tuple.z) + (matrix[15] * tuple.w)
  }; 
}

__device__ Tuple operator+(Tuple tupleA, Tuple tupleB) { 
  return {
    tupleA.x + tupleB.x, 
    tupleA.y + tupleB.y, 
    tupleA.z + tupleB.z, 
    tupleA.w + tupleB.w
  }; 
}

__device__ Tuple operator-(Tuple tupleA, Tuple tupleB) { 
  return {
    tupleA.x - tupleB.x, 
    tupleA.y - tupleB.y, 
    tupleA.z - tupleB.z, 
    tupleA.w - tupleB.w
  }; 
}

__device__ Tuple operator*(Tuple tuple, float scalar) { 
  return {
    tuple.x * scalar, 
    tuple.y * scalar, 
    tuple.z * scalar, 
    tuple.w * scalar
  }; 
}

__device__ Tuple operator*(float scalar, Tuple tuple) { 
  return {
    tuple.x * scalar, 
    tuple.y * scalar, 
    tuple.z * scalar, 
    tuple.w * scalar
  }; 
}

__device__ Ray d_transform(Ray ray, float* matrix) { 
  return {
    (matrix * ray.origin), 
    (matrix * ray.direction)
  }; 
}

__device__ Tuple d_hadamardProduct(Tuple tupleA, Tuple tupleB) { 
  return {
    tupleA.x * tupleB.x, 
    tupleA.y * tupleB.y, 
    tupleA.z * tupleB.z, 
    tupleA.w * tupleB.w
  }; 
}

__device__ float d_magnitude(Tuple tuple) { 
  return sqrt(tuple.x * tuple.x + tuple.y * tuple.y + tuple.z * tuple.z + tuple.w * tuple.w); 
}

__device__ Tuple d_normalize(Tuple tuple) { 
  return {
    tuple.x / d_magnitude(tuple), 
    tuple.y / d_magnitude(tuple), 
    tuple.z / d_magnitude(tuple), 
    tuple.w / d_magnitude(tuple)
  }; 
}

__device__ Tuple d_negate(Tuple tuple) { 
  return {
    -tuple.x, 
    -tuple.y, 
    -tuple.z, 
    -tuple.w
  }; 
}

__device__ Tuple d_project(Ray ray, float t) { 
  return ray.origin + (ray.direction * t); 
}

__device__ float d_dot(Tuple tupleA, Tuple tupleB) { 
  return (tupleA.x * tupleB.x) + (tupleA.y * tupleB.y) + (tupleA.z * tupleB.z) + (tupleA.w * tupleB.w); 
}

__device__ Tuple d_cross(Tuple tupleA, Tuple tupleB) { 
  return {
    (tupleA.y * tupleB.z) - (tupleA.z * tupleB.y), 
    (tupleA.z * tupleB.x) - (tupleA.x * tupleB.z), 
    (tupleA.x * tupleB.y) - (tupleA.y * tupleB.x), 
    1.0f
  }; 
}

__device__ Tuple d_reflect(Tuple tuple, Tuple normal) { 
  return tuple - (normal * 2.0f * d_dot(tuple, normal)); 
}