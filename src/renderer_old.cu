#include <fstream>
#include <stdio.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "structures.h"
#include "analysis.h"

#define FRAME_WIDTH 1000
#define FRAME_HEIGHT 1000

#define IMAGE_WIDTH 5000
#define IMAGE_HEIGHT 5000

#define LIGHT_COUNT 1

#define SPHERE_COUNT 6
#define PLANE_COUNT 2
#define TRIANGLE_COUNT 1

#define REFLECTIVE_SPHERE_COUNT 1
#define REFLECTIVE_PLANE_COUNT 1

#define REFLECTIVE_RAY_EPILSON 0.0001
#define TRIANGLE_INTERSECTION_EPILSON 0.0000001

__constant__ Camera camera[1];

__constant__ Light lightArray[LIGHT_COUNT];

__constant__ Sphere sphereArray[SPHERE_COUNT];
__constant__ Plane planeArray[PLANE_COUNT];
__constant__ Triangle triangleArray[TRIANGLE_COUNT];

__constant__ Sphere reflectiveSphereArray[REFLECTIVE_SPHERE_COUNT];
__constant__ Plane reflectivePlaneArray[REFLECTIVE_PLANE_COUNT];

__device__ Tuple operator*(float* matrix, Tuple tuple) { return {(matrix[0] * tuple.x) + (matrix[1] * tuple.y) + (matrix[2] * tuple.z) + (matrix[3] * tuple.w), (matrix[4] * tuple.x) + (matrix[5] * tuple.y) + (matrix[6] * tuple.z) + (matrix[7] * tuple.w), (matrix[8] * tuple.x) + (matrix[9] * tuple.y) + (matrix[10] * tuple.z) + (matrix[11] * tuple.w), (matrix[12] * tuple.x) + (matrix[13] * tuple.y) + (matrix[14] * tuple.z) + (matrix[15] * tuple.w)}; }
__device__ Ray transform(Ray ray, float* matrix) { return {(matrix * ray.origin), (matrix * ray.direction)}; }
__device__ Tuple operator+(Tuple tupleA, Tuple tupleB) { return {tupleA.x + tupleB.x, tupleA.y + tupleB.y, tupleA.z + tupleB.z, tupleA.w + tupleB.w}; }
__device__ Tuple operator-(Tuple tupleA, Tuple tupleB) { return {tupleA.x - tupleB.x, tupleA.y - tupleB.y, tupleA.z - tupleB.z, tupleA.w - tupleB.w}; }
__device__ Tuple operator*(Tuple tuple, float scalar) { return {tuple.x * scalar, tuple.y * scalar, tuple.z * scalar, tuple.w * scalar}; }
__device__ Tuple operator*(float scalar, Tuple tuple) { return {tuple.x * scalar, tuple.y * scalar, tuple.z * scalar, tuple.w * scalar}; }
__device__ Tuple hadamardProduct(Tuple tupleA, Tuple tupleB) { return {tupleA.x * tupleB.x, tupleA.y * tupleB.y, tupleA.z * tupleB.z, tupleA.w * tupleB.w}; }
__device__ float magnitude(Tuple tuple) { return sqrt(tuple.x * tuple.x + tuple.y * tuple.y + tuple.z * tuple.z + tuple.w * tuple.w); }
__device__ Tuple normalize(Tuple tuple) { return {tuple.x / magnitude(tuple), tuple.y / magnitude(tuple), tuple.z / magnitude(tuple), tuple.w / magnitude(tuple)}; }
__device__ Tuple negate(Tuple tuple) { return {-tuple.x, -tuple.y, -tuple.z, -tuple.w}; }
__device__ Tuple project(Ray ray, float t) { return ray.origin + (ray.direction * t); }
__device__ float dot(Tuple tupleA, Tuple tupleB) { return (tupleA.x * tupleB.x) + (tupleA.y * tupleB.y) + (tupleA.z * tupleB.z) + (tupleA.w * tupleB.w); }
__device__ Tuple cross(Tuple tupleA, Tuple tupleB) { return {(tupleA.y * tupleB.z) - (tupleA.z * tupleB.y), (tupleA.z * tupleB.x) - (tupleA.x * tupleB.z), (tupleA.x * tupleB.y) - (tupleA.y * tupleB.x), 1.0f}; }
__device__ Tuple reflect(Tuple tuple, Tuple normal) { return tuple - (normal * 2.0f * dot(tuple, normal)); }

__device__
int intersectSphere(float* intersectionMagnitude, Sphere sphere, Ray ray) {
  Ray transformedRay = transform(ray, sphere.inverseModelMatrix);

  Tuple sphereToRay = transformedRay.origin - sphere.origin;
  float a = dot(transformedRay.direction, transformedRay.direction);
  float b = 2.0f * dot(sphereToRay, transformedRay.direction);
  float c = dot(sphereToRay, sphereToRay) - (sphere.radius * sphere.radius);

  float discriminant = (b * b) - (4.0f * a * c);
  float pointA = (-b - sqrt(discriminant)) / (2.0f * a);
  float pointB = (-b + sqrt(discriminant)) / (2.0f * a);

  *intersectionMagnitude = (pointA * (pointA <= pointB)) + (pointB * (pointB < pointA));

  return (discriminant >= 0) * (2 - (pointA == pointB)) * (pointA > 0 && pointB > 0);
}

__device__
int intersectPlane(float* intersectionMagnitude, Plane plane, Ray ray) {
  Ray transformedRay = transform(ray, plane.inverseModelMatrix);
  Tuple normal = {0.0f, -1.0f, 0.0f, 0.0f};

  float denominator = dot(normal, transformedRay.direction);
  float t = dot(plane.origin - transformedRay.origin, normal) / denominator;
  *intersectionMagnitude = t;

  return 1 * (t >= 0);
}

__device__
int intersectTriangle(float* intersectionMagnitude, Triangle triangle, Ray ray) {
  Tuple edgeB = triangle.vertexB - triangle.vertexA;
  Tuple edgeC = triangle.vertexC - triangle.vertexA;

  Tuple h = cross(ray.direction, edgeC);
  float a = dot(edgeB, h);

  if (a > -TRIANGLE_INTERSECTION_EPILSON && a < TRIANGLE_INTERSECTION_EPILSON) {
    return 0;
  }

  float f = 1.0f / a;
  Tuple s = ray.origin - triangle.vertexA;
  float u = f * dot(s, h);

  if (u < 0.0f || u > 1.0f) {
    return 0;
  }

  Tuple q = cross(s, edgeB);
  float v = f * dot(ray.direction, q);
  if (v < 0.0f || u + v > 1.0f) {
    return 0;
  }

  float t = f * dot(edgeC, q);
  if (t > TRIANGLE_INTERSECTION_EPILSON && t < 1.0f / TRIANGLE_INTERSECTION_EPILSON) {
    *intersectionMagnitude = t;
    return 1;
  }

  return 0;
}

__device__
Tuple colorFromRay(Ray ray) {
  Tuple color = {0.0f, 0.0f, 0.0f, 0.0f};
  int shapeType = 0;
  int intersectionIndex = -1;
  float intersectionMagnitude = 0.0f;

  #pragma unroll
  for (int x = 0; x < SPHERE_COUNT; x++) {
    float point;
    int count = intersectSphere(&point, sphereArray[x], ray);

    shapeType = (1 * (count > 0 && (point < intersectionMagnitude || intersectionMagnitude == 0))) + (shapeType * (count <= 0 || (point >= intersectionMagnitude && intersectionMagnitude != 0)));
    intersectionIndex = (x * (count > 0 && (point < intersectionMagnitude || intersectionMagnitude == 0))) + (intersectionIndex * (count <= 0 || (point >= intersectionMagnitude && intersectionMagnitude != 0)));
    intersectionMagnitude = (point * (count > 0 && (point < intersectionMagnitude || intersectionMagnitude == 0))) + (intersectionMagnitude * (count <= 0 || (point >= intersectionMagnitude && intersectionMagnitude != 0)));
  }

  #pragma unroll
  for (int x = 0; x < PLANE_COUNT; x++) {
    float point;
    int count = intersectPlane(&point, planeArray[x], ray);

    shapeType = (2 * (count > 0 && (point < intersectionMagnitude || intersectionMagnitude == 0))) + (shapeType * (count <= 0 || (point >= intersectionMagnitude && intersectionMagnitude != 0)));
    intersectionIndex = (x * (count > 0 && (point < intersectionMagnitude || intersectionMagnitude == 0))) + (intersectionIndex * (count <= 0 || (point >= intersectionMagnitude && intersectionMagnitude != 0)));
    intersectionMagnitude = (point * (count > 0 && (point < intersectionMagnitude || intersectionMagnitude == 0))) + (intersectionMagnitude * (count <= 0 || (point >= intersectionMagnitude && intersectionMagnitude != 0)));
  }

  #pragma unroll
  for (int x = 0; x < REFLECTIVE_SPHERE_COUNT; x++) {
    float point;
    int count = intersectSphere(&point, reflectiveSphereArray[x], ray);

    shapeType = (3 * (count > 0 && (point < intersectionMagnitude || intersectionMagnitude == 0))) + (shapeType * (count <= 0 || (point >= intersectionMagnitude && intersectionMagnitude != 0)));
    intersectionIndex = (x * (count > 0 && (point < intersectionMagnitude || intersectionMagnitude == 0))) + (intersectionIndex * (count <= 0 || (point >= intersectionMagnitude && intersectionMagnitude != 0)));
    intersectionMagnitude = (point * (count > 0 && (point < intersectionMagnitude || intersectionMagnitude == 0))) + (intersectionMagnitude * (count <= 0 || (point >= intersectionMagnitude && intersectionMagnitude != 0)));
  }

  #pragma unroll
  for (int x = 0; x < REFLECTIVE_PLANE_COUNT; x++) {
    float point;
    int count = intersectPlane(&point, reflectivePlaneArray[x], ray);

    shapeType = (4 * (count > 0 && (point < intersectionMagnitude || intersectionMagnitude == 0))) + (shapeType * (count <= 0 || (point >= intersectionMagnitude && intersectionMagnitude != 0)));
    intersectionIndex = (x * (count > 0 && (point < intersectionMagnitude || intersectionMagnitude == 0))) + (intersectionIndex * (count <= 0 || (point >= intersectionMagnitude && intersectionMagnitude != 0)));
    intersectionMagnitude = (point * (count > 0 && (point < intersectionMagnitude || intersectionMagnitude == 0))) + (intersectionMagnitude * (count <= 0 || (point >= intersectionMagnitude && intersectionMagnitude != 0)));
  }

  #pragma unroll
  for (int x = 0; x < TRIANGLE_COUNT; x++) {
    float point;
    int count = intersectTriangle(&point, triangleArray[x], ray);

    shapeType = (5 * (count > 0 && (point < intersectionMagnitude || intersectionMagnitude == 0))) + (shapeType * (count <= 0 || (point >= intersectionMagnitude && intersectionMagnitude != 0)));
    intersectionIndex = (x * (count > 0 && (point < intersectionMagnitude || intersectionMagnitude == 0))) + (intersectionIndex * (count <= 0 || (point >= intersectionMagnitude && intersectionMagnitude != 0)));
    intersectionMagnitude = (point * (count > 0 && (point < intersectionMagnitude || intersectionMagnitude == 0))) + (intersectionMagnitude * (count <= 0 || (point >= intersectionMagnitude && intersectionMagnitude != 0)));
  }

  if (intersectionIndex != -1) {
    Ray transformedRay;
    Tuple intersectionPoint;
    Ray lightRay;

    if (shapeType == 1) {
      transformedRay = transform(ray, sphereArray[intersectionIndex].inverseModelMatrix);
      intersectionPoint = project(transformedRay, intersectionMagnitude);
      lightRay = {sphereArray[intersectionIndex].modelMatrix * intersectionPoint, normalize(lightArray[0].position - (sphereArray[intersectionIndex].modelMatrix * intersectionPoint))};
    }

    if (shapeType == 2) {
      transformedRay = transform(ray, planeArray[intersectionIndex].inverseModelMatrix);
      intersectionPoint = project(transformedRay, intersectionMagnitude);
      lightRay = {planeArray[intersectionIndex].modelMatrix * intersectionPoint, normalize(lightArray[0].position - (planeArray[intersectionIndex].modelMatrix * intersectionPoint))};
    }

    if (shapeType == 3) {
      transformedRay = transform(ray, reflectiveSphereArray[intersectionIndex].inverseModelMatrix);
      intersectionPoint = project(transformedRay, intersectionMagnitude);
      lightRay = {reflectiveSphereArray[intersectionIndex].modelMatrix * intersectionPoint, normalize(lightArray[0].position - (reflectiveSphereArray[intersectionIndex].modelMatrix * intersectionPoint))};
    }

    if (shapeType == 4) {
      transformedRay = transform(ray, reflectivePlaneArray[intersectionIndex].inverseModelMatrix);
      intersectionPoint = project(transformedRay, intersectionMagnitude);
      lightRay = {reflectivePlaneArray[intersectionIndex].modelMatrix * intersectionPoint, normalize(lightArray[0].position - (reflectivePlaneArray[intersectionIndex].modelMatrix * intersectionPoint))};
    }

    if (shapeType == 5) {
      transformedRay = transform(ray, triangleArray[intersectionIndex].inverseModelMatrix);
      intersectionPoint = project(transformedRay, intersectionMagnitude);
      lightRay = {triangleArray[intersectionIndex].modelMatrix * intersectionPoint, normalize(lightArray[0].position - (triangleArray[intersectionIndex].modelMatrix * intersectionPoint))};
    }

    int intersecionCount = 0;

    #pragma unroll
    for (int x = 0; x < SPHERE_COUNT; x++) {
      float point = 0;
      intersecionCount += intersectSphere(&point, sphereArray[x], lightRay) * ((x != intersectionIndex) || (shapeType != 1)) * (point < magnitude(lightArray[0].position - intersectionPoint));
    }

    #pragma unroll
    for (int x = 0; x < PLANE_COUNT; x++) {
      float point = 0;
      intersecionCount += intersectPlane(&point, planeArray[x], lightRay) * ((x != intersectionIndex) || (shapeType != 2)) * (point < magnitude(lightArray[0].position - intersectionPoint));
    }

    #pragma unroll
    for (int x = 0; x < REFLECTIVE_SPHERE_COUNT; x++) {
      float point = 0;
      intersecionCount += intersectSphere(&point, reflectiveSphereArray[x], lightRay) * ((x != intersectionIndex) || (shapeType != 3)) * (point < magnitude(lightArray[0].position - intersectionPoint));
    }

    #pragma unroll
    for (int x = 0; x < REFLECTIVE_PLANE_COUNT; x++) {
      float point = 0;
      intersecionCount += intersectPlane(&point, reflectivePlaneArray[x], lightRay) * ((x != intersectionIndex) || (shapeType != 4)) * (point < magnitude(lightArray[0].position - intersectionPoint));
    }

    #pragma unroll
    for (int x = 0; x < TRIANGLE_COUNT; x++) {
      float point = 0;
      intersecionCount += intersectTriangle(&point, triangleArray[x], lightRay) * ((x != intersectionIndex) || (shapeType != 5)) * (point < magnitude(lightArray[0].position - intersectionPoint));
    }

    if (shapeType == 1) {
      Tuple normal = normalize(intersectionPoint - sphereArray[intersectionIndex].origin);
      float lightNormalDifference = dot(normal, lightRay.direction);

      color = (0.1f * sphereArray[intersectionIndex].color) + 
              (0.7f * lightNormalDifference * sphereArray[intersectionIndex].color * (lightNormalDifference > 0) * (intersecionCount == 0));
    }

    if (shapeType == 2) {
      Tuple normal = {0.0f, -1.0f, 0.0f, 0.0f};
      float lightNormalDifference = dot(normal, lightRay.direction);

      color = (0.1f * planeArray[intersectionIndex].color) + 
              (0.7f * lightNormalDifference * planeArray[intersectionIndex].color * (lightNormalDifference > 0) * (intersecionCount == 0));
    }

    if (shapeType == 3) {
      Tuple normal = normalize(intersectionPoint - reflectiveSphereArray[intersectionIndex].origin);
      float lightNormalDifference = dot(normal, lightRay.direction);

      color = (0.1f * reflectiveSphereArray[intersectionIndex].color) + 
              (0.7f * lightNormalDifference * reflectiveSphereArray[intersectionIndex].color * (lightNormalDifference > 0) * (intersecionCount == 0));
    }

    if (shapeType == 4) {
      Tuple normal = {0.0f, -1.0f, 0.0f, 0.0f};
      float lightNormalDifference = dot(normal, lightRay.direction);

      color = (0.1f * reflectivePlaneArray[intersectionIndex].color) + 
              (0.7f * lightNormalDifference * reflectivePlaneArray[intersectionIndex].color * (lightNormalDifference > 0) * (intersecionCount == 0));
    }

    if (shapeType == 5) {
      color = (1.0f * triangleArray[intersectionIndex].color);
    }
  }

  return color;
}

__device__
Ray rayFromReflection(Ray ray, int recursionCount = 0) {
  int shapeType = 0;
  int intersectionIndex = -1;
  float intersectionMagnitude = 0.0f;

  #pragma unroll
  for (int x = 0; x < SPHERE_COUNT; x++) {
    float point;
    int count = intersectSphere(&point, sphereArray[x], ray);

    shapeType = (1 * (count > 0 && (point < intersectionMagnitude || intersectionMagnitude == 0))) + (shapeType * (count <= 0 || (point >= intersectionMagnitude && intersectionMagnitude != 0)));
    intersectionIndex = (x * (count > 0 && (point < intersectionMagnitude || intersectionMagnitude == 0))) + (intersectionIndex * (count <= 0 || (point >= intersectionMagnitude && intersectionMagnitude != 0)));
    intersectionMagnitude = (point * (count > 0 && (point < intersectionMagnitude || intersectionMagnitude == 0))) + (intersectionMagnitude * (count <= 0 || (point >= intersectionMagnitude && intersectionMagnitude != 0)));
  }

  #pragma unroll
  for (int x = 0; x < PLANE_COUNT; x++) {
    float point;
    int count = intersectPlane(&point, planeArray[x], ray);

    shapeType = (2 * (count > 0 && (point < intersectionMagnitude || intersectionMagnitude == 0))) + (shapeType * (count <= 0 || (point >= intersectionMagnitude && intersectionMagnitude != 0)));
    intersectionIndex = (x * (count > 0 && (point < intersectionMagnitude || intersectionMagnitude == 0))) + (intersectionIndex * (count <= 0 || (point >= intersectionMagnitude && intersectionMagnitude != 0)));
    intersectionMagnitude = (point * (count > 0 && (point < intersectionMagnitude || intersectionMagnitude == 0))) + (intersectionMagnitude * (count <= 0 || (point >= intersectionMagnitude && intersectionMagnitude != 0)));
  }

  #pragma unroll
  for (int x = 0; x < REFLECTIVE_SPHERE_COUNT; x++) {
    float point;
    int count = intersectSphere(&point, reflectiveSphereArray[x], ray);

    shapeType = (3 * (count > 0 && (point < intersectionMagnitude || intersectionMagnitude == 0))) + (shapeType * (count <= 0 || (point >= intersectionMagnitude && intersectionMagnitude != 0)));
    intersectionIndex = (x * (count > 0 && (point < intersectionMagnitude || intersectionMagnitude == 0))) + (intersectionIndex * (count <= 0 || (point >= intersectionMagnitude && intersectionMagnitude != 0)));
    intersectionMagnitude = (point * (count > 0 && (point < intersectionMagnitude || intersectionMagnitude == 0))) + (intersectionMagnitude * (count <= 0 || (point >= intersectionMagnitude && intersectionMagnitude != 0)));
  }

  #pragma unroll
  for (int x = 0; x < REFLECTIVE_PLANE_COUNT; x++) {
    float point;
    int count = intersectPlane(&point, reflectivePlaneArray[x], ray);

    shapeType = (4 * (count > 0 && (point < intersectionMagnitude || intersectionMagnitude == 0))) + (shapeType * (count <= 0 || (point >= intersectionMagnitude && intersectionMagnitude != 0)));
    intersectionIndex = (x * (count > 0 && (point < intersectionMagnitude || intersectionMagnitude == 0))) + (intersectionIndex * (count <= 0 || (point >= intersectionMagnitude && intersectionMagnitude != 0)));
    intersectionMagnitude = (point * (count > 0 && (point < intersectionMagnitude || intersectionMagnitude == 0))) + (intersectionMagnitude * (count <= 0 || (point >= intersectionMagnitude && intersectionMagnitude != 0)));
  }

  Ray reflectedRay;
  if (shapeType == 3) {
    Ray transformedRay = transform(ray, reflectiveSphereArray[intersectionIndex].inverseModelMatrix);
    Tuple intersectionPoint = project(transformedRay, intersectionMagnitude);
    Tuple normal = normalize(intersectionPoint - reflectiveSphereArray[intersectionIndex].origin);

    reflectedRay = {reflectiveSphereArray[intersectionIndex].modelMatrix * intersectionPoint, reflect(transformedRay.direction, normal)};
  }

  if (shapeType == 4) {
    Ray transformedRay = transform(ray, reflectivePlaneArray[intersectionIndex].inverseModelMatrix);
    Tuple intersectionPoint = project(transformedRay, intersectionMagnitude - REFLECTIVE_RAY_EPILSON);
    Tuple normal = {0.0f, -1.0f, 0.0f, 0.0f};

    reflectedRay = {reflectivePlaneArray[intersectionIndex].modelMatrix * intersectionPoint, reflect(transformedRay.direction, normal)};
  }

  return reflectedRay;
}

__global__
void lighting(Tuple* colorOut, int renderWidth, int renderHeight) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (idx >= renderWidth || idy >= renderHeight) { return; }

  Tuple pixel = {
    (idx - (renderWidth / 2.0f)) / renderWidth, 
    (idy - (renderHeight / 2.0f)) / renderHeight, 
    0.0f, 1.0f
  };
  Tuple direction = normalize((pixel + camera[0].direction) - camera[0].position);
  Ray ray = {camera[0].position, direction};
  ray = transform(ray, camera[0].modelMatrix);

  colorOut[(idy*renderWidth)+idx] = colorFromRay(ray);
}

__global__
void reflections(Tuple* colorOut, int renderWidth, int renderHeight) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (idx >= renderWidth || idy >= renderHeight) { return; }

  Tuple pixel = {
    (idx - (renderWidth / 2.0f)) / renderWidth, 
    (idy - (renderHeight / 2.0f)) / renderHeight, 
    0.0f, 1.0f
  };
  Tuple direction = normalize((pixel + camera[0].direction) - camera[0].position);
  Ray ray = {camera[0].position, direction};
  ray = transform(ray, camera[0].modelMatrix);

  colorOut[(idy*renderWidth)+idx] = colorFromRay(rayFromReflection(ray));
}

void writeColorDataToFile(const char* filename, unsigned int* colorData) {
  std::ofstream file;
  file.open(filename);
  file << "P3\n" << IMAGE_WIDTH << " " << IMAGE_HEIGHT << "\n255\n";

  for (int x = 0; x < IMAGE_WIDTH * IMAGE_HEIGHT; x++) {
    file << int(colorData[x] & 0x0000FF) << " ";
    file << int((colorData[x] & 0x00FF00) >> 8) << " ";
    file << int((colorData[x] & 0xFF0000) >> 16) << "\n";
  }

  file.close();
}

__global__
void combineLightingReflectionBuffers(unsigned int* cudaBuffer, Tuple* lightingBuffer, Tuple* reflectionsBuffer, int renderWidth, int renderHeight) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (idx >= renderWidth || idy >= renderHeight) { return; }

  Tuple color;
  if (reflectionsBuffer[(idy*renderWidth)+idx].w > 0) {
    color = (0.2 * reflectionsBuffer[(idy*renderWidth)+idx]) + lightingBuffer[(idy*renderWidth)+idx];
  }
  else {
    color = lightingBuffer[(idy*renderWidth)+idx];
  }

  cudaBuffer[(idy*renderWidth)+idx] = (int(fmaxf(0, fminf(255, color.z))) << 16) | (int(fmaxf(0, fminf(255, color.y))) << 8) | (int(fmaxf(0, fminf(255, color.x))));
}

Tuple* lightingBuffer;
Tuple* reflectionsBuffer;

extern "C" void updateCamera(float x, float y, float z, float rotationX, float rotationY) {
  Camera h_camera[] = {{{0.0, 0.0, 0.0, 1.0}, {0.0, 0.0, 1.0, 0.0}}};
  initializeModelMatrix(h_camera[0].modelMatrix, multiply(multiply(createTranslateMatrix(x, y, z), createRotationMatrixY(rotationY)), createRotationMatrixX(rotationX)));
  initializeInverseModelMatrix(h_camera[0].inverseModelMatrix, h_camera[0].modelMatrix);
  cudaMemcpyToSymbol(camera, h_camera, sizeof(Camera));
}

extern "C" void initializeScene() {
  cudaMalloc(&lightingBuffer, FRAME_WIDTH*FRAME_HEIGHT*sizeof(Tuple));
  cudaMalloc(&reflectionsBuffer, FRAME_WIDTH*FRAME_HEIGHT*sizeof(Tuple));

  Camera h_camera[] = {{{0.0, 0.0, 0.0, 1.0}, {0.0, 0.0, 1.0, 0.0}}};
  initializeModelMatrix(h_camera[0].modelMatrix, multiply(multiply(createTranslateMatrix(5.0, -3.5, -6.0), createRotationMatrixY(-M_PI / 4.5)), createRotationMatrixX(-M_PI / 12.0)));
  initializeInverseModelMatrix(h_camera[0].inverseModelMatrix, h_camera[0].modelMatrix);
  cudaMemcpyToSymbol(camera, h_camera, sizeof(Camera));

  Light h_lightArray[] = {{{10.0, -10.0, -5.0, 1.0}, {1.0, 1.0, 1.0, 1.0}}};
  cudaMemcpyToSymbol(lightArray, h_lightArray, LIGHT_COUNT*sizeof(Light));

  Sphere h_sphereArray[] = {
                {{0.0, 0.0, 0.0, 1.0}, 1.0, {178.5, 255.0, 51.0, 1.0}},
                {{0.0, 0.0, 0.0, 1.0}, 1.0, {255.0, 255.0, 127.5, 1.0}},
                {{0.0, 0.0, 0.0, 1.0}, 1.0, {76.5, 51.0, 127.5, 1.0}},
                {{0.0, 0.0, 0.0, 1.0}, 1.0, {255.0, 255.0, 255.0, 1.0}},
                {{0.0, 0.0, 0.0, 1.0}, 1.0, {76.5, 76.5, 255.0, 1.0}},
                {{0.0, 0.0, 0.0, 1.0}, 1.0, {255.5, 76.5, 255.0, 1.0}}
              };
  initializeModelMatrix(&h_sphereArray[0], createTranslateMatrix(-0.5, -1, -2.0));
  initializeModelMatrix(&h_sphereArray[1], multiply(createTranslateMatrix(-0.5, -2, -2.0), createScaleMatrix(0.75, 0.75, 0.75)));
  initializeModelMatrix(&h_sphereArray[2], multiply(createTranslateMatrix(2, -1, 0.5), createScaleMatrix(1.25, 1.25, 1.25)));
  initializeModelMatrix(&h_sphereArray[3], multiply(createTranslateMatrix(2.0, -0.25, -1.5), createScaleMatrix(0.5, 0.5, 0.5)));
  initializeModelMatrix(&h_sphereArray[4], multiply(createTranslateMatrix(1.25, -2, -3.0), createScaleMatrix(0.25, 0.25, 0.25)));
  initializeModelMatrix(&h_sphereArray[5], multiply(createTranslateMatrix(8.0, -2.0, -7.0), createScaleMatrix(2.25, 2.25, 2.25)));
  cudaMemcpyToSymbol(sphereArray, h_sphereArray, SPHERE_COUNT*sizeof(Sphere));

  Plane h_planeArray[] = {
              {{0.0, 0.0, 0.0, 1.0}, {229.5, 127.5, 229.5, 1.0}},
              {{0.0, 0.0, 0.0, 1.0}, {229.5, 229.5, 127.5, 1.0}}
            };
  initializeModelMatrix(&h_planeArray[0], multiply(createTranslateMatrix(0.0, 0.0, 3.0), createRotationMatrixX(M_PI / 2)));
  initializeModelMatrix(&h_planeArray[1], multiply(createTranslateMatrix(-3.0, 0.0, 0.0), createRotationMatrixZ(M_PI / 2)));
  cudaMemcpyToSymbol(planeArray, h_planeArray, PLANE_COUNT*sizeof(Plane));

  Triangle h_triangleArray[] = {
              {{0.0, 0.0, 0.0, 1.0}, {0.0, -1.0, 0.0, 1.0}, {1.0, 0.0, 0.0, 1.0}, {255.0, 255.0, 255.0, 1.0}},
            };
  initializeModelMatrix(&h_triangleArray[0], createIdentityMatrix());
  cudaMemcpyToSymbol(triangleArray, h_triangleArray, TRIANGLE_COUNT*sizeof(Triangle));

  Sphere h_reflectiveSphereArray[] = {
                {{0.0, 0.0, 0.0, 1.0}, 1.0, {255.0, 255.0, 255.0, 1.0}}
  };
  initializeModelMatrix(&h_reflectiveSphereArray[0], createTranslateMatrix(-0.5, -3.0, 0.5));
  cudaMemcpyToSymbol(reflectiveSphereArray, h_reflectiveSphereArray, REFLECTIVE_SPHERE_COUNT*sizeof(Sphere));

  Plane h_reflectivePlaneArray[] = {
            {{0.0, 0.0, 0.0, 1.0}, {127.5, 229.5, 229.5, 1.0}},
          };
  initializeModelMatrix(&h_reflectivePlaneArray[0], createTranslateMatrix(0.0, 0.0, 0.0));
  cudaMemcpyToSymbol(reflectivePlaneArray, h_reflectivePlaneArray, REFLECTIVE_PLANE_COUNT*sizeof(Plane));
}

extern "C" void renderFrame(int blockDimX, int blockDimY, void* cudaBuffer, cudaGraphicsResource_t* cudaTextureResource) {
  dim3 block(blockDimX, blockDimY);
  dim3 grid((FRAME_WIDTH + block.x - 1) / block.x, (FRAME_HEIGHT + block.y - 1) / block.y);
  lighting<<<grid, block>>>(lightingBuffer, FRAME_WIDTH, FRAME_HEIGHT);
  reflections<<<grid, block>>>(reflectionsBuffer, FRAME_WIDTH, FRAME_HEIGHT);
  combineLightingReflectionBuffers<<<grid, block>>>((unsigned int*)cudaBuffer, lightingBuffer, reflectionsBuffer, FRAME_WIDTH, FRAME_HEIGHT);

  cudaArray *texture_ptr;
  cudaGraphicsMapResources(1, cudaTextureResource, 0);
  cudaGraphicsSubResourceGetMappedArray(&texture_ptr, *cudaTextureResource, 0, 0);

  cudaMemcpy2DToArray(texture_ptr, 0, 0,  cudaBuffer, FRAME_WIDTH*4*sizeof(GLubyte), FRAME_WIDTH*4*sizeof(GLubyte), FRAME_HEIGHT, cudaMemcpyDeviceToDevice);
  cudaGraphicsUnmapResources(1, cudaTextureResource, 0);
}

extern "C" void renderImage(int blockDimX, int blockDimY, const char* filename) {
  printf("\n");

  Analysis::setAbsoluteStart();
  Analysis::createLabel(0, "allocate_memory");
  Analysis::createLabel(1, "execute_kernel");
  Analysis::createLabel(2, "copy_device");
  Analysis::createLabel(3, "create_image");

  Analysis::begin();
  unsigned int* h_imageData = (unsigned int*)malloc(IMAGE_WIDTH*IMAGE_HEIGHT*4*sizeof(GLubyte));
  unsigned int* d_imageData;
  cudaMalloc((unsigned int**)&d_imageData, IMAGE_WIDTH*IMAGE_HEIGHT*4*sizeof(GLubyte));

  Tuple *d_lightingData, *d_reflectionsData;
  cudaMalloc((Tuple**)&d_lightingData, IMAGE_WIDTH*IMAGE_HEIGHT*sizeof(Tuple));
  cudaMalloc((Tuple**)&d_reflectionsData, IMAGE_WIDTH*IMAGE_HEIGHT*sizeof(Tuple));
  Analysis::end(0);

  dim3 block(blockDimX, blockDimY);
  dim3 grid((IMAGE_WIDTH + block.x - 1) / block.x, (IMAGE_HEIGHT + block.y - 1) / block.y);

  Analysis::begin();
  printf("rendering ray traced image...\n");
  lighting<<<grid, block>>>(d_lightingData, IMAGE_WIDTH, IMAGE_HEIGHT);
  reflections<<<grid, block>>>(d_reflectionsData, IMAGE_WIDTH, IMAGE_HEIGHT);
  combineLightingReflectionBuffers<<<grid, block>>>(d_imageData, d_lightingData, d_reflectionsData, IMAGE_WIDTH, IMAGE_HEIGHT);
  cudaDeviceSynchronize();
  printf("finished rendering\n");
  Analysis::end(1);

  Analysis::begin();
  cudaMemcpy(h_imageData, d_imageData, IMAGE_WIDTH*IMAGE_HEIGHT*4*sizeof(GLubyte), cudaMemcpyDeviceToHost);
  Analysis::end(2);

  Analysis::begin();
  writeColorDataToFile(filename, h_imageData);
  printf("saved image as: [%s]\n", filename);
  Analysis::end(3);

  Analysis::printAll(IMAGE_WIDTH, IMAGE_HEIGHT);

  free(h_imageData);
  cudaFree(d_imageData);
  cudaFree(d_lightingData);
  cudaFree(d_reflectionsData);
}