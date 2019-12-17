#include <fstream>
#include <stdio.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "structures.h"
#include "analysis.h"

#define IMAGE_WIDTH 1000
#define IMAGE_HEIGHT 1000

#define LIGHT_COUNT 1

#define SPHERE_COUNT 6
#define PLANE_COUNT 2

#define REFLECTIVE_SPHERE_COUNT 1
#define REFLECTIVE_PLANE_COUNT 1

#define REFLECTIVE_RAY_EPILSON 0.0001

__constant__ Camera camera[1];

__constant__ Light lightArray[LIGHT_COUNT];

__constant__ Sphere sphereArray[SPHERE_COUNT];
__constant__ Plane planeArray[PLANE_COUNT];

__constant__ Sphere reflectiveSphereArray[REFLECTIVE_SPHERE_COUNT];
__constant__ Plane reflectivePlaneArray[REFLECTIVE_PLANE_COUNT];

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

    if (shapeType == 1) {
      Tuple normal = normalize(intersectionPoint - sphereArray[intersectionIndex].origin);
      float lightNormalDifference = dot(normal, lightRay.direction);

      Tuple lightReflection = reflect(negate(lightRay.direction), normal);
      Tuple eyeDirection = (camera[0].inverseModelMatrix * lightRay.direction) - camera[0].position;

      float reflectEyeDifference = dot(lightReflection, eyeDirection);

      color = (0.1f * sphereArray[intersectionIndex].color) + 
              (0.7f * lightNormalDifference * sphereArray[intersectionIndex].color * (lightNormalDifference > 0) * (intersecionCount == 0)) +
              (0.2f * sphereArray[intersectionIndex].color * pow(reflectEyeDifference, 200.0f) * (reflectEyeDifference > 0) * (intersecionCount == 0));
    }

    if (shapeType == 2) {
      Tuple normal = {0.0f, -1.0f, 0.0f, 0.0f};
      float lightNormalDifference = dot(normal, lightRay.direction);

      Tuple lightReflection = reflect(negate(lightRay.direction), normal);
      Tuple eyeDirection = (camera[0].inverseModelMatrix * lightRay.direction) - camera[0].position;

      float reflectEyeDifference = dot(lightReflection, eyeDirection);

      color = (0.1f * planeArray[intersectionIndex].color) + 
              (0.7f * lightNormalDifference * planeArray[intersectionIndex].color * (lightNormalDifference > 0) * (intersecionCount == 0)) +
              (0.2f * planeArray[intersectionIndex].color * pow(reflectEyeDifference, 200.0f) * (reflectEyeDifference > 0) * (intersecionCount == 0));
    }

    if (shapeType == 3) {
      Tuple normal = normalize(intersectionPoint - reflectiveSphereArray[intersectionIndex].origin);
      float lightNormalDifference = dot(normal, lightRay.direction);

      Tuple lightReflection = reflect(negate(lightRay.direction), normal);
      Tuple eyeDirection = (camera[0].inverseModelMatrix * lightRay.direction) - camera[0].position;

      float reflectEyeDifference = dot(lightReflection, eyeDirection);

      color = (0.1f * reflectiveSphereArray[intersectionIndex].color) + 
              (0.7f * lightNormalDifference * reflectiveSphereArray[intersectionIndex].color * (lightNormalDifference > 0) * (intersecionCount == 0)) +
              (0.2f * reflectiveSphereArray[intersectionIndex].color * pow(reflectEyeDifference, 200.0f) * (reflectEyeDifference > 0) * (intersecionCount == 0));
    }

    if (shapeType == 4) {
      Tuple normal = {0.0f, -1.0f, 0.0f, 0.0f};
      float lightNormalDifference = dot(normal, lightRay.direction);

      Tuple lightReflection = reflect(negate(lightRay.direction), normal);
      Tuple eyeDirection = (camera[0].inverseModelMatrix * lightRay.direction) - camera[0].position;

      float reflectEyeDifference = dot(lightReflection, eyeDirection);

      color = (0.1f * reflectivePlaneArray[intersectionIndex].color) + 
              (0.7f * lightNormalDifference * reflectivePlaneArray[intersectionIndex].color * (lightNormalDifference > 0) * (intersecionCount == 0)) +
              (0.2f * reflectivePlaneArray[intersectionIndex].color * pow(reflectEyeDifference, 200.0f) * (reflectEyeDifference > 0) * (intersecionCount == 0));
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
void lighting(Tuple* colorOut) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (idx >= IMAGE_WIDTH || idy >= IMAGE_HEIGHT) { return; }

  Tuple pixel = {
    (idx - (IMAGE_WIDTH / 2.0f)) / IMAGE_WIDTH, 
    (idy - (IMAGE_HEIGHT / 2.0f)) / IMAGE_HEIGHT, 
    0.0f, 1.0f
  };
  Tuple direction = normalize((pixel + camera[0].direction) - camera[0].position);
  Ray ray = {camera[0].position, direction};
  ray = transform(ray, camera[0].modelMatrix);

  colorOut[(idy*IMAGE_WIDTH)+idx] = colorFromRay(ray);
}

__global__
void reflections(Tuple* colorOut) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (idx >= IMAGE_WIDTH || idy >= IMAGE_HEIGHT) { return; }

  Tuple pixel = {
    (idx - (IMAGE_WIDTH / 2.0f)) / IMAGE_WIDTH, 
    (idy - (IMAGE_HEIGHT / 2.0f)) / IMAGE_HEIGHT, 
    0.0f, 1.0f
  };
  Tuple direction = normalize((pixel + camera[0].direction) - camera[0].position);
  Ray ray = {camera[0].position, direction};
  ray = transform(ray, camera[0].modelMatrix);

  colorOut[(idy*IMAGE_WIDTH)+idx] = colorFromRay(rayFromReflection(ray));
}

void combineLightingReflections(Tuple* first, Tuple* second) {
  for (int x = 0; x < IMAGE_WIDTH * IMAGE_HEIGHT; x++) {
    if (second[x].w > 0) {
      first[x] = second[x];
    }
  }
}

void writeColorDataToFile(const char* filename, Tuple* colorData) {
  std::ofstream file;
  file.open(filename);
  file << "P3\n" << IMAGE_WIDTH << " " << IMAGE_HEIGHT << "\n255\n";

  for (int x = 0; x < IMAGE_WIDTH * IMAGE_HEIGHT; x++) {
    file << int(colorData[x].x) << " ";
    file << int(colorData[x].y) << " ";
    file << int(colorData[x].z) << "\n";
  }

  file.close();
}

__global__
void combineLightingReflectionBuffers(unsigned int* cudaBuffer, Tuple* lightingBuffer, Tuple* reflectionsBuffer) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (idx >= IMAGE_WIDTH || idy >= IMAGE_HEIGHT) { return; }

  Tuple color;
  if (reflectionsBuffer[(idy*IMAGE_WIDTH)+idx].w > 0) {
    color = (0.2 * reflectionsBuffer[(idy*IMAGE_WIDTH)+idx]) + lightingBuffer[(idy*IMAGE_WIDTH)+idx];
  }
  else {
    color = lightingBuffer[(idy*IMAGE_WIDTH)+idx];
  }

  cudaBuffer[(idy*IMAGE_WIDTH)+idx] = (int(fmaxf(0, fminf(255, color.z))) << 16) | (int(fmaxf(0, fminf(255, color.y))) << 8) | int(fmaxf(0, fminf(255, color.x)));
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
  cudaMalloc(&lightingBuffer, 1000*1000*sizeof(Tuple));
  cudaMalloc(&reflectionsBuffer, 1000*1000*sizeof(Tuple));

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
  dim3 grid((IMAGE_WIDTH + block.x - 1) / block.x, (IMAGE_HEIGHT + block.y - 1) / block.y);
  lighting<<<grid, block>>>(lightingBuffer);
  reflections<<<grid, block>>>(reflectionsBuffer);
  combineLightingReflectionBuffers<<<grid, block>>>((unsigned int*)cudaBuffer, lightingBuffer, reflectionsBuffer);

  cudaArray *texture_ptr;
  cudaGraphicsMapResources(1, cudaTextureResource, 0);
  cudaGraphicsSubResourceGetMappedArray(&texture_ptr, *cudaTextureResource, 0, 0);

  cudaMemcpy2DToArray(texture_ptr, 0, 0,  cudaBuffer, 1000*4*sizeof(GLubyte), 1000*4*sizeof(GLubyte), 1000, cudaMemcpyDeviceToDevice);
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
  Camera h_camera[] = {{{0.0, 0.0, 0.0, 1.0}, {0.0, 0.0, 1.0, 0.0}}};
  initializeModelMatrix(h_camera[0].modelMatrix, multiply(multiply(createTranslateMatrix(5.0, -3.5, -6.0), createRotationMatrixY(-M_PI / 4.5)), createRotationMatrixX(-M_PI / 12.0)));
  initializeInverseModelMatrix(h_camera[0].inverseModelMatrix, h_camera[0].modelMatrix);
  cudaMemcpyToSymbol(camera, h_camera, sizeof(Camera));

  const Light h_lightArray[] = {{{10.0, -10.0, -5.0, 1.0}, {1.0, 1.0, 1.0, 1.0}}};
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
              {{0.0, 0.0, 0.0, 1.0}, {127.5, 229.5, 229.5, 1.0}},
              {{0.0, 0.0, 0.0, 1.0}, {229.5, 127.5, 229.5, 1.0}},
              {{0.0, 0.0, 0.0, 1.0}, {229.5, 229.5, 127.5, 1.0}}
            };
  initializeModelMatrix(&h_planeArray[0], createTranslateMatrix(0.0, 0.0, 0.0));
  initializeModelMatrix(&h_planeArray[1], multiply(createTranslateMatrix(0.0, 0.0, 3.0), createRotationMatrixX(M_PI / 2)));
  initializeModelMatrix(&h_planeArray[2], multiply(createTranslateMatrix(-3.0, 0.0, 0.0), createRotationMatrixZ(M_PI / 2)));
  cudaMemcpyToSymbol(planeArray, h_planeArray, PLANE_COUNT*sizeof(Plane));

  Sphere h_reflectiveSphereArray[] = {
                {{0.0, 0.0, 0.0, 1.0}, 1.0, {255.0, 255.0, 255.0, 1.0}}
  };
  initializeModelMatrix(&h_reflectiveSphereArray[0], createTranslateMatrix(-0.5, -3.0, 0.5));
  cudaMemcpyToSymbol(reflectiveSphereArray, h_reflectiveSphereArray, REFLECTIVE_SPHERE_COUNT*sizeof(Sphere));

  Tuple* h_lightingData = (Tuple*)malloc(IMAGE_WIDTH*IMAGE_HEIGHT*sizeof(Tuple));
  Tuple* h_reflectionsData = (Tuple*)malloc(IMAGE_WIDTH*IMAGE_HEIGHT*sizeof(Tuple));
  Tuple *d_lightingData, *d_reflectionsData;
  cudaMalloc((Tuple**)&d_lightingData, IMAGE_WIDTH*IMAGE_HEIGHT*sizeof(Tuple));
  cudaMalloc((Tuple**)&d_reflectionsData, IMAGE_WIDTH*IMAGE_HEIGHT*sizeof(Tuple));
  Analysis::end(0);

  dim3 block(blockDimX, blockDimY);
  dim3 grid((IMAGE_WIDTH + block.x - 1) / block.x, (IMAGE_HEIGHT + block.y - 1) / block.y);

  Analysis::begin();
  printf("rendering ray traced image...\n");
  lighting<<<grid, block>>>(d_lightingData);
  reflections<<<grid, block>>>(d_reflectionsData);
  cudaDeviceSynchronize();
  printf("finished rendering\n");
  Analysis::end(1);

  Analysis::begin();
  cudaMemcpy(h_lightingData, d_lightingData, IMAGE_WIDTH*IMAGE_HEIGHT*sizeof(Tuple), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_reflectionsData, d_reflectionsData, IMAGE_WIDTH*IMAGE_HEIGHT*sizeof(Tuple), cudaMemcpyDeviceToHost);
  combineLightingReflections(h_lightingData, h_reflectionsData);
  cudaFree(d_lightingData);
  cudaFree(d_reflectionsData);
  Analysis::end(2);

  Analysis::begin();
  writeColorDataToFile(filename, h_lightingData);
  printf("saved image as: [%s]\n", filename);
  Analysis::end(3);

  Analysis::printAll(IMAGE_WIDTH, IMAGE_HEIGHT);

  cudaDeviceReset();
  free(h_lightingData);
  free(h_reflectionsData);
  printf("\n");
}