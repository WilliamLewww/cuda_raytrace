#include <algorithm>
#include <vector>
#include <stdio.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "raytrace_structures.h"
#include "model.h"
#include "analysis.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define REFLECTIVE_RAY_EPILSON 0.0001
#define SHADOW_EPILSON 0.00001
#define TRIANGLE_INTERSECTION_EPILSON 0.0000001

#define DIRECTIONAL_LIGHT_COUNT 1

__constant__ CudaCamera camera;
__constant__ DirectionalLight directionalLightArray[DIRECTIONAL_LIGHT_COUNT];

__constant__ int meshDescriptorCount;
__constant__ int meshSegmentCount;

__device__ Tuple operator*(float* matrix, Tuple tuple) { return {(matrix[0] * tuple.x) + (matrix[1] * tuple.y) + (matrix[2] * tuple.z) + (matrix[3] * tuple.w), (matrix[4] * tuple.x) + (matrix[5] * tuple.y) + (matrix[6] * tuple.z) + (matrix[7] * tuple.w), (matrix[8] * tuple.x) + (matrix[9] * tuple.y) + (matrix[10] * tuple.z) + (matrix[11] * tuple.w), (matrix[12] * tuple.x) + (matrix[13] * tuple.y) + (matrix[14] * tuple.z) + (matrix[15] * tuple.w)}; }
__device__ Tuple operator+(Tuple tupleA, Tuple tupleB) { return {tupleA.x + tupleB.x, tupleA.y + tupleB.y, tupleA.z + tupleB.z, tupleA.w + tupleB.w}; }
__device__ Tuple operator-(Tuple tupleA, Tuple tupleB) { return {tupleA.x - tupleB.x, tupleA.y - tupleB.y, tupleA.z - tupleB.z, tupleA.w - tupleB.w}; }
__device__ Tuple operator*(Tuple tuple, float scalar) { return {tuple.x * scalar, tuple.y * scalar, tuple.z * scalar, tuple.w * scalar}; }
__device__ Tuple operator*(float scalar, Tuple tuple) { return {tuple.x * scalar, tuple.y * scalar, tuple.z * scalar, tuple.w * scalar}; }
__device__ Ray d_transform(Ray ray, float* matrix) { return {(matrix * ray.origin), (matrix * ray.direction)}; }
__device__ Tuple d_hadamardProduct(Tuple tupleA, Tuple tupleB) { return {tupleA.x * tupleB.x, tupleA.y * tupleB.y, tupleA.z * tupleB.z, tupleA.w * tupleB.w}; }
__device__ float d_magnitude(Tuple tuple) { return sqrt(tuple.x * tuple.x + tuple.y * tuple.y + tuple.z * tuple.z + tuple.w * tuple.w); }
__device__ Tuple d_normalize(Tuple tuple) { return {tuple.x / d_magnitude(tuple), tuple.y / d_magnitude(tuple), tuple.z / d_magnitude(tuple), tuple.w / d_magnitude(tuple)}; }
__device__ Tuple d_negate(Tuple tuple) { return {-tuple.x, -tuple.y, -tuple.z, -tuple.w}; }
__device__ Tuple d_project(Ray ray, float t) { return ray.origin + (ray.direction * t); }
__device__ float d_dot(Tuple tupleA, Tuple tupleB) { return (tupleA.x * tupleB.x) + (tupleA.y * tupleB.y) + (tupleA.z * tupleB.z) + (tupleA.w * tupleB.w); }
__device__ Tuple d_cross(Tuple tupleA, Tuple tupleB) { return {(tupleA.y * tupleB.z) - (tupleA.z * tupleB.y), (tupleA.z * tupleB.x) - (tupleA.x * tupleB.z), (tupleA.x * tupleB.y) - (tupleA.y * tupleB.x), 1.0f}; }
__device__ Tuple d_reflect(Tuple tuple, Tuple normal) { return tuple - (normal * 2.0f * d_dot(tuple, normal)); }

__device__
int intersectTriangle(float* intersectionMagnitude, MeshDescriptor meshDescriptor, MeshSegment meshSegment, Ray ray) {
  Ray transformedRay = d_transform(ray, meshDescriptor.inverseModelMatrix);

  Tuple edgeB = meshSegment.vertexB - meshSegment.vertexA;
  Tuple edgeC = meshSegment.vertexC - meshSegment.vertexA;

  Tuple h = d_cross(transformedRay.direction, edgeC);
  float a = d_dot(edgeB, h);
  float f = 1.0f / a;
  Tuple s = transformedRay.origin - meshSegment.vertexA;
  float u = f * d_dot(s, h);
  Tuple q = d_cross(s, edgeB);
  float v = f * d_dot(transformedRay.direction, q);
  float t = f * d_dot(edgeC, q);

  int intersecting = (t > TRIANGLE_INTERSECTION_EPILSON && t < 1.0f / TRIANGLE_INTERSECTION_EPILSON) * (a <= -TRIANGLE_INTERSECTION_EPILSON || a >= TRIANGLE_INTERSECTION_EPILSON) * (u >= 0.0f && u <= 1.0f) * (v >= 0.0f && u + v <= 1.0f);
  *intersectionMagnitude = t;

  return intersecting;
}

__device__
Tuple colorFromRay(Ray ray, MeshDescriptor* meshDescriptorArray, MeshSegment* meshSegmentArray) {
  Tuple color = {0.0f, 0.0f, 0.0f, 0.0f};
  int intersectionIndex = -1;
  int intersectionDescriptorIndex = -1;
  float intersectionMagnitude = 0.0f;

  int segmentOffset = 0;
  #pragma unroll
  for (int y = 0; y < meshDescriptorCount; y++) {
    for (int x = segmentOffset; x < segmentOffset + meshDescriptorArray[y].segmentCount; x++) {
      float point;
      int count = intersectTriangle(&point, meshDescriptorArray[y], meshSegmentArray[x], ray);

      intersectionIndex = (x * (count > 0 && (point < intersectionMagnitude || intersectionMagnitude == 0))) + (intersectionIndex * (count <= 0 || (point >= intersectionMagnitude && intersectionMagnitude != 0)));
      intersectionDescriptorIndex = (y * (count > 0 && (point < intersectionMagnitude || intersectionMagnitude == 0))) + (intersectionDescriptorIndex * (count <= 0 || (point >= intersectionMagnitude && intersectionMagnitude != 0)));
      intersectionMagnitude = (point * (count > 0 && (point < intersectionMagnitude || intersectionMagnitude == 0))) + (intersectionMagnitude * (count <= 0 || (point >= intersectionMagnitude && intersectionMagnitude != 0)));
    }

    segmentOffset += meshDescriptorArray[y].segmentCount;
  }

  if (intersectionIndex != -1) {
    Ray transformedRay = d_transform(ray, meshDescriptorArray[intersectionDescriptorIndex].inverseModelMatrix);
    Tuple intersectionPoint = d_project(transformedRay, intersectionMagnitude - SHADOW_EPILSON);
    Ray lightRay = {meshDescriptorArray[intersectionDescriptorIndex].modelMatrix * intersectionPoint, d_normalize(directionalLightArray[0].position - (meshDescriptorArray[intersectionDescriptorIndex].modelMatrix * intersectionPoint))};

    int intersecionCount = 0;
    segmentOffset = 0;

    #pragma unroll
    for (int y = 0; y < meshDescriptorCount; y++) {
      for (int x = segmentOffset; x < segmentOffset + meshDescriptorArray[y].segmentCount; x++) {
        float point = 0;
        intersecionCount += intersectTriangle(&point, meshDescriptorArray[y], meshSegmentArray[x], lightRay) * (point < d_magnitude(directionalLightArray[0].position - intersectionPoint));
      }

      segmentOffset += meshDescriptorArray[y].segmentCount;
    }

    Tuple normal = d_normalize(meshDescriptorArray[intersectionDescriptorIndex].modelMatrix * meshSegmentArray[intersectionIndex].normal);
    float lightNormalDifference = d_dot(normal, lightRay.direction);

    Tuple viewDirection = d_normalize((camera.modelMatrix * camera.position) - lightRay.origin);
    Tuple reflectDirection = d_reflect(d_negate(lightRay.direction), normal);
    float reflectedViewDifference = pow(d_dot(viewDirection, reflectDirection), 32);

    color = (0.1f * meshSegmentArray[intersectionIndex].color) + 
            (0.85f * lightNormalDifference * meshSegmentArray[intersectionIndex].color * (lightNormalDifference > 0) * (intersecionCount == 0)) +
            (0.5f * reflectedViewDifference * meshSegmentArray[intersectionIndex].color * (reflectedViewDifference > 0) * (intersecionCount == 0));
  }

  return color;
}

  // float specularIntensity = 0.5;
  // vec3 viewDirection = normalize(u_viewPosition - fragmentPosition);
  // vec3 reflectDirection = reflect(-lightDirection, normal);
  // float specularLight = pow(max(dot(viewDirection, reflectDirection), 0.0), 32);
  // vec3 specular = specularIntensity * specularLight * u_lightColor;

__device__
Ray rayFromReflection(Ray ray, MeshDescriptor* meshDescriptorArray, MeshSegment* meshSegmentArray, int recursionCount = 0) {
  int intersectionIndex = -1;
  int intersectionDescriptorIndex = -1;
  float intersectionMagnitude = 0.0f;

  int segmentOffset = 0;
  #pragma unroll
  for (int y = 0; y < meshDescriptorCount; y++) {
    for (int x = segmentOffset; x < segmentOffset + meshDescriptorArray[y].segmentCount; x++) {
      float point;
      int count = intersectTriangle(&point, meshDescriptorArray[y], meshSegmentArray[x], ray);

      intersectionIndex = (x * (count > 0 && (point < intersectionMagnitude || intersectionMagnitude == 0))) + (intersectionIndex * (count <= 0 || (point >= intersectionMagnitude && intersectionMagnitude != 0)));
      intersectionDescriptorIndex = (y * (count > 0 && (point < intersectionMagnitude || intersectionMagnitude == 0))) + (intersectionDescriptorIndex * (count <= 0 || (point >= intersectionMagnitude && intersectionMagnitude != 0)));
      intersectionMagnitude = (point * (count > 0 && (point < intersectionMagnitude || intersectionMagnitude == 0))) + (intersectionMagnitude * (count <= 0 || (point >= intersectionMagnitude && intersectionMagnitude != 0)));
    }

    segmentOffset += meshDescriptorArray[y].segmentCount;
  }

  Ray reflectedRay = ray;
  if (intersectionDescriptorIndex != -1 && meshDescriptorArray[intersectionDescriptorIndex].reflective) {
    Ray transformedRay = d_transform(ray, meshDescriptorArray[intersectionDescriptorIndex].inverseModelMatrix);
    Tuple intersectionPoint = d_project(transformedRay, intersectionMagnitude - REFLECTIVE_RAY_EPILSON);
    Tuple normal = d_normalize(meshDescriptorArray[intersectionDescriptorIndex].modelMatrix * meshSegmentArray[intersectionIndex].normal);

    reflectedRay = {meshDescriptorArray[intersectionDescriptorIndex].modelMatrix * intersectionPoint, d_reflect(ray.direction, normal)};
  }

  return reflectedRay;
}

__global__
void lighting(Tuple* colorOut, MeshDescriptor* meshDescriptorArray, MeshSegment* meshSegmentArray, int renderWidth, int renderHeight) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (idx >= renderWidth || idy >= renderHeight) { return; }

  Tuple pixel = {
    (idx - (renderWidth / 2.0f)) / renderWidth, 
    (idy - (renderHeight / 2.0f)) / renderHeight, 
    0.0f, 1.0f
  };
  Tuple direction = d_normalize((pixel + camera.direction) - camera.position);
  Ray ray = {camera.position, direction};
  ray = d_transform(ray, camera.modelMatrix);

  colorOut[(idy*renderWidth)+idx] = colorFromRay(ray, meshDescriptorArray, meshSegmentArray);
}

__global__
void reflections(Tuple* colorOut, MeshDescriptor* meshDescriptorArray, MeshSegment* meshSegmentArray, int renderWidth, int renderHeight) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (idx >= renderWidth || idy >= renderHeight) { return; }

  Tuple pixel = {
    (idx - (renderWidth / 2.0f)) / renderWidth, 
    (idy - (renderHeight / 2.0f)) / renderHeight, 
    0.0f, 1.0f
  };
  Tuple direction = d_normalize((pixel + camera.direction) - camera.position);
  Ray ray = {camera.position, direction};
  ray = d_transform(ray, camera.modelMatrix);

  colorOut[(idy*renderWidth)+idx] = colorFromRay(rayFromReflection(ray, meshDescriptorArray, meshSegmentArray), meshDescriptorArray, meshSegmentArray);
}

void writeColorDataToFile(const char* filename, int imageWidth, int imageHeight, unsigned int* colorData) {
  uint8_t* pixels = new uint8_t[imageWidth * imageHeight * 4];
  for (int x = 0; x < imageWidth * imageHeight; x++) {
    pixels[(x * 4)] = int(colorData[x] & 0x0000FF);
    pixels[(x * 4) + 1] = int((colorData[x] & 0x00FF00) >> 8);
    pixels[(x * 4) + 2] = int((colorData[x] & 0xFF0000) >> 16);
    pixels[(x * 4) + 3] = 255;
  }

  stbi_write_png(filename, imageWidth, imageHeight, 4, pixels, imageWidth * 4 * sizeof(uint8_t));
  delete [] pixels;
}

__global__
void combineLightingReflectionBuffers(unsigned int* d_colorBuffer, Tuple* d_lightingBuffer, Tuple* d_reflectionsBuffer, int renderWidth, int renderHeight) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (idx >= renderWidth || idy >= renderHeight) { return; }

  Tuple color;
  if (d_reflectionsBuffer[(idy*renderWidth)+idx].w > 0) {
    color = (0.2 * d_reflectionsBuffer[(idy*renderWidth)+idx]) + d_lightingBuffer[(idy*renderWidth)+idx];
  }
  else {
    color = d_lightingBuffer[(idy*renderWidth)+idx];
  }

  d_colorBuffer[(idy*renderWidth)+idx] = (int(fmaxf(0, fminf(255, color.z))) << 16) | (int(fmaxf(0, fminf(255, color.y))) << 8) | (int(fmaxf(0, fminf(255, color.x))));
}

__global__
void getClosestHitDescriptorKernel(int* result, MeshDescriptor* d_meshDescriptorBuffer, MeshSegment* d_meshSegmentBuffer) {
  Tuple pixel = {0.0f, 0.0f, 0.0f, 1.0f};
  Tuple direction = d_normalize((pixel + camera.direction) - camera.position);
  Ray ray = {camera.position, direction};
  ray = d_transform(ray, camera.modelMatrix);

  int intersectionDescriptorIndex = -1;
  float intersectionMagnitude = 0.0f;

  int segmentOffset = 0;
  #pragma unroll
  for (int y = 0; y < meshDescriptorCount; y++) {
    for (int x = segmentOffset; x < segmentOffset + d_meshDescriptorBuffer[y].segmentCount; x++) {
      float point;
      int count = intersectTriangle(&point, d_meshDescriptorBuffer[y], d_meshSegmentBuffer[x], ray);

      intersectionDescriptorIndex = (y * (count > 0 && (point < intersectionMagnitude || intersectionMagnitude == 0))) + (intersectionDescriptorIndex * (count <= 0 || (point >= intersectionMagnitude && intersectionMagnitude != 0)));
      intersectionMagnitude = (point * (count > 0 && (point < intersectionMagnitude || intersectionMagnitude == 0))) + (intersectionMagnitude * (count <= 0 || (point >= intersectionMagnitude && intersectionMagnitude != 0)));
    }

    segmentOffset += d_meshDescriptorBuffer[y].segmentCount;
  }

  *result = intersectionDescriptorIndex;
}

extern "C" int getClosestHitDescriptor(MeshDescriptor* d_meshDescriptorBuffer, MeshSegment* d_meshSegmentBuffer) {
  int* h_closestHit = (int*)malloc(sizeof(int));
  int* d_closestHit;
  cudaMalloc((int**)&d_closestHit, sizeof(int));

  getClosestHitDescriptorKernel<<<1, 1>>>(d_closestHit, d_meshDescriptorBuffer, d_meshSegmentBuffer);
  cudaDeviceSynchronize();

  cudaMemcpy(h_closestHit, d_closestHit, sizeof(int), cudaMemcpyDeviceToHost);

  return *h_closestHit;
}

extern "C" void initializeScene(int* h_meshDescriptorCount, int* h_meshSegmentCount) {
  cudaMemcpyToSymbol(meshDescriptorCount, h_meshDescriptorCount, sizeof(int));
  cudaMemcpyToSymbol(meshSegmentCount, h_meshSegmentCount, sizeof(int));
}

extern "C" void updateDirectionalLight(float x, float y, float z, float red, float green, float blue) {
  DirectionalLight h_directionalLightArray[] = {{{x, y, z, 1.0}, {red, green, blue, 1.0}}};
  cudaMemcpyToSymbol(directionalLightArray, h_directionalLightArray, DIRECTIONAL_LIGHT_COUNT*sizeof(DirectionalLight));
}

extern "C" void updateCudaCamera(float x, float y, float z, float pitch, float yaw) {
  CudaCamera h_camera = {{0.0, 0.0, 0.0, 1.0}, {0.0, 0.0, 1.0, 0.0}};
  initializeModelMatrix(h_camera.modelMatrix, multiply(multiply(createTranslateMatrix(x, y, z), createRotationMatrixY(yaw)), createRotationMatrixX(pitch)));
  initializeInverseModelMatrix(h_camera.inverseModelMatrix, h_camera.modelMatrix);
  cudaMemcpyToSymbol(camera, &h_camera, sizeof(CudaCamera));
}

extern "C" void updateScene() {

}

extern "C" void renderFrame(int blockDimX, int blockDimY, void* d_colorBuffer, cudaGraphicsResource_t* cudaTextureResource, int frameWidth, int frameHeight, Tuple* d_lightingBuffer, Tuple* d_reflectionsBuffer, MeshDescriptor* d_meshDescriptorBuffer, MeshSegment* d_meshSegmentBuffer) {
  dim3 block(blockDimX, blockDimY);
  dim3 grid((frameWidth + block.x - 1) / block.x, (frameHeight + block.y - 1) / block.y);
  lighting<<<grid, block>>>(d_lightingBuffer, d_meshDescriptorBuffer, d_meshSegmentBuffer, frameWidth, frameHeight);
  reflections<<<grid, block>>>(d_reflectionsBuffer, d_meshDescriptorBuffer, d_meshSegmentBuffer, frameWidth, frameHeight);
  combineLightingReflectionBuffers<<<grid, block>>>((unsigned int*)d_colorBuffer, d_lightingBuffer, d_reflectionsBuffer, frameWidth, frameHeight);

  cudaArray *texture_ptr;
  cudaGraphicsMapResources(1, cudaTextureResource, 0);
  cudaGraphicsSubResourceGetMappedArray(&texture_ptr, *cudaTextureResource, 0, 0);

  cudaMemcpy2DToArray(texture_ptr, 0, 0,  d_colorBuffer, frameWidth*4*sizeof(GLubyte), frameWidth*4*sizeof(GLubyte), frameHeight, cudaMemcpyDeviceToDevice);
  cudaGraphicsUnmapResources(1, cudaTextureResource, 0);
}

extern "C" void renderImage(int blockDimX, int blockDimY, const char* filename, int imageWidth, int imageHeight, MeshDescriptor* d_meshDescriptorBuffer, MeshSegment* d_meshSegmentBuffer) {
  printf("\n");

  Analysis::setAbsoluteStart();
  Analysis::createLabel(0, "allocate_memory");
  Analysis::createLabel(1, "execute_kernel");
  Analysis::createLabel(2, "copy_device");
  Analysis::createLabel(3, "create_image");

  Analysis::begin();
  unsigned int* h_imageData = (unsigned int*)malloc(imageWidth*imageHeight*4*sizeof(GLubyte));
  unsigned int* d_imageData;
  cudaMalloc((unsigned int**)&d_imageData, imageWidth*imageHeight*4*sizeof(GLubyte));

  Tuple *d_lightingData, *d_reflectionsData;
  cudaMalloc((Tuple**)&d_lightingData, imageWidth*imageHeight*sizeof(Tuple));
  cudaMalloc((Tuple**)&d_reflectionsData, imageWidth*imageHeight*sizeof(Tuple));
  Analysis::end(0);

  dim3 block(blockDimX, blockDimY);
  dim3 grid((imageWidth + block.x - 1) / block.x, (imageHeight + block.y - 1) / block.y);

  Analysis::begin();
  printf("rendering ray traced image...\n");
  lighting<<<grid, block>>>(d_lightingData, d_meshDescriptorBuffer, d_meshSegmentBuffer, imageWidth, imageHeight);
  reflections<<<grid, block>>>(d_reflectionsData, d_meshDescriptorBuffer, d_meshSegmentBuffer, imageWidth, imageHeight);
  combineLightingReflectionBuffers<<<grid, block>>>(d_imageData, d_lightingData, d_reflectionsData, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  printf("finished rendering\n");
  Analysis::end(1);

  Analysis::begin();
  cudaMemcpy(h_imageData, d_imageData, imageWidth*imageHeight*4*sizeof(GLubyte), cudaMemcpyDeviceToHost);
  Analysis::end(2);

  Analysis::begin();
  writeColorDataToFile(filename, imageWidth, imageHeight, h_imageData);
  printf("saved image as: [%s]\n", filename);
  Analysis::end(3);

  Analysis::printAll(imageWidth, imageHeight);

  free(h_imageData);
  cudaFree(d_imageData);
  cudaFree(d_lightingData);
  cudaFree(d_reflectionsData);
}