#include <algorithm>
#include <fstream>
#include <stdio.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "structures.h"
#include "model.h"
#include "analysis.h"

#define FRAME_WIDTH 1000
#define FRAME_HEIGHT 1000

#define IMAGE_WIDTH 5000
#define IMAGE_HEIGHT 5000

#define LIGHT_COUNT 1

#define MESH_DESCRIPTOR_COUNT 2
#define MESH_SEGMENT_COUNT 24

#define REFLECTIVE_RAY_EPILSON 0.0001
#define SHADOW_EPILSON 0.00001
#define TRIANGLE_INTERSECTION_EPILSON 0.0000001

__constant__ Camera camera[1];

__constant__ Light lightArray[LIGHT_COUNT];

__constant__ MeshDescriptor meshDescriptorArray[MESH_DESCRIPTOR_COUNT];
__constant__ MeshSegment meshSegmentArray[MESH_SEGMENT_COUNT];

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
int intersectTriangle(float* intersectionMagnitude, MeshDescriptor meshDescriptor, MeshSegment meshSegment, Ray ray) {
  Ray transformedRay = transform(ray, meshDescriptor.inverseModelMatrix);

  Tuple edgeB = meshSegment.vertexB - meshSegment.vertexA;
  Tuple edgeC = meshSegment.vertexC - meshSegment.vertexA;

  Tuple h = cross(transformedRay.direction, edgeC);
  float a = dot(edgeB, h);
  float f = 1.0f / a;
  Tuple s = transformedRay.origin - meshSegment.vertexA;
  float u = f * dot(s, h);
  Tuple q = cross(s, edgeB);
  float v = f * dot(transformedRay.direction, q);
  float t = f * dot(edgeC, q);

  int intersecting = (t > TRIANGLE_INTERSECTION_EPILSON && t < 1.0f / TRIANGLE_INTERSECTION_EPILSON) * (a <= -TRIANGLE_INTERSECTION_EPILSON || a >= TRIANGLE_INTERSECTION_EPILSON) * (u >= 0.0f && u <= 1.0f) * (v >= 0.0f && u + v <= 1.0f);
  *intersectionMagnitude = t;

  return intersecting;
}

__device__
Tuple colorFromRay(Ray ray) {
  Tuple color = {0.0f, 0.0f, 0.0f, 0.0f};
  int intersectionIndex = -1;
  int intersectionDescriptorIndex = -1;
  float intersectionMagnitude = 0.0f;

  int segmentOffset = 0;
  #pragma unroll
  for (int y = 0; y < MESH_DESCRIPTOR_COUNT; y++) {
    for (int x = segmentOffset; x < segmentOffset + meshDescriptorArray[y].segmentCount; x++) {
      float point;
      int count = intersectTriangle(&point, meshDescriptorArray[y], meshSegmentArray[x], ray);

      intersectionIndex = (x * (count > 0 && (point < intersectionMagnitude || intersectionMagnitude == 0))) + (intersectionIndex * (count <= 0 || (point >= intersectionMagnitude && intersectionMagnitude != 0)));
      intersectionDescriptorIndex = (y * (count > 0 && (point < intersectionMagnitude || intersectionMagnitude == 0))) + (intersectionDescriptorIndex * (count <= 0 || (point >= intersectionMagnitude && intersectionMagnitude != 0)));
      intersectionMagnitude = (point * (count > 0 && (point < intersectionMagnitude || intersectionMagnitude == 0))) + (intersectionMagnitude * (count <= 0 || (point >= intersectionMagnitude && intersectionMagnitude != 0)));
    }
  }

  if (intersectionIndex != -1) {
    Ray transformedRay = transform(ray, meshDescriptorArray[intersectionDescriptorIndex].inverseModelMatrix);
    Tuple intersectionPoint = project(transformedRay, intersectionMagnitude - SHADOW_EPILSON);
    Ray lightRay = {meshDescriptorArray[intersectionDescriptorIndex].modelMatrix * intersectionPoint, normalize(lightArray[0].position - (meshDescriptorArray[intersectionDescriptorIndex].modelMatrix * intersectionPoint))};

    int intersecionCount = 0;

    segmentOffset = 0;
    #pragma unroll
    for (int y = 0; y < MESH_DESCRIPTOR_COUNT; y++) {
      for (int x = segmentOffset; x < segmentOffset + meshDescriptorArray[y].segmentCount; x++) {
        float point = 0;
        intersecionCount += intersectTriangle(&point, meshDescriptorArray[y], meshSegmentArray[x], lightRay) * (point < magnitude(lightArray[0].position - intersectionPoint));
      }
    }

    float lightNormalDifference = dot(meshSegmentArray[intersectionIndex].normal, lightRay.direction);

    color = (0.1f * meshSegmentArray[intersectionIndex].color) + 
            (0.7f * lightNormalDifference * meshSegmentArray[intersectionIndex].color * (lightNormalDifference > 0) * (intersecionCount == 0));
  }

  return color;
}

__device__
Ray rayFromReflection(Ray ray, int recursionCount = 0) {
  return ray;
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

  MeshDescriptor* h_meshDescriptorArray = new MeshDescriptor[MESH_DESCRIPTOR_COUNT];
  MeshSegment* h_meshSegmentArray = new MeshSegment[MESH_SEGMENT_COUNT];

  Model modelA = createModelFromOBJ("res/cube.obj");
  Model modelB = createModelFromOBJ("res/cube.obj");
  initializeModelMatrix(&modelA.meshDescriptor, createScaleMatrix(5.0, 0.15, 5.0));
  initializeModelMatrix(&modelB.meshDescriptor, createTranslateMatrix(0.0, -2.0, 0.0));

  h_meshDescriptorArray[0] = modelA.meshDescriptor;
  h_meshDescriptorArray[1] = modelB.meshDescriptor;
  memcpy(&h_meshSegmentArray[0], modelA.meshSegmentArray, 12*sizeof(MeshSegment));
  memcpy(&h_meshSegmentArray[12], modelB.meshSegmentArray, 12*sizeof(MeshSegment));

  cudaMemcpyToSymbol(meshDescriptorArray, h_meshDescriptorArray, MESH_DESCRIPTOR_COUNT*sizeof(MeshDescriptor));
  cudaMemcpyToSymbol(meshSegmentArray, h_meshSegmentArray, MESH_SEGMENT_COUNT*sizeof(MeshSegment));
}

extern "C" void updateScene() {

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