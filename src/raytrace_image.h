#pragma once
#include <vector>
#include <math.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "raytrace_structures.h"
#include "model.h"
#include "input.h"

class RaytraceImage {
private:
  int frameWidth, frameHeight;
  int imageWidth, imageHeight;

  float cameraPositionX, cameraPositionY, cameraPositionZ;
  float cameraRotationX, cameraRotationY;

  std::vector<Model> modelList;
  int h_meshDescriptorCount, h_meshSegmentCount;

  MeshDescriptor* meshDescriptorBuffer;
  MeshSegment* meshSegmentBuffer;

  struct cudaGraphicsResource* cudaTextureResource;

  Tuple* lightingBuffer;
  Tuple* reflectionsBuffer;
  void* colorBuffer;

  bool shouldTakePhoto;
public:  
  void initialize();
  void update();
  void render();

  void updateResolution(int width, int height, GLuint textureResource);
};