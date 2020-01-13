#pragma once
#include <vector>
#include <math.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "raytrace_structures.h"
#include "model_handler.h"
#include "model.h"
#include "input.h"

class RaytraceImage {
private:
  int frameWidth, frameHeight;
  int imageWidth, imageHeight;

  float cameraPositionX, cameraPositionY, cameraPositionZ;
  float cameraPitch, cameraYaw;

  std::vector<Model*> modelList;
  int h_meshDescriptorCount, h_meshSegmentCount;

  struct cudaGraphicsResource* cudaTextureResource;

  MeshDescriptor* d_meshDescriptorBuffer;
  MeshSegment* d_meshSegmentBuffer;

  Tuple* d_lightingBuffer;
  Tuple* d_reflectionsBuffer;
  void* d_colorBuffer;

  bool shouldTakePhoto;
public:
  RaytraceImage(ModelHandler* modelHandler);
  ~RaytraceImage();

  void update();
  void render();

  void updateResolution(int width, int height, GLuint textureResource);
};