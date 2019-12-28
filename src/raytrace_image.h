#pragma once
#include <math.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "input.h"

class RaytraceImage {
private:
  float cameraPositionX, cameraPositionY, cameraPositionZ;
  float cameraRotationX, cameraRotationY;

  struct cudaGraphicsResource* cudaTextureResource;
  void* cudaBuffer;

  bool shouldTakePhoto;
public:
  ~RaytraceImage();
  
  void initialize();
  void update();
  void render();

  void updateResolution(int width, int height, GLuint textureResource);
};