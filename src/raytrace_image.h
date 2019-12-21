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

  bool alreadyMadeImage;
public:
  void initialize(GLuint textureResource);
  void update();
  void render();
};