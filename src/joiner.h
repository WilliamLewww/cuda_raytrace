#pragma once
#include <math.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "input.h"

class Joiner {
private:
  GLuint* shaderProgramHandle;

  struct cudaGraphicsResource* cudaTextureResource;
  GLuint textureResource;
  void* cudaBuffer;

  GLuint vao, vbo[2];
  GLuint textureHandle;
public:
  void initialize(GLuint* shaderProgramHandle);
  void update();
  void render();
};