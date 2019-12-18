#pragma once
#include <math.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "input.h"

class RaytraceRectangle {
private:
  float cameraPositionX, cameraPositionY, cameraPositionZ;
  float cameraRotationX, cameraRotationY;
  
  GLuint* shaderProgramHandle;

  struct cudaGraphicsResource* cudaTextureResource;
  GLuint textureResource;
  void* cudaBuffer;
  
  GLuint vao, vbo[2];
  GLuint textureHandle;

  GLfloat vertices[12];
  GLfloat textureCoordinates[12];
public:
  void initialize(GLuint* shaderProgramHandle);
  void update();
  void render();
};