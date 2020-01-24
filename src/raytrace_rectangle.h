#pragma once
#include <math.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "raytrace_image.h"

class RaytraceRectangle {
private:
  int imageResolution;

  RaytraceImage* image;

  GLuint* shaderProgramHandle;
  GLuint textureResource;
  
  GLuint vao, vbo[2];
  GLuint textureHandle;

  GLfloat vertices[12];
  GLfloat textureCoordinates[12];

  void initializeImage(int width, int height);
public:
  RaytraceRectangle(GLuint* shaderProgramHandle, ModelContainer* modelContainer);
  ~RaytraceRectangle();

  int getImageResolution();
  void incrementResolution();
  void decrementResolution();
  
  void update(Camera* camera, DirectionalLight* directionalLight);
  void render();
};