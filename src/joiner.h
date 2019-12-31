#pragma once
#include <math.h>
#include <string>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "input.h"
#include "raytrace_rectangle.h"
#include "text_rectangle.h"

class Joiner {
private:
  bool shouldDecreaseImageResolution, shouldIncreaseImageResolution;
  
  RaytraceRectangle* raytraceRectangle;
  TextRectangle* textRectangle;
public:
  void initialize(GLuint* shaderProgramHandle);
  void update();
  void render();
};