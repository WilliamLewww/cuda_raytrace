#pragma once
#include <math.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "input.h"
#include "raytrace_rectangle.h"
#include "character_rectangle.h"

class Joiner {
private:
  RaytraceRectangle* raytraceRectangle;
  CharacterRectangle* characterRectangle;
public:
  void initialize(GLuint* shaderProgramHandle);
  void update();
  void render();
};