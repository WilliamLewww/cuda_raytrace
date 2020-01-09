#pragma once
#include <string>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "input.h"
#include "model_handler.h"
#include "shader_handler.h"
#include "raytrace_container.h"

class Joiner {
private:
  RaytraceContainer* raytraceContainer;
public:
  Joiner(ShaderHandler* shaderHandler, FontHandler* fontHandler, ModelHandler* modelHandler);
  ~Joiner();

  void update();
  void render();
};