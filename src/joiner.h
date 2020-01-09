#pragma once
#include "input.h"
#include "shader_handler.h"
#include "font_handler.h"
#include "model_handler.h"
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