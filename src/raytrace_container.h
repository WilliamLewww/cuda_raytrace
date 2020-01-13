#pragma once
#include "input.h"
#include "model_handler.h"
#include "shader_handler.h"
#include "raytrace_rectangle.h"
#include "text_container.h"

class RaytraceContainer {
private:
  bool shouldDecreaseImageResolution, shouldIncreaseImageResolution;
  
  RaytraceRectangle* raytraceRectangle;
  TextContainer* textContainer;
public:
  RaytraceContainer(ShaderHandler* shaderHandler, FontHandler* fontHandler, ModelHandler* modelHandler);
  ~RaytraceContainer();

  void update(Camera* camera);
  void render();
};