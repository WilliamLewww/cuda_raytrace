#pragma once
#include "input.h"
#include "model_handler.h"
#include "shader_handler.h"
#include "raytrace_rectangle.h"
#include "model_container.h"
#include "text_container.h"

class RaytraceContainer {
private:  
  RaytraceRectangle* raytraceRectangle;
  TextContainer* textContainer;
public:
  RaytraceContainer(ShaderHandler* shaderHandler, FontHandler* fontHandler, ModelContainer* modelContainer);
  ~RaytraceContainer();

  void update(Camera* camera);
  void render();
};