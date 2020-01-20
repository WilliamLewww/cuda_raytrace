#pragma once
#include "input.h"
#include "shader_handler.h"
#include "font_handler.h"
#include "model_handler.h"
#include "raster_container.h"
#include "raytrace_container.h"
#include "model_loader_container.h"

class Joiner {
private:
  enum RenderMode {
    RENDERMODE_RASTER,
    RENDERMODE_RAYTRACE,
    RENDERMODE_MODELLOADER,
  };

  Camera* camera;
  RasterContainer* rasterContainer;
  RaytraceContainer* raytraceContainer;

  ModelLoaderContainer* modelLoaderContainer;

  RenderMode renderMode;
public:
  Joiner(ShaderHandler* shaderHandler, FontHandler* fontHandler, ModelHandler* modelHandler);
  ~Joiner();

  void update(float deltaTime);
  void render();
};