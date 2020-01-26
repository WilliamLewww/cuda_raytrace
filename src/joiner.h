#pragma once
#include "input.h"
#include "shader_handler.h"
#include "font_handler.h"
#include "model_handler.h"
#include "raster_container.h"
#include "raytrace_container.h"
#include "model_container.h"
#include "model_loader_container.h"
#include "model_property_container.h"

enum RenderMode {
  RENDERMODE_RASTER,
  RENDERMODE_RAYTRACE,
  RENDERMODE_MODELLOADER,
  RENDERMODE_PROPERTYCONTAINER,
};

class Joiner {
private:  
  ShaderHandler* shaderHandler;
  FontHandler* fontHandler;

  DirectionalLight* directionalLight;

  Camera* camera;
  ModelContainer* modelContainer;

  RasterContainer* rasterContainer;
  RaytraceContainer* raytraceContainer;

  ModelLoaderContainer* modelLoaderContainer;
  ModelPropertyContainer* modelPropertyContainer;

  RenderMode renderMode;
public:
  Joiner(ShaderHandler* shaderHandler, FontHandler* fontHandler);
  ~Joiner();

  void update(float deltaTime);
  void render();
};