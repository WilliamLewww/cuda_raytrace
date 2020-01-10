#pragma once
#include "input.h"
#include "model_handler.h"
#include "shader_handler.h"
#include "text_container.h"
#include "raster_model.h"

class RasterContainer {
private:
  RasterModel* rasterModel;

  float viewMatrix[16];
  float projectionMatrix[16];
public:
  RasterContainer(ShaderHandler* shaderHandler, FontHandler* fontHandler, ModelHandler* modelHandler);
  ~RasterContainer();

  void update();
  void render();
};