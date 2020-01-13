#pragma once
#include "model_handler.h"
#include "shader_handler.h"
#include "text_container.h"
#include "raster_model.h"
#include "raster_camera.h"

class RasterContainer {
private:
  RasterCamera* rasterCamera;
  std::vector<RasterModel*> rasterModelList;
public:
  RasterContainer(ShaderHandler* shaderHandler, FontHandler* fontHandler, ModelHandler* modelHandler);
  ~RasterContainer();

  void update();
  void render();
};