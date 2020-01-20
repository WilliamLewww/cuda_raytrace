#pragma once
#include <cuda_runtime.h>

#include "model_handler.h"
#include "shader_handler.h"
#include "text_container.h"
#include "raster_model.h"
#include "camera.h"

class RasterContainer {
private:
  ModelHandler* modelHandler;
  ShaderHandler* shaderHandler;
  
  Model* selectedModel;

  std::vector<RasterModel*> rasterModelList;
  TextContainer* textContainer;
public:
  RasterContainer(ShaderHandler* shaderHandler, FontHandler* fontHandler, ModelHandler* modelHandler);
  ~RasterContainer();

  void update(float deltaTime,Camera* camera);
  void render(Camera* camera);
};