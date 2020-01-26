#pragma once
#include "shader_handler.h"
#include "text_container.h"
#include "camera.h"
#include "raytrace_structures.h"
#include "raster_model.h"
#include "model_handler.h"

class ModelPropertyContainer {
private:
  Model* selectedModel;
  RasterModel* selectedRasterModel;

  TextContainer* textContainer;
public:
  ModelPropertyContainer(Model* selectedModel, ShaderHandler* shaderHandler, FontHandler* fontHandler);
  ~ModelPropertyContainer();

  void update();
  void render(Camera* camera, DirectionalLight* directionalLight);
};