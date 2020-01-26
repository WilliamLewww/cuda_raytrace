#pragma once
#include "shader_handler.h"
#include "text_container.h"
#include "camera.h"
#include "raytrace_structures.h"
#include "raster_model.h"
#include "model_handler.h"

class ModelPropertyContainer {
private:
  Camera* camera;
  Model* selectedModel;
  RasterModel* selectedRasterModel;

  float* modelMatrix;
  float rotationX;
  float rotationY;

  TextContainer* textContainer;
public:
  ModelPropertyContainer(Model* selectedModel, ShaderHandler* shaderHandler, FontHandler* fontHandler);
  ~ModelPropertyContainer();

  void update(float deltaTime);
  void render(DirectionalLight* directionalLight);
};