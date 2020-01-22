#pragma once
#include <vector>
#include <string>
#include <dirent.h>

#include "input.h"
#include "camera.h"
#include "shader_handler.h"
#include "font_handler.h"
#include "text_container.h"
#include "model_container.h"
#include "colored_rectangle.h"

class ModelLoaderContainer {
private:
  int test;
  ShaderHandler* shaderHandler;

  Camera* camera;
  TextContainer* textContainer;
  ModelContainer* modelContainer;

  Model* selectedModelClone;
  RasterModel* selectedRasterModelClone;

  ColoredRectangle* modelBackgroundRectangle;
  ColoredRectangle *upRectangle, *downRectangle;

  int loadedModelLowerBounds;
  int loadedModelUpperBounds;
  std::vector<std::string> modelNameList;

  void loadModels();
  void selectModel(Model* model);
public:
  ModelLoaderContainer(ShaderHandler* shaderHandler, FontHandler* fontHandler);
  ~ModelLoaderContainer();

  void update(float deltaTime);
  void render();
};