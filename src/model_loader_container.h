#pragma once
#include <vector>
#include <string>
#include <dirent.h>

#include "camera.h"
#include "shader_handler.h"
#include "font_handler.h"
#include "text_container.h"
#include "model_container.h"
#include "colored_rectangle.h"

class ModelLoaderContainer {
private:
  Camera* camera;

  TextContainer* textContainer;
  ModelContainer* modelContainer;

  ColoredRectangle* coloredRectangle;

  int loadedModelLowerBounds;
  int loadedModelUpperBounds;
  std::vector<std::string> modelNameList;
public:
  ModelLoaderContainer(ShaderHandler* shaderHandler, FontHandler* fontHandler);
  ~ModelLoaderContainer();

  void update(float deltaTime);
  void render();
};