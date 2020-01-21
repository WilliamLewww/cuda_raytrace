#pragma once
#include <vector>
#include <string>
#include <dirent.h>

#include "shader_handler.h"
#include "model_handler.h"
#include "font_handler.h"
#include "text_container.h"

class ModelLoaderContainer {
private:
  TextContainer* textContainer;
  ModelHandler* modelHandler;

  std::vector<std::string> modelNameList;
public:
  ModelLoaderContainer(ShaderHandler* shaderHandler, FontHandler* fontHandler);
  ~ModelLoaderContainer();

  void update();
  void render();
};