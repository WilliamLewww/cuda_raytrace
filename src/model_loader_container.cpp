#include "model_loader_container.h"

ModelLoaderContainer::ModelLoaderContainer(ShaderHandler* shaderHandler, FontHandler* fontHandler) {
  textContainer = new TextContainer(shaderHandler->getShaderFromName("textured_rectangle"), fontHandler->getFontFromName("Ubuntu"), "Model Loader", -0.95, 0.85);
  modelHandler = new ModelHandler();

  DIR* directory;
  struct dirent *directoryEntry;
  if ((directory = opendir("res/")) != NULL) {
    while ((directoryEntry = readdir(directory)) != NULL) {
      std::string filename = directoryEntry->d_name;
      filename = "res/" + filename;
      if (filename.find(".obj") != std::string::npos) {
        modelNameList.push_back(filename);
      }
    }
    closedir(directory);
  }
}

ModelLoaderContainer::~ModelLoaderContainer() {
  delete modelHandler;
  delete textContainer;
}

void ModelLoaderContainer::update() {

}

void ModelLoaderContainer::render() {
  textContainer->render();
}