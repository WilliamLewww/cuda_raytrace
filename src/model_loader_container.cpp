#include "model_loader_container.h"

ModelLoaderContainer::ModelLoaderContainer(ShaderHandler* shaderHandler, FontHandler* fontHandler) {
  camera = new Camera();
  camera->setPosition(0.0, 0.0, -5.0);
  camera->setPitch(0.0);
  camera->setYaw(0.0);

  textContainer = new TextContainer(shaderHandler->getShaderFromName("textured_rectangle"), fontHandler->getFontFromName("Ubuntu"), "Model Loader", -0.95, 0.85);
  modelContainer = new ModelContainer();

  modelBackgroundRectangle = new ColoredRectangle(shaderHandler->getShaderFromName("colored_rectangle"), 0.0, 0.0, 1000.0, 1000.0, 0.2, 0.2, 0.2);
  upRectangle = new ColoredRectangle(shaderHandler->getShaderFromName("colored_rectangle"), 0.0, 825.0, 150.0, 50.0, 0.8, 0.2, 0.2);
  downRectangle = new ColoredRectangle(shaderHandler->getShaderFromName("colored_rectangle"), 0.0, 25.0, 150.0, 50.0, 0.2, 0.8, 0.2);

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

  loadedModelLowerBounds = 0;
  loadedModelUpperBounds = 5;

  for (int x = loadedModelLowerBounds; x < std::min(loadedModelUpperBounds, int(modelNameList.size())); x++) {
    modelContainer->emplaceModel(shaderHandler->getShaderFromName("random_colored_model"), modelNameList[x].c_str(), 1);
    modelContainer->addTransformation(x, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, M_PI / 8.0, 0.0, 0.0);
  }
}

ModelLoaderContainer::~ModelLoaderContainer() {
  delete downRectangle;
  delete upRectangle;
  delete modelBackgroundRectangle;
  delete modelContainer;
  delete textContainer;
}

void ModelLoaderContainer::update(float deltaTime) {
  for (int x = loadedModelLowerBounds; x < std::min(loadedModelUpperBounds, int(modelNameList.size())); x++) {
    modelContainer->addTransformation(x, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, M_PI / 12.0 * deltaTime, 0.0);
  }
}

void ModelLoaderContainer::render() {
  for (int x = 0; x < modelContainer->getSize(); x++) {
    glViewport(0, 675 - (x * 150), 150, 150);
    modelBackgroundRectangle->render();
    modelContainer->getRasterModel(x)->render(camera->getViewMatrix(), camera->getProjectionMatrix());
  }

  glViewport(0, 0, 1000, 1000);
  upRectangle->render();
  downRectangle->render();
  textContainer->render();
}