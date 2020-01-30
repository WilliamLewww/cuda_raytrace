#include "model_loader_container.h"

extern "C" {
  void initializeScene(int* h_meshDescriptorCount, int* h_meshSegmentCount);
}

ModelLoaderContainer::ModelLoaderContainer(ShaderHandler* shaderHandler, FontHandler* fontHandler, ModelContainer* masterModelContainer, Camera* masterCamera) {
  isAddingModel = true;

  this->masterModelContainer = masterModelContainer;
  this->masterCamera = masterCamera;

  this->shaderHandler = shaderHandler;

  camera = new Camera();
  camera->setPosition(0.0, 0.0, -5.0);
  camera->setPitch(0.0);
  camera->setYaw(0.0);

  textContainer = new TextContainer(shaderHandler->getShaderFromName("textured_rectangle"), fontHandler->getFontFromName("Ubuntu"), "Model Loader", 0.0, 0.0);
  modelContainer = new ModelContainer();

  modelBackgroundRectangle = new ColoredRectangle(shaderHandler->getShaderFromName("colored_rectangle"), 0.0, 0.0, 1000.0, 1000.0, 0.2, 0.2, 0.2);
  upRectangle = new ColoredRectangle(shaderHandler->getShaderFromName("colored_rectangle"), 0.0, 825.0, 150.0, 50.0, 0.8, 0.2, 0.2);
  downRectangle = new ColoredRectangle(shaderHandler->getShaderFromName("colored_rectangle"), 0.0, 25.0, 150.0, 50.0, 0.2, 0.8, 0.2);
  applyRectangle = new ColoredRectangle(shaderHandler->getShaderFromName("colored_rectangle"), 875.0, 25.0, 100.0, 100.0, 0.2, 0.8, 0.2);
  cancelRectangle = new ColoredRectangle(shaderHandler->getShaderFromName("colored_rectangle"), 765.0, 25.0, 100.0, 100.0, 0.8, 0.2, 0.2);

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

  selectedModelClone = nullptr;
  selectedRasterModelClone = nullptr;

  loadModels();
  selectModel(modelContainer->getModel(0));
}

ModelLoaderContainer::~ModelLoaderContainer() {
  delete selectedRasterModelClone;
  delete selectedModelClone;
  delete downRectangle;
  delete upRectangle;
  delete applyRectangle;
  delete cancelRectangle;
  delete modelBackgroundRectangle;
  delete modelContainer;
  delete textContainer;
}

bool ModelLoaderContainer::checkAddingModel() {
  return isAddingModel;
}

void ModelLoaderContainer::update(float deltaTime) {
  if (selectedModelClone != nullptr) {
    float scaleX = 0.0, scaleY = 0.0, scaleZ = 0.0;
    float pitch = 0.0, yaw = 0.0, roll = 0.0;

    if (abs(Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_X)) > 0.08) {
      yaw +=  Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_X) * (deltaTime * 2);
    }

    if (abs(Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_Y)) > 0.08) {
      pitch += Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_Y) * -(deltaTime * 2);
    }

    if (Input::checkGamepadButtonDown(GLFW_GAMEPAD_BUTTON_LEFT_BUMPER)) {
      scaleX += (deltaTime * 2);
      scaleY += (deltaTime * 2);
      scaleZ += (deltaTime * 2);
    }

    if (Input::checkGamepadButtonDown(GLFW_GAMEPAD_BUTTON_RIGHT_BUMPER)) {
      scaleX += -(deltaTime * 2);
      scaleY += -(deltaTime * 2);
      scaleZ += -(deltaTime * 2);
    }

    selectedModelClone->addTransformation(0.0, 0.0, 0.0, scaleX, scaleY, scaleZ, pitch, yaw, roll);
  }
  for (int x = 0; x < modelContainer->getSize(); x++) {
    modelContainer->addTransformation(x, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, M_PI / 12.0 * deltaTime, 0.0);
  }

  if (Input::checkLeftClick()) {
    double cursorPositionX = Input::getCursorPositionX();
    double cursorPositionY = Input::getCursorPositionY();

    if (cursorPositionX >= 0 && cursorPositionX <= 150) {
      if (loadedModelLowerBounds > 0 && cursorPositionY >= 125 && cursorPositionY <= 175) {
        loadedModelLowerBounds -= 5;
        loadedModelUpperBounds -= 5;

        loadModels();
      }

      if (loadedModelUpperBounds < modelNameList.size() && cursorPositionY >= 925 && cursorPositionY <= 975) {
        loadedModelLowerBounds += 5;
        loadedModelUpperBounds += 5;

        loadModels();
      }

      if (cursorPositionY > 175 && cursorPositionY < 925) {
        int index = int(cursorPositionY - 175) / 150;
        if (index < modelContainer->getSize()) {
          selectModel(modelContainer->getModel(index));
        }
      }
    }

    if (cursorPositionY >= 875 && cursorPositionY <= 975) {
      if (cursorPositionX >= 765 && cursorPositionX <= 865) {
        isAddingModel = false;
      }
      if (cursorPositionX >= 875 && cursorPositionX <= 975) {
        Tuple modelPosition = masterCamera->getPosition();
        modelPosition.x += cos(-masterCamera->getYaw() + (M_PI / 2)) * 5.0;
        modelPosition.z += sin(-masterCamera->getYaw() + (M_PI / 2)) * 5.0;

        selectedModelClone->updateTransformation(modelPosition.x, modelPosition.y, modelPosition.z, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0);
        masterModelContainer->emplaceModel(RASTERMODELTYPE_RANDOM_PHONG, shaderHandler->getShaderFromName("random_colored_phong_model"), selectedModelClone);
        selectedModelClone->updateTransformation(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0);

        masterModelContainer->updateDeviceMesh();
        initializeScene(masterModelContainer->getHostMeshDescriptorCount(), masterModelContainer->getHostMeshSegmentCount());
        isAddingModel = false;
      }
    }
  }
}

void ModelLoaderContainer::loadModels() {
  isAddingModel = true;
  modelContainer->deleteAllModels();

  for (int x = loadedModelLowerBounds; x < std::min(loadedModelUpperBounds, int(modelNameList.size())); x++) {
    modelContainer->emplaceModel(RASTERMODELTYPE_RANDOM, shaderHandler->getShaderFromName("random_colored_model"), modelNameList[x].c_str(), 0);
    modelContainer->addTransformation(x - loadedModelLowerBounds, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, M_PI / 8.0, 0.0, 0.0);
  }
}

void ModelLoaderContainer::selectModel(Model* model) {
  if (selectedModelClone != nullptr) {
    delete selectedModelClone;
  }

  if (selectedRasterModelClone != nullptr) {
    delete selectedRasterModelClone;
  }

  selectedModelClone = ModelHandler::createModel(model);
  selectedRasterModelClone = ModelHandler::createRasterModel(RASTERMODELTYPE_RANDOM, shaderHandler->getShaderFromName("random_colored_model"), selectedModelClone);
}

void ModelLoaderContainer::render(DirectionalLight* directionalLight) {
  glViewport(0, 0, 1000, 1000);
  if (selectedModelClone != nullptr) {
    selectedRasterModelClone->render(camera, directionalLight);
  }

  for (int x = 0; x < modelContainer->getSize(); x++) {
    glViewport(0, 675 - (x * 150), 150, 150);
    modelBackgroundRectangle->render();
    modelContainer->getRasterModel(x)->render(camera, directionalLight);
  }

  glViewport(0, 0, 1000, 1000);
  upRectangle->render();
  downRectangle->render();
  applyRectangle->render();
  cancelRectangle->render();
  textContainer->render();
}