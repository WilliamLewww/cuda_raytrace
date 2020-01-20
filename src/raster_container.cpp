#include "raster_container.h"

extern "C" {
  int getClosestHitDescriptor(MeshDescriptor* d_meshDescriptorBuffer, MeshSegment* d_meshSegmentBuffer);

  void initializeScene(int* h_meshDescriptorCount, int* h_meshSegmentCount);

  void updateCudaCamera(float x, float y, float z, float pitch, float yaw);
}

RasterContainer::RasterContainer(ShaderHandler* shaderHandler, FontHandler* fontHandler, ModelHandler* modelHandler) { 
  this->shaderHandler = shaderHandler;

  this->modelHandler = modelHandler;
  selectedModel = nullptr;

  for (int x = 0; x < modelHandler->getModelListSize(); x++) {
    rasterModelList.push_back(modelHandler->createRasterModel(shaderHandler->getShaderFromName("random_colored_model"), x));
  }

  textContainer = new TextContainer(shaderHandler->getShaderFromName("textured_rectangle"), fontHandler->getFontFromName("Ubuntu"), "Raster", -0.95, 0.85);
}

RasterContainer::~RasterContainer() {
  delete textContainer;

  for (int x = 0; x < rasterModelList.size(); x++) {
    delete rasterModelList[x];
  }
}

void RasterContainer::update(float deltaTime, Camera* camera) {
  if (Input::checkCirclePressed()) {
    if (selectedModel == nullptr) {
      Tuple cameraPosition = camera->getPosition();
      updateCudaCamera(cameraPosition.x, cameraPosition.y, cameraPosition.z, camera->getPitch(), camera->getYaw());

      int closestHitDescriptor = getClosestHitDescriptor(modelHandler->getDeviceMeshDescriptorBuffer(), modelHandler->getDeviceMeshSegmentBuffer());
      if (closestHitDescriptor != -1) {
        selectedModel = modelHandler->getModel(closestHitDescriptor);
        camera->setMoving(false);
      }
    }
    else {
      selectedModel = nullptr;
      camera->setMoving(true);
    }

    modelHandler->updateDeviceMesh();
  }

  if (selectedModel != nullptr) {
    float positionX = 0.0, positionY = 0.0, positionZ = 0.0;
    float scaleX = 0.0, scaleY = 0.0, scaleZ = 0.0;
    float pitch = 0.0, yaw = 0.0, roll = 0.0;

    if (abs(Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_X)) > 0.08) {
      positionX += cos(-camera->getYaw()) * Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_X) * (deltaTime * 2);
      positionZ += sin(-camera->getYaw()) * Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_X) * (deltaTime * 2);
    }

    if (abs(Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_Y)) > 0.08) {
      positionX += cos(-camera->getYaw() + (M_PI / 2)) * Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_Y) * -(deltaTime * 2);
      positionZ += sin(-camera->getYaw() + (M_PI / 2)) * Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_Y) * -(deltaTime * 2);
    }

    if (Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_TRIGGER) > -0.92) {
      positionY += (Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_TRIGGER) + 1.0) * -(deltaTime * 2);
    }

    if (Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER) > -0.92) {
      positionY += (Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER) + 1.0) * (deltaTime * 2);
    }

    if (abs(Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_X)) > 0.08) {
      roll += cos(camera->getYaw()) * Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_X) * (deltaTime * 2);
      pitch += sin(camera->getYaw()) * Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_X) * (deltaTime * 2);
    }

    if (abs(Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_Y)) > 0.08) {
      roll += cos(camera->getYaw() + (M_PI / 2)) * Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_Y) * (deltaTime * 2);
      pitch += sin(camera->getYaw() + (M_PI / 2)) * Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_Y) * (deltaTime * 2);
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

    if (Input::checkTrianglePressed()) {
      modelHandler->addModel(modelHandler->createModel(selectedModel));
      rasterModelList.push_back(modelHandler->createRasterModel(shaderHandler->getShaderFromName("random_colored_model"), modelHandler->getModelListSize() - 1));

      modelHandler->updateDeviceMesh();
      initializeScene(modelHandler->getHostMeshDescriptorCount(), modelHandler->getHostMeshSegmentCount());
    }

    if (Input::checkCrossPressed()) {
      modelHandler->addModel(selectedModel->createReducedModel());
      int index = modelHandler->getIndexFromAddress(selectedModel);

      modelHandler->removeModel(index);
      rasterModelList.erase(rasterModelList.begin() + index);

      rasterModelList.push_back(modelHandler->createRasterModel(shaderHandler->getShaderFromName("random_colored_model"), modelHandler->getModelListSize() - 1));
      selectedModel = modelHandler->getModel(modelHandler->getModelListSize() - 1);

      modelHandler->updateDeviceMesh();
      initializeScene(modelHandler->getHostMeshDescriptorCount(), modelHandler->getHostMeshSegmentCount());
    }

    selectedModel->addTransformation(positionX, positionY, positionZ, scaleX, scaleY, scaleZ, pitch, yaw, roll);
  }
}

void RasterContainer::render(Camera* camera) {
  for (int x = 0; x < rasterModelList.size(); x++) {
    rasterModelList[x]->render(camera->getViewMatrix(), camera->getProjectionMatrix());
  }

  textContainer->render();
}