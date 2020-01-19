#include "raster_container.h"

extern "C" {
  int getClosestHitDescriptor(MeshDescriptor* d_meshDescriptorBuffer, MeshSegment* d_meshSegmentBuffer);

  void updateCudaCamera(float x, float y, float z, float pitch, float yaw);
}

RasterContainer::RasterContainer(ShaderHandler* shaderHandler, FontHandler* fontHandler, ModelHandler* modelHandler) { 
  this->modelHandler = modelHandler;

  for (int x = 0; x < modelHandler->getModelListSize(); x++) {
    rasterModelList.push_back(modelHandler->createRasterModel(shaderHandler->getShaderFromName("random_colored_model"), x));
  }

  textContainer = new TextContainer(shaderHandler->getShaderFromName("textured_rectangle"), fontHandler->getFontFromName("Ubuntu"), "Raster", -0.95, 0.85);

  selectedModel = nullptr;
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
      
      modelHandler->updateDeviceMesh();

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
  }

  if (selectedModel != nullptr) {
    float positionX = 0.0, positionY = 0.0, positionZ = 0.0;
    float pitch = 0.0, yaw = 0.0;

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
      yaw += Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_X) * (deltaTime * 2);
    }

    if (abs(Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_Y)) > 0.08) {
      pitch += Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_Y) * -(deltaTime * 2);
    }

    selectedModel->setModelMatrix(multiply(selectedModel->getModelMatrix(), createTranslateMatrix(positionX, positionY, positionZ)));
    selectedModel->setModelMatrix(multiply(selectedModel->getModelMatrix(), createRotationMatrixY(yaw)));
    selectedModel->setModelMatrix(multiply(selectedModel->getModelMatrix(), createRotationMatrixX(pitch)));
  }
}

void RasterContainer::render(Camera* camera) {
  for (int x = 0; x < rasterModelList.size(); x++) {
    rasterModelList[x]->render(camera->getViewMatrix(), camera->getProjectionMatrix());
  }

  textContainer->render();
}