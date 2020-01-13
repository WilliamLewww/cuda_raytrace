#include "raytrace_container.h"

RaytraceContainer::RaytraceContainer(ShaderHandler* shaderHandler, FontHandler* fontHandler, ModelHandler* modelHandler) {
  shouldDecreaseImageResolution = false;
  shouldIncreaseImageResolution = false;

  raytraceRectangle = new RaytraceRectangle(shaderHandler->getShaderFromName("textured_rectangle"), modelHandler);

  std::string resolutionString = std::to_string(raytraceRectangle->getImageResolution()) + "x" + std::to_string(raytraceRectangle->getImageResolution());
  textContainer = new TextContainer(shaderHandler->getShaderFromName("textured_rectangle"), fontHandler->getFontFromName("Ubuntu"), resolutionString, -0.95, 0.85);
}

RaytraceContainer::~RaytraceContainer() {
  delete textContainer;
  delete raytraceRectangle;
}

void RaytraceContainer::update(Camera* camera) {
  if (Input::checkGamepadButtonDown(GLFW_GAMEPAD_BUTTON_TRIANGLE) && !shouldIncreaseImageResolution) {
    shouldIncreaseImageResolution = true;
  }

  if (!Input::checkGamepadButtonDown(GLFW_GAMEPAD_BUTTON_TRIANGLE) && shouldIncreaseImageResolution) {
    raytraceRectangle->incrementResolution();
    std::string resolutionString = std::to_string(raytraceRectangle->getImageResolution()) + "x" + std::to_string(raytraceRectangle->getImageResolution());
    textContainer->changeText(resolutionString);
    shouldIncreaseImageResolution = false;
  }

  if (Input::checkGamepadButtonDown(GLFW_GAMEPAD_BUTTON_CROSS) && !shouldDecreaseImageResolution) {
    shouldDecreaseImageResolution = true;
  }

  if (!Input::checkGamepadButtonDown(GLFW_GAMEPAD_BUTTON_CROSS) && shouldDecreaseImageResolution) {
    raytraceRectangle->decrementResolution();
    std::string resolutionString = std::to_string(raytraceRectangle->getImageResolution()) + "x" + std::to_string(raytraceRectangle->getImageResolution());
    textContainer->changeText(resolutionString);
    shouldDecreaseImageResolution = false;
  }

  raytraceRectangle->update(camera);
}

void RaytraceContainer::render() {
  raytraceRectangle->render();
  textContainer->render();
}