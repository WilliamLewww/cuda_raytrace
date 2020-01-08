#include "joiner.h"

Joiner::Joiner(ShaderHandler* shaderHandler, FontHandler* fontHandler) {
  raytraceRectangle = new RaytraceRectangle(shaderHandler->getShaderFromName("textured_rectangle"));

  std::string resolutionString = std::to_string(raytraceRectangle->getImageResolution()) + "x" + std::to_string(raytraceRectangle->getImageResolution());
  textContainer = new TextContainer(shaderHandler->getShaderFromName("textured_rectangle"), fontHandler->getFontFromName("Ubuntu"), resolutionString, -0.95, 0.85);
}

Joiner::~Joiner() {
  delete textContainer;
  delete raytraceRectangle;
}

void Joiner::update() {
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

  raytraceRectangle->update();
  textContainer->update();
}

void Joiner::render() {
  raytraceRectangle->render();
  textContainer->render();
}