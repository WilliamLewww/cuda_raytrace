#include "raster_container.h"

RasterContainer::RasterContainer(ShaderHandler* shaderHandler, FontHandler* fontHandler, ModelHandler* modelHandler) {
  rasterCamera = new RasterCamera();
  rasterModel = modelHandler->createRasterModel(shaderHandler->getShaderFromName("colored_model"), 1);
}

RasterContainer::~RasterContainer() {
  delete rasterModel;
  delete rasterCamera;
}

void RasterContainer::update() {
  if (abs(Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_X)) > 0.08) {

  }
  if (abs(Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_Y)) > 0.08) {

  }
  if (abs(Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_X)) > 0.08) {

  }

  if (abs(Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_Y)) > 0.08) {

  }
}

void RasterContainer::render() {
  rasterModel->render(rasterCamera->getViewMatrix(), rasterCamera->getProjectionMatrix());
}