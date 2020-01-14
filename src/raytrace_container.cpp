#include "raytrace_container.h"

RaytraceContainer::RaytraceContainer(ShaderHandler* shaderHandler, FontHandler* fontHandler, ModelHandler* modelHandler) {
  raytraceRectangle = new RaytraceRectangle(shaderHandler->getShaderFromName("textured_rectangle"), modelHandler);

  std::string resolutionString = std::to_string(raytraceRectangle->getImageResolution()) + "x" + std::to_string(raytraceRectangle->getImageResolution());
  textContainer = new TextContainer(shaderHandler->getShaderFromName("textured_rectangle"), fontHandler->getFontFromName("Ubuntu"), resolutionString, -0.95, 0.85);
}

RaytraceContainer::~RaytraceContainer() {
  delete textContainer;
  delete raytraceRectangle;
}

void RaytraceContainer::update(Camera* camera) {
  if (Input::checkTrianglePressed()) {
    raytraceRectangle->incrementResolution();
    std::string resolutionString = std::to_string(raytraceRectangle->getImageResolution()) + "x" + std::to_string(raytraceRectangle->getImageResolution());
    textContainer->changeText(resolutionString);
  }

  if (Input::checkCrossPressed()) {
    raytraceRectangle->decrementResolution();
    std::string resolutionString = std::to_string(raytraceRectangle->getImageResolution()) + "x" + std::to_string(raytraceRectangle->getImageResolution());
    textContainer->changeText(resolutionString);
  }

  raytraceRectangle->update(camera);
}

void RaytraceContainer::render() {
  raytraceRectangle->render();
  textContainer->render();
}