#include "joiner.h"

void Joiner::initialize(GLuint* shaderProgramHandle) {
  raytraceRectangle = new RaytraceRectangle();
  raytraceRectangle->initialize(shaderProgramHandle);

  std::string resolutionString = std::to_string(raytraceRectangle->getImageResolution()) + "x" + std::to_string(raytraceRectangle->getImageResolution());

  textRectangle = new TextRectangle();
  textRectangle->initialize(shaderProgramHandle, resolutionString, -0.95, 0.85);
}

void Joiner::update() {
  raytraceRectangle->update();
}

void Joiner::render() {
  raytraceRectangle->render();
  textRectangle->render();
}