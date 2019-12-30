#include "joiner.h"

void Joiner::initialize(GLuint* shaderProgramHandle) {
  raytraceRectangle = new RaytraceRectangle();
  raytraceRectangle->initialize(shaderProgramHandle);

  textRectangle = new TextRectangle();
  textRectangle->initialize(shaderProgramHandle, "abcdefghijklmnopqrstuvwxyz");
}

void Joiner::update() {
  raytraceRectangle->update();
}

void Joiner::render() {
  raytraceRectangle->render();
  textRectangle->render();
}