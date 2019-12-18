#include "joiner.h"

void Joiner::initialize(GLuint* shaderProgramHandle) {
  rectangle = new RaytraceRectangle();
  rectangle->initialize(shaderProgramHandle);
}

void Joiner::update() {
  rectangle->update();
}

void Joiner::render() {
  rectangle->render();
}