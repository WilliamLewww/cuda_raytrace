#include "joiner.h"

void Joiner::initialize(GLuint* shaderProgramHandle) {
  raytraceRectangle = new RaytraceRectangle();
  raytraceRectangle->initialize(shaderProgramHandle);

  characterRectangle = new CharacterRectangle();
  characterRectangle->initialize(shaderProgramHandle, 'c');
}

void Joiner::update() {
  raytraceRectangle->update();
}

void Joiner::render() {
  raytraceRectangle->render();
  characterRectangle->render();
}