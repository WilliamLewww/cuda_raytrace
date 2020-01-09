#include "joiner.h"

Joiner::Joiner(ShaderHandler* shaderHandler, FontHandler* fontHandler, ModelHandler* modelHandler) {
  modelHandler->addModel("res/cube.obj", 1);
  modelHandler->addModel("res/donut.obj", 0);
  modelHandler->addModel(modelHandler->createReducedModel(1));
  modelHandler->removeModel(1);

  modelHandler->setModelMatrix(0, createScaleMatrix(5.0, 0.15, 5.0));
  modelHandler->setModelMatrix(1, createTranslateMatrix(0.0, -2.0, 0.0));

  raytraceContainer = new RaytraceContainer(shaderHandler, fontHandler, modelHandler);
}

Joiner::~Joiner() {
  delete raytraceContainer;
}

void Joiner::update() {
  raytraceContainer->update();
}

void Joiner::render() {
  raytraceContainer->render();
}