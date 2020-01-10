#include "joiner.h"

Joiner::Joiner(ShaderHandler* shaderHandler, FontHandler* fontHandler, ModelHandler* modelHandler) {
  modelHandler->addModel("res/cube.obj", 1);
  modelHandler->addModel("res/donut.obj", 0);
  modelHandler->setModelMatrix(0, createScaleMatrix(5.0, 0.15, 5.0));
  modelHandler->setModelMatrix(1, createTranslateMatrix(0.0, -2.0, 0.0));

  // rasterContainer = new RasterContainer(shaderHandler, fontHandler, modelHandler);
  raytraceContainer = new RaytraceContainer(shaderHandler, fontHandler, modelHandler);
}

Joiner::~Joiner() {
  delete raytraceContainer;
  // delete rasterContainer;
}

void Joiner::update() {
  // rasterContainer->update();
  raytraceContainer->update();
}

void Joiner::render() {
  // rasterContainer->render();
  raytraceContainer->render();
}