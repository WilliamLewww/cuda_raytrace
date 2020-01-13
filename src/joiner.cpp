#include "joiner.h"

Joiner::Joiner(ShaderHandler* shaderHandler, FontHandler* fontHandler, ModelHandler* modelHandler) {
  camera = new Camera();

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
  camera->update();

  // rasterContainer->update();
  raytraceContainer->update(camera);
}

void Joiner::render() {
  // rasterContainer->render(camera);
  raytraceContainer->render();
}