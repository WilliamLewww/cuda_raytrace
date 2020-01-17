#include "joiner.h"

Joiner::Joiner(ShaderHandler* shaderHandler, FontHandler* fontHandler, ModelHandler* modelHandler) {
  camera = new Camera();

  modelHandler->addModel("res/cube.obj", 1);
  modelHandler->addModel("res/torus.obj", 0);
  modelHandler->addModel(modelHandler->createReducedModel(1));
  modelHandler->setModelMatrix(0, createScaleMatrix(5.0, 0.15, 5.0));
  modelHandler->setModelMatrix(1, createTranslateMatrix(0.0, -2.0, 0.0));
  modelHandler->setModelMatrix(2, createTranslateMatrix(1.0, -2.0, 1.0));

  rasterContainer = new RasterContainer(shaderHandler, fontHandler, modelHandler);
  raytraceContainer = new RaytraceContainer(shaderHandler, fontHandler, modelHandler);

  renderMode = 0;
}

Joiner::~Joiner() {
  delete raytraceContainer;
  delete rasterContainer;
}

void Joiner::update(float deltaTime) {
  camera->update(deltaTime);

  if (Input::checkSquarePressed()) {
    renderMode += 1;
    
    if (renderMode > 1) {
      renderMode = 0;
    }
  }

  if (renderMode == 0) {
    rasterContainer->update();
  }

  if (renderMode == 1) {
    raytraceContainer->update(camera);
  }
}

void Joiner::render() {
  if (renderMode == 0) {
    rasterContainer->render(camera);
  }
  
  if (renderMode == 1) {
    raytraceContainer->render();
  }
}