#include "joiner.h"

Joiner::Joiner(ShaderHandler* shaderHandler, FontHandler* fontHandler, ModelHandler* modelHandler) {
  camera = new Camera();

  modelHandler->addModel("res/cube.obj", 1);
  modelHandler->addModel("res/donut.obj", 0);
  modelHandler->updateTransformation(0, 0.0, 0.0, 0.0, 5.0, 0.15, 5.0, 0.0, 0.0, 0.0);
  modelHandler->updateTransformation(1, 0.0, -2.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0);

  rasterContainer = new RasterContainer(shaderHandler, fontHandler, modelHandler);
  raytraceContainer = new RaytraceContainer(shaderHandler, fontHandler, modelHandler);

  modelLoaderContainer = new ModelLoaderContainer(shaderHandler, fontHandler);

  renderMode = RENDERMODE_MODELLOADER;
}

Joiner::~Joiner() {
  delete modelLoaderContainer;
  delete raytraceContainer;
  delete rasterContainer;
}

void Joiner::update(float deltaTime) {
  camera->update(deltaTime);

  if (Input::checkSquarePressed()) {
    if (renderMode == RENDERMODE_RASTER) {
      renderMode = RENDERMODE_RAYTRACE;
    }
    else {
      renderMode = RENDERMODE_RASTER;
    }
  }

  if (renderMode == RENDERMODE_RASTER) {
    rasterContainer->update(deltaTime, camera);
  }

  if (renderMode == RENDERMODE_RAYTRACE) {
    raytraceContainer->update(camera);
  }

  if (renderMode == RENDERMODE_MODELLOADER) {
    modelLoaderContainer->update();
  }
}

void Joiner::render() {
  if (renderMode == RENDERMODE_RASTER) {
    rasterContainer->render(camera);
  }
  
  if (renderMode == RENDERMODE_RAYTRACE) {
    raytraceContainer->render();
  }

  if (renderMode == RENDERMODE_MODELLOADER) {
    modelLoaderContainer->render();
  }
}