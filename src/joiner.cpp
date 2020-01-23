#include "joiner.h"

Joiner::Joiner(ShaderHandler* shaderHandler, FontHandler* fontHandler) {
  camera = new Camera();
  modelContainer = new ModelContainer();

  modelContainer->emplaceModel(shaderHandler->getShaderFromName("random_colored_model"), "res/cube.obj", 1);
  modelContainer->emplaceModel(shaderHandler->getShaderFromName("random_colored_model"), "res/donut.obj", 0);
  modelContainer->updateTransformation(0, 0.0, 0.0, 0.0, 5.0, 0.15, 5.0, 0.0, 0.0, 0.0);
  modelContainer->updateTransformation(1, 0.0, -2.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0);

  rasterContainer = new RasterContainer(shaderHandler, fontHandler, modelContainer);
  raytraceContainer = new RaytraceContainer(shaderHandler, fontHandler, modelContainer);
  modelLoaderContainer = new ModelLoaderContainer(shaderHandler, fontHandler, modelContainer);

  renderMode = RENDERMODE_MODELLOADER;
}

Joiner::~Joiner() {
  delete modelLoaderContainer;
  delete raytraceContainer;
  delete rasterContainer;
  delete modelContainer;
  delete camera;
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
    modelLoaderContainer->update(deltaTime);
    if (!modelLoaderContainer->checkAddingModel()) {
      renderMode = RENDERMODE_RASTER;
    }
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