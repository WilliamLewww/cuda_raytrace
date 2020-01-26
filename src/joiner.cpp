#include "joiner.h"

Joiner::Joiner(ShaderHandler* shaderHandler, FontHandler* fontHandler) {
  this->shaderHandler = shaderHandler;
  this->fontHandler = fontHandler;

  directionalLight = new DirectionalLight{{10.0, -10.0, -5.0, 1.0}, {1.0, 1.0, 1.0, 1.0}};

  camera = new Camera();
  modelContainer = new ModelContainer();

  modelContainer->emplaceModel(RASTERMODELTYPE_RANDOM_PHONG, shaderHandler->getShaderFromName("random_colored_phong_model"), "res/cube.obj", 1);
  modelContainer->updateTransformation(0, 0.0, 0.0, 0.0, 5.0, 0.15, 5.0, 0.0, 0.0, 0.0);

  rasterContainer = new RasterContainer(shaderHandler, fontHandler, modelContainer);
  raytraceContainer = new RaytraceContainer(shaderHandler, fontHandler, modelContainer);
  modelLoaderContainer = new ModelLoaderContainer(shaderHandler, fontHandler, modelContainer, camera);

  modelPropertyContainer = nullptr;

  renderMode = RENDERMODE_MODELLOADER;
}

Joiner::~Joiner() {
  if (modelPropertyContainer != nullptr) {
    delete modelPropertyContainer;
  }

  delete modelLoaderContainer;
  delete raytraceContainer;
  delete rasterContainer;
  delete modelContainer;
  delete camera;
  delete directionalLight;
}

void Joiner::update(float deltaTime) {
  if (Input::checkSquarePressed()) {
    if (renderMode == RENDERMODE_RASTER && !rasterContainer->checkModelSelected()) {
      renderMode = RENDERMODE_RAYTRACE;
    }
    else {
      renderMode = RENDERMODE_RASTER;
    }
  }

  if (renderMode == RENDERMODE_RASTER) {
    rasterContainer->update(deltaTime, camera);

    if (Input::checkCrossPressed()) {
      if (!rasterContainer->checkModelSelected()) {
        renderMode = RENDERMODE_MODELLOADER;
        modelLoaderContainer->loadModels();
      }
      else {
        modelPropertyContainer = new ModelPropertyContainer(rasterContainer->getSelectedModel(), shaderHandler, fontHandler);
        renderMode = RENDERMODE_PROPERTYCONTAINER;
      }
    }
  }

  if (renderMode == RENDERMODE_RAYTRACE) {
    raytraceContainer->update(camera, directionalLight);
  }

  if (renderMode == RENDERMODE_MODELLOADER) {
    modelLoaderContainer->update(deltaTime);
    if (!modelLoaderContainer->checkAddingModel()) {
      renderMode = RENDERMODE_RASTER;
    }
  } else {
    camera->update(deltaTime);
  }

  if (renderMode == RENDERMODE_PROPERTYCONTAINER) {
    modelPropertyContainer->update();
  }
}

void Joiner::render() {
  if (renderMode == RENDERMODE_RASTER) {
    rasterContainer->render(camera, directionalLight);
  }
  
  if (renderMode == RENDERMODE_RAYTRACE) {
    raytraceContainer->render();
  }

  if (renderMode == RENDERMODE_MODELLOADER) {
    modelLoaderContainer->render(directionalLight);
  }

  if (renderMode == RENDERMODE_PROPERTYCONTAINER) {
    modelPropertyContainer->render(camera, directionalLight);
  }
}