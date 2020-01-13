#include "raster_container.h"

RasterContainer::RasterContainer(ShaderHandler* shaderHandler, FontHandler* fontHandler, ModelHandler* modelHandler) {
  rasterCamera = new RasterCamera();
  
  rasterModelList.push_back(modelHandler->createRasterModel(shaderHandler->getShaderFromName("colored_model"), 0));
  rasterModelList.push_back(modelHandler->createRasterModel(shaderHandler->getShaderFromName("colored_model"), 1));
}

RasterContainer::~RasterContainer() {
  for (int x = 0; x < rasterModelList.size(); x++) {
    delete rasterModelList[x];
  }

  delete rasterCamera;
}

void RasterContainer::update() {
  rasterCamera->update();
}

void RasterContainer::render() {
  for (int x = 0; x < rasterModelList.size(); x++) {
    rasterModelList[x]->render(rasterCamera->getViewMatrix(), rasterCamera->getProjectionMatrix());
  }
}