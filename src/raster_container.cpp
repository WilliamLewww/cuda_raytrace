#include "raster_container.h"

RasterContainer::RasterContainer(ShaderHandler* shaderHandler, FontHandler* fontHandler, ModelHandler* modelHandler) {  
  rasterModelList.push_back(modelHandler->createRasterModel(shaderHandler->getShaderFromName("colored_model"), 0));
  rasterModelList.push_back(modelHandler->createRasterModel(shaderHandler->getShaderFromName("colored_model"), 1));
}

RasterContainer::~RasterContainer() {
  for (int x = 0; x < rasterModelList.size(); x++) {
    delete rasterModelList[x];
  }
}

void RasterContainer::update() {
  
}

void RasterContainer::render(Camera* camera) {
  for (int x = 0; x < rasterModelList.size(); x++) {
    rasterModelList[x]->render(camera->getViewMatrix(), camera->getProjectionMatrix());
  }
}