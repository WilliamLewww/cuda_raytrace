#include "raster_container.h"

RasterContainer::RasterContainer(ShaderHandler* shaderHandler, FontHandler* fontHandler, ModelHandler* modelHandler) {  
  rasterModelList.push_back(modelHandler->createRasterModel(shaderHandler->getShaderFromName("colored_model"), 0));
  rasterModelList.push_back(modelHandler->createRasterModel(shaderHandler->getShaderFromName("colored_model"), 1));

  textContainer = new TextContainer(shaderHandler->getShaderFromName("textured_rectangle"), fontHandler->getFontFromName("Ubuntu"), "Raster", -0.95, 0.85);
}

RasterContainer::~RasterContainer() {
  delete textContainer;

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

  textContainer->render();
}