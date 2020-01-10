#include "raster_container.h"

RasterContainer::RasterContainer(ShaderHandler* shaderHandler, FontHandler* fontHandler, ModelHandler* modelHandler) {
  rasterCamera = new RasterCamera();
  rasterModel = modelHandler->createRasterModel(shaderHandler->getShaderFromName("colored_model"), 1);
}

RasterContainer::~RasterContainer() {
  delete rasterModel;
  delete rasterCamera;
}

void RasterContainer::update() {
  rasterCamera->update();
}

void RasterContainer::render() {
  rasterModel->render(rasterCamera->getViewMatrix(), rasterCamera->getProjectionMatrix());
}