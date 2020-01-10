#include "raster_container.h"

RasterContainer::RasterContainer(ShaderHandler* shaderHandler, FontHandler* fontHandler, ModelHandler* modelHandler) {
  rasterModel = modelHandler->createRasterModel(shaderHandler->getShaderFromName("colored_model"), 1);
}

RasterContainer::~RasterContainer() {
  delete rasterModel;
}

void RasterContainer::update() {

}

void RasterContainer::render() {
  rasterModel->render();
}