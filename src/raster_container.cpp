#include "raster_container.h"

extern "C" {
  int getClosestHitDescriptor(MeshDescriptor* d_meshDescriptorBuffer, MeshSegment* d_meshSegmentBuffer);

  void updateCudaCamera(float x, float y, float z, float pitch, float yaw);
}

RasterContainer::RasterContainer(ShaderHandler* shaderHandler, FontHandler* fontHandler, ModelHandler* modelHandler) { 
  this->modelHandler = modelHandler;

  for (int x = 0; x < modelHandler->getModelListSize(); x++) {
    rasterModelList.push_back(modelHandler->createRasterModel(shaderHandler->getShaderFromName("random_colored_model"), x));
  }

  textContainer = new TextContainer(shaderHandler->getShaderFromName("textured_rectangle"), fontHandler->getFontFromName("Ubuntu"), "Raster", -0.95, 0.85);
}

RasterContainer::~RasterContainer() {
  delete textContainer;

  for (int x = 0; x < rasterModelList.size(); x++) {
    delete rasterModelList[x];
  }
}

void RasterContainer::update(Camera* camera) {
  if (Input::checkCirclePressed()) {
    Tuple cameraPosition = camera->getPosition();
    updateCudaCamera(cameraPosition.x, cameraPosition.y, cameraPosition.z, camera->getPitch(), camera->getYaw());
    
    modelHandler->updateDeviceMesh();

    int closestHitDescriptor = getClosestHitDescriptor(modelHandler->getDeviceMeshDescriptorBuffer(), modelHandler->getDeviceMeshSegmentBuffer());
  }
}

void RasterContainer::render(Camera* camera) {
  for (int x = 0; x < rasterModelList.size(); x++) {
    rasterModelList[x]->render(camera->getViewMatrix(), camera->getProjectionMatrix());
  }

  textContainer->render();
}