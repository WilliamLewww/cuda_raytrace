#include "raster_container.h"

RasterContainer::RasterContainer(ShaderHandler* shaderHandler, FontHandler* fontHandler, ModelHandler* modelHandler) {
  rasterModel = modelHandler->createRasterModel(shaderHandler->getShaderFromName("colored_model"), 1);

  Tuple eye = {0.0, 0.0, 0.0, 1.0};
  Tuple target = {0.0, 0.0, 1.0, 1.0};
  Tuple up = {0.0, 1.0, 0.0, 1.0};

  Tuple diff = {eye.x - target.x, eye.y - target.y, eye.z - target.z, eye.w - target.w};

  Tuple z = normalize(diff);
  Tuple x = normalize(cross(up, z));
  Tuple y = cross(z, x); 

  viewMatrix[0] = x.x; viewMatrix[1] = x.y; viewMatrix[2] = x.z;  viewMatrix[3] = -dot(x, eye);
  viewMatrix[4] = y.x; viewMatrix[5] = y.y; viewMatrix[6] = y.z;  viewMatrix[7] = -dot(y, eye);
  viewMatrix[8] = z.x; viewMatrix[9] = z.y; viewMatrix[10] = z.z; viewMatrix[11] = -dot(z, eye);
  viewMatrix[12] = 0;  viewMatrix[13] = 0;  viewMatrix[14] = 0;   viewMatrix[15] = 1;

  float scale = (1.0 / tan((90.0 / 2.0) * (M_PI / 180.0)));
  float near = 0.01;
  float far = 10.0;

  float a = -(far / (far - near));
  float b = -((far * near) / (far - near));

  projectionMatrix[0] =  scale;  projectionMatrix[1] =  0;  projectionMatrix[2] =  0;  projectionMatrix[3] =  0;
  projectionMatrix[4] =  0;  projectionMatrix[5] =  scale;  projectionMatrix[6] =  0;  projectionMatrix[7] =  0;
  projectionMatrix[8] =  0;  projectionMatrix[9] =  0;  projectionMatrix[10] = a;  projectionMatrix[11] = -1;
  projectionMatrix[12] = 0;  projectionMatrix[13] = 0;  projectionMatrix[14] = b;  projectionMatrix[15] = 0;
}

RasterContainer::~RasterContainer() {
  delete rasterModel;
}

void RasterContainer::update() {

}

void RasterContainer::render() {
  rasterModel->render(viewMatrix, projectionMatrix);
}