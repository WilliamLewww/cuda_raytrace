#include "raster_camera.h"

RasterCamera::RasterCamera() {
  float scale = (1.0 / tan((90.0 / 2.0) * (M_PI / 180.0)));
  float near = 0.01;
  float far = 10.0;

  float a = -(far / (far - near));
  float b = -((far * near) / (far - near));

  projectionMatrix[0] =  scale;  projectionMatrix[1] =  0;  projectionMatrix[2] =  0;  projectionMatrix[3] =  0;
  projectionMatrix[4] =  0;  projectionMatrix[5] =  scale;  projectionMatrix[6] =  0;  projectionMatrix[7] =  0;
  projectionMatrix[8] =  0;  projectionMatrix[9] =  0;  projectionMatrix[10] = a;  projectionMatrix[11] = -1;
  projectionMatrix[12] = 0;  projectionMatrix[13] = 0;  projectionMatrix[14] = b;  projectionMatrix[15] = 0;

  position = {0.0, 0.0, 0.0, 1.0};
  pitch = 0.0;
  yaw = 0.0;

  updateViewMatrix();
}

RasterCamera::~RasterCamera() {

}

void RasterCamera::updateViewMatrix() {
  Tuple x = {cos(yaw), 0, -sin(yaw), 0.0};
  Tuple y = {sin(yaw) * sin(pitch), cos(pitch), cos(yaw) * sin(pitch), 0.0};
  Tuple z = {sin(yaw) * cos(pitch), -sin(pitch), cos(pitch) * cos(yaw), 0.0};

  viewMatrix[0] = x.x; viewMatrix[1] = x.y; viewMatrix[2] = x.z;  viewMatrix[3] = -dot(x, position);
  viewMatrix[4] = y.x; viewMatrix[5] = y.y; viewMatrix[6] = y.z;  viewMatrix[7] = -dot(y, position);
  viewMatrix[8] = z.x; viewMatrix[9] = z.y; viewMatrix[10] = z.z; viewMatrix[11] = -dot(z, position);
  viewMatrix[12] = 0;  viewMatrix[13] = 0;  viewMatrix[14] = 0;   viewMatrix[15] = 1;
}

float* RasterCamera::getViewMatrix() {
  return viewMatrix;
}

float* RasterCamera::getProjectionMatrix() {
  return projectionMatrix;
}