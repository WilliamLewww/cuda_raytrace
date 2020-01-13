#include "raster_camera.h"

RasterCamera::RasterCamera() : Camera() {
  up = glm::vec3(0.0f, -1.0f, 0.0f);
  projectionMatrix = glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 100.0f);

  getViewMatrix();
}

RasterCamera::~RasterCamera() {

}

float* RasterCamera::getViewMatrix() {
  front = glm::vec3(cos(pitch) * cos(yaw), sin(pitch), cos(pitch) * sin(yaw));
  front = glm::normalize(front);

  glm::vec3 glmPosition = glm::vec3(position.x, position.y, position.z);
  viewMatrix = glm::lookAt(glmPosition, glmPosition - front, up);

  return glm::value_ptr(viewMatrix);
}

float* RasterCamera::getProjectionMatrix() {
  return glm::value_ptr(projectionMatrix);
}