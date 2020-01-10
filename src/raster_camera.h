#pragma once
#include <math.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "input.h"
#include "raytrace_structures.h"

class RasterCamera {
private:
  glm::vec3 position;
  glm::vec3 front;
  glm::vec3 up;
  float pitch;
  float yaw;

  glm::mat4 viewMatrix;
  glm::mat4 projectionMatrix;
public:
  RasterCamera();
  ~RasterCamera();

  float* getViewMatrix();
  float* getProjectionMatrix();

  void update();
};