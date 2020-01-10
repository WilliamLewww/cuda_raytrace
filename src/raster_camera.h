#pragma once
#include <math.h>

#include "input.h"
#include "raytrace_structures.h"

class RasterCamera {
private:
  Tuple position;
  float pitch;
  float yaw;

  float viewMatrix[16];
  float projectionMatrix[16];

  void updateViewMatrix();
public:
  RasterCamera();
  ~RasterCamera();

  float* getViewMatrix();
  float* getProjectionMatrix();

  void update();
};