#pragma once
#include "raytrace_structures.h"

class Camera {
protected:
  Tuple position;
  float pitch;
  float yaw;
public:
  Camera();
  ~Camera();
};