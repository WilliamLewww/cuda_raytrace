#pragma once
#include "input.h"
#include "raytrace_structures.h"

class Camera {
protected:
  Tuple position;
  float pitch;
  float yaw;

  Tuple direction;
  Tuple up;

  float viewMatrix[16];
  float projectionMatrix[16];

  bool isMoving;

  void handleController(float deltaTime);
public:
  Camera();
  ~Camera();

  Tuple getPosition();
  float getPitch();
  float getYaw();

  void setMoving(bool isMoving);
  bool getMoving();

  float* getViewMatrix();
  float* getProjectionMatrix();

  void update(float deltaTime);
};