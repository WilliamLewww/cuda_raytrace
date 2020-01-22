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

  void setPosition(float x, float y, float z);
  void setPitch(float pitch);
  void setYaw(float yaw);

  Tuple getPosition();
  float getPitch();
  float getYaw();

  void setMoving(bool isMoving);
  bool getMoving();

  float* getViewMatrix();
  float* getProjectionMatrix();

  void update(float deltaTime);
};