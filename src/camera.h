#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "input.h"
#include "raytrace_structures.h"

class Camera {
protected:
  Tuple position;
  float pitch;
  float yaw;

  glm::vec3 front;
  glm::vec3 up;

  glm::mat4 viewMatrix;
  glm::mat4 projectionMatrix;

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