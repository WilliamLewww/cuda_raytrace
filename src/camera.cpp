#include "camera.h"

Camera::Camera() {
  position = {5.0, -3.5, -6.0, 1.0};
  pitch = -M_PI / 12.0;
  yaw = -M_PI / 4.5;

  up = glm::vec3(0.0f, -1.0f, 0.0f);
  projectionMatrix = glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 100.0f);

  getViewMatrix();
}

Camera::~Camera() {

}

Tuple Camera::getPosition() {
  return position;
}

float Camera::getPitch() {
  return pitch;
}

float Camera::getYaw() {
  return yaw;
}

float* Camera::getViewMatrix() {
  front = glm::vec3(cos(pitch) * cos(-yaw - (M_PI / 2)), sin(pitch), cos(pitch) * sin(-yaw - (M_PI / 2)));
  front = glm::normalize(front);

  glm::vec3 glmPosition = glm::vec3(position.x, position.y, position.z);
  viewMatrix = glm::lookAt(glmPosition, glmPosition - front, up);

  return glm::value_ptr(viewMatrix);
}

float* Camera::getProjectionMatrix() {
  return glm::value_ptr(projectionMatrix);
}

void Camera::update() {
  if (abs(Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_X)) > 0.08) {
    position.x += cos(-yaw) * Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_X) * 0.05;
    position.z += sin(-yaw) * Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_X) * 0.05;
  }
  if (abs(Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_Y)) > 0.08) {
    position.x += cos(-yaw + (M_PI / 2)) * Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_Y) * -0.05;
    position.z += sin(-yaw + (M_PI / 2)) * Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_Y) * -0.05;
  }

  if (abs(Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_X)) > 0.08) {
    yaw += Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_X) * 0.05;
  }

  if (abs(Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_Y)) > 0.08) {
    pitch += Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_Y) * -0.05;
  }

  if (Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_TRIGGER) > -0.92) {
    position.y += (Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_TRIGGER) + 1.0) * -0.03;
  }

  if (Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER) > -0.92) {
    position.y += (Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER) + 1.0) * 0.03;
  }
}