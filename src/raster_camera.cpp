#include "raster_camera.h"

RasterCamera::RasterCamera() {
  position = glm::vec3(0.0f, 0.0f, 0.0f);
  up = glm::vec3(0.0f, -1.0f, 0.0f);

  pitch = 0.0;
  yaw = 0.0;

  front = glm::vec3(cos(pitch) * cos(yaw), sin(pitch), cos(pitch) * sin(yaw));
  front = glm::normalize(front);
  viewMatrix = glm::lookAt(position, position + front, up);

  projectionMatrix = glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 100.0f);
}

RasterCamera::~RasterCamera() {

}

float* RasterCamera::getViewMatrix() {
  return glm::value_ptr(viewMatrix);
}

float* RasterCamera::getProjectionMatrix() {
  return glm::value_ptr(projectionMatrix);
}

void RasterCamera::update() {
  if (abs(Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_X)) > 0.08) {
    position.x += cos(yaw + (M_PI / 2)) * Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_X) * -0.00005;
    position.z += sin(yaw + (M_PI / 2)) * Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_X) * -0.00005;
  }
  if (abs(Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_Y)) > 0.08) {
    position.x += cos(yaw) * Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_Y) * -0.00005;
    position.z += sin(yaw) * Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_Y) * -0.00005;
  }

  if (abs(Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_X)) > 0.08) {
    yaw += Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_X) * -0.0005;
  }

  if (abs(Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_Y)) > 0.08) {
    pitch += Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_Y) * 0.0005;
  }

  if (Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_TRIGGER) > -0.92) {
    position.y += (Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_TRIGGER) + 1.0) * -0.00003;
  }

  if (Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER) > -0.92) {
    position.y += (Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER) + 1.0) * 0.00003;
  }

  front = glm::vec3(cos(pitch) * cos(yaw), sin(pitch), cos(pitch) * sin(yaw));
  front = glm::normalize(front);
  viewMatrix = glm::lookAt(position, position + front, up);
}