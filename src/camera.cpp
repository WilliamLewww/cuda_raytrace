#include "camera.h"

#include <stdio.h>
Camera::Camera() {
  position = {5.0, -3.5, -6.0, 1.0};
  pitch = -M_PI / 12.0;
  yaw = -M_PI / 4.5;
}

Camera::~Camera() {

}

void Camera::update() {
  if (abs(Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_X)) > 0.08) {
    position.x += cos(yaw + (M_PI / 2)) * Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_X) * 0.0005;
    position.z += sin(yaw + (M_PI / 2)) * Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_X) * 0.0005;
  }
  if (abs(Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_Y)) > 0.08) {
    position.x += cos(yaw) * Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_Y) * 0.0005;
    position.z += sin(yaw) * Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_Y) * 0.0005;
  }

  if (abs(Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_X)) > 0.08) {
    yaw += Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_X) * -0.0005;
  }

  if (abs(Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_Y)) > 0.08) {
    pitch += Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_Y) * -0.0005;
  }

  if (Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_TRIGGER) > -0.92) {
    position.y += (Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_TRIGGER) + 1.0) * -0.00003;
  }

  if (Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER) > -0.92) {
    position.y += (Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER) + 1.0) * 0.00003;
  }
}