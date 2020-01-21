#include "camera.h"

Camera::Camera() {
  position = {5.0, -3.5, -6.0, 1.0};
  pitch = -M_PI / 12.0;
  yaw = -M_PI / 4.5;

  initializeModelMatrix(projectionMatrix, createZeroMatrix());
  projectionMatrix[0] = (1.0 / tan((M_PI / 4.0) / 2.0)) / 1.0;
  projectionMatrix[5] = (1.0 / tan((M_PI / 4.0) / 2.0));
  projectionMatrix[10] = (100.0 + 0.01) / (0.01 - 100.0);
  projectionMatrix[11] = (2.0 * 100.0 * 0.01) / (0.01 - 100.0);
  projectionMatrix[14] = -1.0;

  isMoving = true;

  direction = {0.0, 0.0, 0.0, 0.0};
  up = {0.0, -1.0, 0.0, 0.0};

  initializeModelMatrix(viewMatrix, createIdentityMatrix());
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

void Camera::setMoving(bool isMoving) {
  this->isMoving = isMoving;
}

bool Camera::getMoving() {
  return isMoving;
}

float* Camera::getViewMatrix() {
  direction = {cosf(pitch) * cosf(-yaw - (M_PI / 2.0)), sinf(pitch), cosf(pitch) * sinf(-yaw - (M_PI / 2.0)), 0.0};
  direction = normalize(direction);

  Tuple difference = {position.x - direction.x, position.y - direction.y, position.z - direction.z, position.w - direction.w};

  Tuple zaxis = {position.x - difference.x, position.y - difference.y, position.z - difference.z, position.w - difference.w};
  zaxis = normalize(zaxis);

  Tuple xaxis = normalize(cross(normalize(up), zaxis));
  Tuple yaxis = cross(zaxis, xaxis);

  float* translation = createIdentityMatrix();
  translation[3] = -position.x;
  translation[7] = -position.y;
  translation[11] = -position.z;

  float* rotation = createIdentityMatrix();
  rotation[0] = xaxis.x;
  rotation[1] = xaxis.y;
  rotation[2] = xaxis.z;
  rotation[4] = yaxis.x;
  rotation[5] = yaxis.y;
  rotation[6] = yaxis.z;
  rotation[8] = zaxis.x;
  rotation[9] = zaxis.y;
  rotation[10] = zaxis.z;

  initializeModelMatrix(viewMatrix, multiply(rotation, translation));

  return viewMatrix;
}

float* Camera::getProjectionMatrix() {
  return projectionMatrix;
}

void Camera::update(float deltaTime) {
  if (isMoving) {
    handleController(deltaTime);
  }
}

void Camera::handleController(float deltaTime) {
  if (abs(Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_X)) > 0.08) {
    position.x += cos(-yaw) * Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_X) * (deltaTime * 2);
    position.z += sin(-yaw) * Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_X) * (deltaTime * 2);
  }
  if (abs(Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_Y)) > 0.08) {
    position.x += cos(-yaw + (M_PI / 2)) * Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_Y) * -(deltaTime * 2);
    position.z += sin(-yaw + (M_PI / 2)) * Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_Y) * -(deltaTime * 2);
  }

  if (abs(Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_X)) > 0.08) {
    yaw += Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_X) * (deltaTime * 2);
  }

  if (abs(Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_Y)) > 0.08) {
    pitch += Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_Y) * -(deltaTime * 2);
  }

  if (Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_TRIGGER) > -0.92) {
    position.y += (Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_TRIGGER) + 1.0) * -(deltaTime * 2);
  }

  if (Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER) > -0.92) {
    position.y += (Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER) + 1.0) * (deltaTime * 2);
  }
}