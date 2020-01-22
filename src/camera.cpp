#include "camera.h"

Camera::Camera() {
  position = {5.0, -3.5, -6.0, 1.0};
  pitch = -M_PI / 8.0;
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

void Camera::setPosition(float x, float y, float z) {
  this->position = {x, y, z, 1.0};
}

void Camera::setPitch(float pitch) {
  this->pitch = pitch;
}

void Camera::setYaw(float yaw) {
  this->yaw = yaw;
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

  Tuple positionVector = {-position.x, -position.y, -position.z, 0.0};

  viewMatrix[0] = xaxis.x; viewMatrix[1] = xaxis.y; viewMatrix[2] = xaxis.z; viewMatrix[3] = dot(xaxis, positionVector);
  viewMatrix[4] = yaxis.x; viewMatrix[5] = yaxis.y; viewMatrix[6] = yaxis.z; viewMatrix[7] = dot(yaxis, positionVector);
  viewMatrix[8] = zaxis.x; viewMatrix[9] = zaxis.y; viewMatrix[10] = zaxis.z; viewMatrix[11] = dot(zaxis, positionVector);
  viewMatrix[12] = 0.0; viewMatrix[13] = 0.0; viewMatrix[14] = 0.0; viewMatrix[15] = 1.0;

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