#include "input.h"

std::vector<int> Input::keyDownList;
std::vector<int> Input::keyPressList;

bool Input::isLeftClick;

double Input::cursorPositionX;
double Input::cursorPositionY;

bool Input::isSquarePressed;
bool Input::isSquareReleased;
bool Input::isCrossPressed;
bool Input::isCrossReleased;
bool Input::isCirclePressed;
bool Input::isCircleReleased;
bool Input::isTrianglePressed;
bool Input::isTriangleReleased;

void Input::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
  if (action == GLFW_PRESS) {
    keyDownList.push_back(key);
    keyPressList.push_back(key);
  }

  if (action == GLFW_RELEASE) {
    for (int x = 0; x < keyDownList.size(); x++) {
      if (key == keyDownList[x]) {
        keyDownList.erase(keyDownList.begin() + x);
      }
    }
  }
}

void Input::cursorButtonCallback(GLFWwindow* window, int button, int action, int mods) {
  if (action == GLFW_PRESS) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
      isLeftClick = true;
    }
  }
}

void Input::cursorPositionCallback(GLFWwindow* window, double x, double y) {
  cursorPositionX = x;
  cursorPositionY = y;
}

bool Input::checkKeyDown(int key) {
  for (int x = 0; x < keyDownList.size(); x++) {
    if (key == keyDownList[x]) {
      return true;
    }
  }

  return false;
}

bool Input::checkKeyPress(int key) {
  for (int x = 0; x < keyPressList.size(); x++) {
    if (key == keyPressList[x]) {
      return true;
    }
  }

  return false;
}

bool Input::checkLeftClick() {
  return isLeftClick;
}

double Input::getCursorPositionX() {
  return cursorPositionX;
}

double Input::getCursorPositionY() {
  return cursorPositionY;
}

void Input::checkControllerPresses() {
  if (checkGamepadButtonDown(GLFW_GAMEPAD_BUTTON_SQUARE)) {
    if (isSquareReleased) {
      isSquarePressed = true;
    }
    isSquareReleased = false;
  }
  else {
    isSquareReleased = true;
  }

  if (checkGamepadButtonDown(GLFW_GAMEPAD_BUTTON_CROSS)) {
    if (isCrossReleased) {
      isCrossPressed = true;
    }
    isCrossReleased = false;
  }
  else {
    isCrossReleased = true;
  }

  if (checkGamepadButtonDown(GLFW_GAMEPAD_BUTTON_CIRCLE)) {
    if (isCircleReleased) {
      isCirclePressed = true;
    }
    isCircleReleased = false;
  }
  else {
    isCircleReleased = true;
  }

  if (checkGamepadButtonDown(GLFW_GAMEPAD_BUTTON_TRIANGLE)) {
    if (isTriangleReleased) {
      isTrianglePressed = true;
    }
    isTriangleReleased = false;
  }
  else {
    isTriangleReleased = true;
  }
}

bool Input::checkSquarePressed() {
  return isSquarePressed;
}

bool Input::checkCrossPressed() {
  return isCrossPressed;
}

bool Input::checkCirclePressed() {
  return isCirclePressed;
}

bool Input::checkTrianglePressed() {
  return isTrianglePressed;
}

void Input::refreshInput() {
  keyPressList.clear();
  isLeftClick = false;

  isSquarePressed = false;
  isCrossPressed = false;
  isCirclePressed = false;
  isTrianglePressed = false;
}

bool Input::checkGamepadButtonDown(int button) {
  GLFWgamepadstate state;

  if (glfwGetGamepadState(GLFW_JOYSTICK_1, &state)) {
    if (state.buttons[button]) {
      return true;
    }
  }

  return false;
}

float Input::checkGamepadAxis(int axis) {
  GLFWgamepadstate state;

  if (glfwGetGamepadState(GLFW_JOYSTICK_1, &state)) {
    return state.axes[axis];
  }

  return -1.0;
}