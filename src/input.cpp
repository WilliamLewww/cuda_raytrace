#include "input.h"

std::vector<int> Input::keyDownList;

void Input::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
  if (action == GLFW_PRESS) {
    keyDownList.push_back(key);
  }

  if (action == GLFW_RELEASE) {
    for (int x = 0; x < keyDownList.size(); x++) {
      if (key == keyDownList[x]) {
        keyDownList.erase(keyDownList.begin() + x);
      }
    }
  }
}

bool Input::checkKeyDown(int key) {
  for (int x = 0; x < keyDownList.size(); x++) {
    if (key == keyDownList[x]) {
      return true;
    }
  }

  return false;
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