#pragma once
#include <vector>
#include <iostream>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

class Input {
private:
  static std::vector<int> keyDownList;
  static std::vector<int> keyPressList;

  static bool isLeftClick;

  static double cursorPositionX;
  static double cursorPositionY;

  static bool isSquarePressed;
  static bool isSquareReleased;

  static bool isCrossPressed;
  static bool isCrossReleased;

  static bool isCirclePressed;
  static bool isCircleReleased;

  static bool isTrianglePressed;
  static bool isTriangleReleased;

public:
  static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
  static void cursorButtonCallback(GLFWwindow* window, int button, int action, int mods);
  static void cursorPositionCallback(GLFWwindow* window, double x, double y);

  static bool checkKeyDown(int key);
  static bool checkKeyPress(int key);

  static bool checkLeftClick();

  static double getCursorPositionX();
  static double getCursorPositionY();

  static void checkControllerPresses();
  static bool checkSquarePressed();
  static bool checkCrossPressed();
  static bool checkCirclePressed();
  static bool checkTrianglePressed();

  static void refreshInput();

  static bool checkGamepadButtonDown(int button);
  static float checkGamepadAxis(int axis);
};