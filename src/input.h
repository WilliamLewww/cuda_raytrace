#pragma once
#include <vector>
#include <iostream>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

class Input {
private:
  static std::vector<int> keyDownList;
public:
  static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
  static bool checkKeyDown(int key);
};