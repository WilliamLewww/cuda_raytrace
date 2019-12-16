#pragma once
#include <fstream>
#include <string>
#include <math.h>
#include <vector>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "input.h"
#include "joiner.h"

class Engine {
private:
  GLFWwindow* window;
  GLuint shaderProgramHandle;

  Input* input;
  Joiner* joiner;

  void update();
  void render();

  std::string readShaderSource(const char* filepath);
  GLuint createShaderProgram(std::string vertexShaderString, std::string fragmentShaderString);
public:
  void initialize();
  void run();
  void exit();
};