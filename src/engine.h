#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "input.h"
#include "font_handler.h"
#include "model_handler.h"
#include "shader_handler.h"
#include "joiner.h"

class Engine {
private:
  GLFWwindow* window;

  FontHandler* fontHandler;
  ShaderHandler* shaderHandler;
  ModelHandler* modelHandler;
  
  Joiner* joiner;

  void update();
  void render();
public:
  Engine();
  ~Engine();

  void run();
};