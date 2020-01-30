#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "input.h"
#include "font_handler.h"
#include "model_handler.h"
#include "shader_handler.h"
#include "common_definitions.h"
#include "joiner.h"

class Engine {
private:
  GLFWwindow* window;

  FontHandler* fontHandler;
  ShaderHandler* shaderHandler;
  
  Joiner* joiner;

  float frameStart;
  float frameEnd;
  float deltaTime;

  void update(float deltaTime);
  void render();
public:
  Engine();
  ~Engine();

  void run();
};