#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "common_definitions.h"

class ColoredRectangle {
private:
  GLuint* shaderProgramHandle;

  float positionX, positionY;
  float width, height;

  float red, green, blue;

  GLuint vao, vbo[1];
  
  GLuint colorLocationHandle;
  GLuint resolutionLocationHandle;

  GLfloat vertices[12];
public:
  ColoredRectangle(GLuint* shaderProgramHandle, float positionX, float positionY, float width, float height, float red, float green, float blue);
  ~ColoredRectangle();

  void render();
};