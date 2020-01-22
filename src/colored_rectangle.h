#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>

class ColoredRectangle {
private:
  GLuint* shaderProgramHandle;

  float red, green, blue;

  GLuint vao, vbo[1];
  GLuint colorLocationHandle;
  GLfloat vertices[12];
public:
  ColoredRectangle(GLuint* shaderProgramHandle, float red, float green, float blue);
  ~ColoredRectangle();

  void render();
};