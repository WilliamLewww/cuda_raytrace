#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>

class ColoredRectangle {
private:
  GLuint* shaderProgramHandle;

  GLuint vao, vbo[1];
  GLuint colorHandle;

  GLfloat vertices[12];
public:
  ColoredRectangle(GLuint* shaderProgramHandle, float positionX, float positionY, float width, float height);
  ~ColoredRectangle();

  void render();
};