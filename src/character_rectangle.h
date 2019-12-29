#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "font.h"

class CharacterRectangle {
private:
  GLuint* shaderProgramHandle;
  GLuint textureResource;

  GLuint vao, vbo[2];
  GLuint textureHandle;

  GLfloat vertices[12];
  GLfloat textureCoordinates[12];
public:
  void initialize(GLuint* shaderProgramHandle, char symbol);
  void render();
};