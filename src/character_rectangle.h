#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "font.h"

class CharacterRectangle {
private:
  Font* font;
  Character* character;

  GLuint* shaderProgramHandle;
  GLuint textureResource;

  GLuint vao, vbo[2];
  GLuint textureHandle;

  GLfloat vertices[12];
  GLfloat textureCoordinates[12];
public:
  float getOffsetX();

  void initialize(GLuint* shaderProgramHandle, const char symbol, float positionX);
  void render();
};