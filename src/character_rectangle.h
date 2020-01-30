#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "common_definitions.h"
#include "font_handler.h"

class CharacterRectangle {
private:
  Font* font;
  Character* character;

  GLuint* shaderProgramHandle;
  GLuint textureResource;

  GLuint vao, vbo[2];

  GLuint textureLocationHandle;
  GLuint resolutionLocationHandle;

  GLfloat vertices[12];
  GLfloat textureCoordinates[12];
public:
  CharacterRectangle(GLuint* shaderProgramHandle, Font* font, const char symbol, float positionX, float positionY);
  ~CharacterRectangle();

  float getOffsetX();

  void render();
};