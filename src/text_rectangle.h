#pragma once
#include <vector>
#include <string>

#include "character_rectangle.h"

class TextRectangle {
private:
  std::vector<CharacterRectangle> characterRectangleList;
public:
  void initialize(GLuint* shaderProgramHandle, std::string text, float positionX, float positionY);

  void update();
  void render();
};