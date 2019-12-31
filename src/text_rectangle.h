#pragma once
#include <vector>
#include <string>

#include "character_rectangle.h"
#include "squiggle_animation_text.h"

class TextRectangle {
private:
  std::vector<CharacterRectangle> characterRectangleList;
  GLuint* shaderProgramHandle;

  float positionX, positionY;

  SquiggleAnimationText squiggleAnimationText;
public:
  void initialize(GLuint* shaderProgramHandle, std::string text, float positionX, float positionY);
  void changeText(std::string text);

  void update();
  void render();
};