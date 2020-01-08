#pragma once
#include <vector>
#include <string>

#include "character_rectangle.h"
#include "squiggle_animation_text.h"

class TextContainer {
private:
  std::vector<CharacterRectangle> characterRectangleList;
  
  GLuint* shaderProgramHandle;
  Font* font;

  float positionX, positionY;

  SquiggleAnimationText squiggleAnimationText;
public:
  TextContainer(GLuint* shaderProgramHandle, Font* font, std::string text, float positionX, float positionY);
  ~TextContainer();

  void changeText(std::string text);

  void update();
  void render();
};