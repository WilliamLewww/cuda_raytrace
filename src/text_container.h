#pragma once
#include <vector>
#include <string>

#include "character_rectangle.h"

class TextContainer {
private:
  std::vector<CharacterRectangle*> characterRectangleList;
  
  GLuint* shaderProgramHandle;
  Font* font;

  float positionX, positionY;
public:
  TextContainer(GLuint* shaderProgramHandle, Font* font, std::string text, float positionX, float positionY);
  ~TextContainer();

  void changeText(std::string text);

  void render();
};