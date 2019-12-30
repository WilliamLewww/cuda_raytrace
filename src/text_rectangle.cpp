#include "text_rectangle.h"

void TextRectangle::initialize(GLuint* shaderProgramHandle, std::string text) {
  float offsetX = 0.0;
  for (int x = 0; x < text.size(); x++) {
    characterRectangleList.push_back(CharacterRectangle());
    characterRectangleList[characterRectangleList.size() - 1].initialize(shaderProgramHandle, text[x], offsetX);
    offsetX += characterRectangleList[characterRectangleList.size() - 1].getOffsetX();
  }
}

void TextRectangle::render() {
  for (int x = 0; x < characterRectangleList.size(); x++) {
    characterRectangleList[x].render();
  }
}