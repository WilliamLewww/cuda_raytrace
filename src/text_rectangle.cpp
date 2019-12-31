#include "text_rectangle.h"

void TextRectangle::initialize(GLuint* shaderProgramHandle, std::string text, float positionX, float positionY) {
  this->shaderProgramHandle = shaderProgramHandle;
  this->positionX = positionX;
  this->positionY = positionY;

  changeText(text);
}

void TextRectangle::changeText(std::string text) {
  characterRectangleList.clear();
  
  float offsetX = 0.0;
  for (int x = 0; x < text.size(); x++) {
    characterRectangleList.push_back(CharacterRectangle());
    characterRectangleList[characterRectangleList.size() - 1].initialize(shaderProgramHandle, text[x], positionX + offsetX, positionY);
    offsetX += characterRectangleList[characterRectangleList.size() - 1].getOffsetX();
  }

  squiggleAnimationText.initialize(&characterRectangleList);
}

void TextRectangle::update() {
  squiggleAnimationText.animate();
}

void TextRectangle::render() {
  for (int x = 0; x < characterRectangleList.size(); x++) {
    characterRectangleList[x].render();
  }
}