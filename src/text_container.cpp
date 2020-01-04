#include "text_container.h"

void TextContainer::initialize(GLuint* shaderProgramHandle, Font* font, std::string text, float positionX, float positionY) {
  this->shaderProgramHandle = shaderProgramHandle;
  this->font = font;

  this->positionX = positionX;
  this->positionY = positionY;

  changeText(text);
}

void TextContainer::changeText(std::string text) {
  characterRectangleList.clear();
  
  float offsetX = 0.0;
  for (int x = 0; x < text.size(); x++) {
    characterRectangleList.push_back(CharacterRectangle());
    characterRectangleList[characterRectangleList.size() - 1].initialize(shaderProgramHandle, font, text[x], positionX + offsetX, positionY);
    offsetX += characterRectangleList[characterRectangleList.size() - 1].getOffsetX();
  }

  squiggleAnimationText.initialize(&characterRectangleList);
}

void TextContainer::update() {
  squiggleAnimationText.animate();
}

void TextContainer::render() {
  for (int x = 0; x < characterRectangleList.size(); x++) {
    characterRectangleList[x].render();
  }
}