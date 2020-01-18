#include "text_container.h"

TextContainer::TextContainer(GLuint* shaderProgramHandle, Font* font, std::string text, float positionX, float positionY) {
  this->shaderProgramHandle = shaderProgramHandle;
  this->font = font;

  this->positionX = positionX;
  this->positionY = positionY;

  changeText(text);
}

TextContainer::~TextContainer() {
  for (int x = 0; x < characterRectangleList.size(); x++) {
    delete characterRectangleList[x];
  }
}

void TextContainer::changeText(std::string text) {
  characterRectangleList.clear();
  
  float offsetX = 0.0;
  for (int x = 0; x < text.size(); x++) {
    characterRectangleList.push_back(new CharacterRectangle(shaderProgramHandle, font, text[x], positionX + offsetX, positionY));
    offsetX += characterRectangleList[characterRectangleList.size() - 1]->getOffsetX();
  }
}

void TextContainer::render() {
  for (int x = 0; x < characterRectangleList.size(); x++) {
    characterRectangleList[x]->render();
  }
}