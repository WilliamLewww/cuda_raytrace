#include "character_rectangle.h"

CharacterRectangle::CharacterRectangle(GLuint* shaderProgramHandle, Font* font, const char symbol, float positionX, float positionY) {
  this->font = font;
  int index = FontHandler::getIndexFromSymbol(*this->font, symbol);

  character = &this->font->characterList[index];

  float minX = (float(this->font->characterList[index].x) / this->font->width);
  float minY = (float(this->font->characterList[index].y) / this->font->height);
  float maxX = minX + (float(this->font->characterList[index].width) / this->font->width);
  float maxY = minY + (float(this->font->characterList[index].height) / this->font->height);

  float clipX = (float(this->font->characterList[index].width) / this->font->width) / 4.0;
  float clipY = (float(this->font->characterList[index].height) / this->font->height) / 4.0;

  float offsetX = positionX;
  float offsetY = positionY + (float(this->font->characterList[index].originY) / this->font->height) / 4.0;

  vertices[0] =  -clipX + offsetX;  vertices[1] =  -clipY + offsetY;
  vertices[2] =  clipX + offsetX;   vertices[3] =  -clipY + offsetY;
  vertices[4] =  -clipX + offsetX;  vertices[5] =  clipY + offsetY;
  vertices[6] =  -clipX + offsetX;  vertices[7] =  clipY + offsetY;
  vertices[8] =  clipX + offsetX;   vertices[9] =  -clipY + offsetY;
  vertices[10] = clipX + offsetX;   vertices[11] = clipY + offsetY;

  textureCoordinates[0] =  minX;   textureCoordinates[1] =  maxY;
  textureCoordinates[2] =  maxX;   textureCoordinates[3] =  maxY;
  textureCoordinates[4] =  minX;   textureCoordinates[5] =  minY;
  textureCoordinates[6] =  minX;   textureCoordinates[7] =  minY;
  textureCoordinates[8] =  maxX;   textureCoordinates[9] =  maxY;
  textureCoordinates[10] = maxX;   textureCoordinates[11] = minY;

  this->shaderProgramHandle = shaderProgramHandle;

  glGenVertexArrays(1, &vao);
  glGenBuffers(2, vbo);

  glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
  glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(float), vertices, GL_STATIC_DRAW);

  glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
  glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(float), textureCoordinates, GL_STATIC_DRAW);

  textureHandle = glGetUniformLocation(*this->shaderProgramHandle, "u_texture");
}

CharacterRectangle::~CharacterRectangle() {

}

float CharacterRectangle::getOffsetX() {
  return (float(character->width) / font->width);
}

void CharacterRectangle::addPosition(float positionX, float positionY) {
  vertices[0] +=  positionX;   vertices[1] +=  positionY;
  vertices[2] +=  positionX;   vertices[3] +=  positionY;
  vertices[4] +=  positionX;   vertices[5] +=  positionY;
  vertices[6] +=  positionX;   vertices[7] +=  positionY;
  vertices[8] +=  positionX;   vertices[9] +=  positionY;
  vertices[10] += positionX;   vertices[11] += positionY;
}

void CharacterRectangle::render() {
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, font->textureResource);

  glUseProgram(*shaderProgramHandle);

  glBindVertexArray(vao);

  glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
  glEnableVertexAttribArray(0);

  glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);
  glEnableVertexAttribArray(1);

  glUniform1i(textureHandle, 0);

  glDrawArrays(GL_TRIANGLES, 0, 6);

  glBindTexture(GL_TEXTURE_2D, 0);
  glDisable(GL_BLEND);
  glUseProgram(0);
}