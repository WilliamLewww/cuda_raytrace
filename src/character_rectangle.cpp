#include "character_rectangle.h"

CharacterRectangle::CharacterRectangle(GLuint* shaderProgramHandle, Font* font, const char symbol, float positionX, float positionY) {
  this->font = font;
  int index = FontHandler::getIndexFromSymbol(*this->font, symbol);

  character = &this->font->characterList[index];

  float minX = (float(this->font->characterList[index].x) / this->font->width);
  float minY = (float(this->font->characterList[index].y) / this->font->height);
  float maxX = minX + (float(this->font->characterList[index].width) / this->font->width);
  float maxY = minY + (float(this->font->characterList[index].height) / this->font->height);

  vertices[0] =  positionX;                                            vertices[1] =  positionY - this->font->characterList[index].originY;
  vertices[2] =  positionX + this->font->characterList[index].width;   vertices[3] =  positionY - this->font->characterList[index].originY;
  vertices[4] =  positionX;                                            vertices[5] =  positionY - this->font->characterList[index].originY + this->font->characterList[index].height;
  vertices[6] =  positionX;                                            vertices[7] =  positionY - this->font->characterList[index].originY + this->font->characterList[index].height;
  vertices[8] =  positionX + this->font->characterList[index].width;   vertices[9] =  positionY - this->font->characterList[index].originY;
  vertices[10] = positionX + this->font->characterList[index].width;   vertices[11] = positionY - this->font->characterList[index].originY + this->font->characterList[index].height;

  textureCoordinates[0] =  minX;   textureCoordinates[1] =  minY;
  textureCoordinates[2] =  maxX;   textureCoordinates[3] =  minY;
  textureCoordinates[4] =  minX;   textureCoordinates[5] =  maxY;
  textureCoordinates[6] =  minX;   textureCoordinates[7] =  maxY;
  textureCoordinates[8] =  maxX;   textureCoordinates[9] =  minY;
  textureCoordinates[10] = maxX;   textureCoordinates[11] = maxY;

  this->shaderProgramHandle = shaderProgramHandle;

  glGenVertexArrays(1, &vao);
  glGenBuffers(2, vbo);

  glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
  glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(float), vertices, GL_STATIC_DRAW);

  glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
  glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(float), textureCoordinates, GL_STATIC_DRAW);

  textureLocationHandle = glGetUniformLocation(*this->shaderProgramHandle, "u_texture");
  resolutionLocationHandle = glGetUniformLocation(*this->shaderProgramHandle, "u_resolution");
}

CharacterRectangle::~CharacterRectangle() {

}

float CharacterRectangle::getOffsetX() {
  return float(character->width);
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

  glUniform1i(textureLocationHandle, 0);
  glUniform2f(resolutionLocationHandle, SCREEN_WIDTH, SCREEN_HEIGHT);

  glDrawArrays(GL_TRIANGLES, 0, 6);

  glBindTexture(GL_TEXTURE_2D, 0);
  glDisable(GL_BLEND);
  glUseProgram(0);
}