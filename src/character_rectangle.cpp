#include "character_rectangle.h"

float CharacterRectangle::getOffsetX() {
  return (float(character->width) / font->width);
}

void CharacterRectangle::initialize(GLuint* shaderProgramHandle, const char symbol, float positionX) {
  font = FontHolder::findFontFromName("Ubuntu");
  int index = FontHolder::findIndexFromSymbol(*font, symbol);

  character = &font->characters[index];

  float minX = (float(font->characters[index].x) / font->width);
  float minY = (float(font->characters[index].y) / font->height);
  float maxX = minX + (float(font->characters[index].width) / font->width);
  float maxY = minY + (float(font->characters[index].height) / font->height);

  float clipX = (float(font->characters[index].width) / font->width) / 2.0;
  float clipY = (float(font->characters[index].height) / font->height) / 2.0;

  float positionY = 0.0;

  vertices[0] =  -clipX + positionX;  vertices[1] =  -clipY + positionY;
  vertices[2] =  clipX + positionX;   vertices[3] =  -clipY + positionY;
  vertices[4] =  -clipX + positionX;  vertices[5] =  clipY + positionY;
  vertices[6] =  -clipX + positionX;  vertices[7] =  clipY + positionY;
  vertices[8] =  clipX + positionX;   vertices[9] =  -clipY + positionY;
  vertices[10] = clipX + positionX;   vertices[11] = clipY + positionY;

  textureCoordinates[0] =  minX;   textureCoordinates[1] =  maxY;
  textureCoordinates[2] =  maxX;   textureCoordinates[3] =  maxY;
  textureCoordinates[4] =  minX;   textureCoordinates[5] =  minY;
  textureCoordinates[6] =  minX;   textureCoordinates[7] =  minY;
  textureCoordinates[8] =  maxX;   textureCoordinates[9] =  maxY;
  textureCoordinates[10] = maxX;   textureCoordinates[11] = minY;

  this->shaderProgramHandle = shaderProgramHandle;

  glGenVertexArrays(1, &vao);
  glGenBuffers(2, vbo);

  textureHandle = glGetUniformLocation(*this->shaderProgramHandle, "u_texture");
}

void CharacterRectangle::render() {
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, FontHolder::fontUbuntuTextureResource);

  glUseProgram(*shaderProgramHandle);

  glBindVertexArray(vao);

  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
  glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(float), vertices, GL_STATIC_DRAW);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

  glEnableVertexAttribArray(1);
  glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
  glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(float), textureCoordinates, GL_STATIC_DRAW);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);

  glUniform1i(textureHandle, 0);

  glDrawArrays(GL_TRIANGLES, 0, 6);

  glBindTexture(GL_TEXTURE_2D, 0);
  glDisable(GL_BLEND);
}