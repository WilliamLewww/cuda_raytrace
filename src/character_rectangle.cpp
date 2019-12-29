#include "character_rectangle.h"

void CharacterRectangle::initialize(GLuint* shaderProgramHandle, char symbol) {
  vertices[0] =  0.0;   vertices[1] =  0.0;
  vertices[2] =  1.0;   vertices[3] =  0.0;
  vertices[4] =  0.0;   vertices[5] =  1.0;
  vertices[6] =  0.0;   vertices[7] =  1.0;
  vertices[8] =  1.0;   vertices[9] =  0.0;
  vertices[10] = 1.0;   vertices[11] = 1.0;

  Font* font = FontHolder::findFontFromName("Ubuntu");
  int index = FontHolder::findIndexFromSymbol(*font, symbol);

  float minX = (float(font->characters[index].x) / font->width);
  float minY = (float(font->characters[index].y) / font->height);
  float maxX = minX + (float(font->characters[index].width) / font->width);
  float maxY = minY + (float(font->characters[index].height) / font->height);

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