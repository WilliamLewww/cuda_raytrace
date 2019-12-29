#include "character_rectangle.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

int CharacterRectangle::findIndexFromSymbol(char symbol) {
  for (int x = 0; x < fontUbuntu.characterCount; x++) {
    if (fontUbuntu.characters[x].symbol == symbol) {
      return x;
    }
  }

  return -1;
}

void CharacterRectangle::initialize(GLuint* shaderProgramHandle, char symbol) {
  vertices[0] = -1.0;   vertices[1] = -1.0;
  vertices[2] =  1.0;   vertices[3] = -1.0;
  vertices[4] = -1.0;   vertices[5] =  1.0;
  vertices[6] = -1.0;   vertices[7] =  1.0;
  vertices[8] =  1.0;   vertices[9] = -1.0;
  vertices[10] = 1.0;   vertices[11] = 1.0;

  int index = findIndexFromSymbol(symbol);

  float minX = (float(fontUbuntu.characters[index].x) / fontUbuntu.width);
  float minY = (float(fontUbuntu.characters[index].y) / fontUbuntu.height);
  float maxX = minX + (float(fontUbuntu.characters[index].width) / fontUbuntu.width);
  float maxY = minY + (float(fontUbuntu.characters[index].height) / fontUbuntu.height);

  textureCoordinates[0] =  minX;   textureCoordinates[1] =  maxY;
  textureCoordinates[2] =  maxX;   textureCoordinates[3] =  maxY;
  textureCoordinates[4] =  minX;   textureCoordinates[5] =  minY;
  textureCoordinates[6] =  minX;   textureCoordinates[7] =  minY;
  textureCoordinates[8] =  maxX;   textureCoordinates[9] =  maxY;
  textureCoordinates[10] = maxX;   textureCoordinates[11] = minY;

  int w, h, comp;
  unsigned char* image = stbi_load("res/font_ubuntu.png", &w, &h, &comp, STBI_rgb_alpha);

  glGenTextures(1, &textureResource);
  glBindTexture(GL_TEXTURE_2D, textureResource);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, image);
  glBindTexture(GL_TEXTURE_2D, 0);

  stbi_image_free(image);

  this->shaderProgramHandle = shaderProgramHandle;

  glGenVertexArrays(1, &vao);
  glGenBuffers(2, vbo);

  textureHandle = glGetUniformLocation(*this->shaderProgramHandle, "u_texture");
}

void CharacterRectangle::render() {
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, textureResource);

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
}