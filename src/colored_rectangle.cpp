#include "colored_rectangle.h"

ColoredRectangle::ColoredRectangle(GLuint* shaderProgramHandle, float positionX, float positionY, float width, float height, float red, float green, float blue) {
  this->shaderProgramHandle = shaderProgramHandle;

  this->positionX = positionX;
  this->positionY = positionY;

  this->width = width;
  this->height = height;

  this->red = red;
  this->green = green;
  this->blue = blue;

  vertices[0] = positionX;           vertices[1] = positionY;
  vertices[2] = positionX + width;   vertices[3] = positionY;
  vertices[4] = positionX;           vertices[5] = positionY + height;
  vertices[6] = positionX;           vertices[7] = positionY + height;
  vertices[8] = positionX + width;   vertices[9] = positionY;
  vertices[10] = positionX + width;  vertices[11] = positionY + height;

  glGenVertexArrays(1, &vao);
  glGenBuffers(1, vbo);

  glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
  glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(float), vertices, GL_STATIC_DRAW);

  colorLocationHandle = glGetUniformLocation(*this->shaderProgramHandle, "u_color");
  resolutionLocationHandle = glGetUniformLocation(*this->shaderProgramHandle, "u_resolution");
}

ColoredRectangle::~ColoredRectangle() {

}

void ColoredRectangle::render() {
  glUseProgram(*shaderProgramHandle);

  glBindVertexArray(vao);

  glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
  glEnableVertexAttribArray(0);

  glUniform4f(colorLocationHandle, red, green, blue, 1.0);
  glUniform2f(resolutionLocationHandle, SCREEN_WIDTH, SCREEN_HEIGHT);

  glDrawArrays(GL_TRIANGLES, 0, 6);
  glUseProgram(0);
}