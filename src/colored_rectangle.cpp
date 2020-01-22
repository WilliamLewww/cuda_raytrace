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

  vertices[0] = ((positionX / 1000.0) * 2.0) - 1.0;             vertices[1] = ((positionY / 1000.0) * 2.0) - 1.0;
  vertices[2] = (((positionX + width) / 1000.0) * 2.0) - 1.0;   vertices[3] = ((positionY / 1000.0) * 2.0) - 1.0;
  vertices[4] = ((positionX / 1000.0) * 2.0) - 1.0;             vertices[5] = (((positionY + height) / 1000.0) * 2.0) - 1.0;
  vertices[6] = ((positionX / 1000.0) * 2.0) - 1.0;             vertices[7] = (((positionY + height) / 1000.0) * 2.0) - 1.0;
  vertices[8] = (((positionX + width) / 1000.0) * 2.0) - 1.0;   vertices[9] = ((positionY / 1000.0) * 2.0) - 1.0;
  vertices[10] = (((positionX + width) / 1000.0) * 2.0) - 1.0;  vertices[11] = (((positionY + height) / 1000.0) * 2.0) - 1.0;

  glGenVertexArrays(1, &vao);
  glGenBuffers(1, vbo);

  glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
  glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(float), vertices, GL_STATIC_DRAW);

  colorLocationHandle = glGetUniformLocation(*this->shaderProgramHandle, "u_color");
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

  glDrawArrays(GL_TRIANGLES, 0, 6);
  glUseProgram(0);
}