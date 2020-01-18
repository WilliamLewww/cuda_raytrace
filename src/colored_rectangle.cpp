#include "colored_rectangle.h"

ColoredRectangle::ColoredRectangle(GLuint* shaderProgramHandle, float positionX, float positionY, float width, float height) {
  vertices[0] =  -positionX;            vertices[1] =  -positionY;
  vertices[2] =  positionX + width;     vertices[3] =  -positionY;
  vertices[4] =  -positionX;            vertices[5] =  positionY + height;
  vertices[6] =  -positionX;            vertices[7] =  positionY + height;
  vertices[8] =  positionX + width;     vertices[9] =  -positionY;
  vertices[10] = positionX + width;     vertices[11] = positionY + height;

  this->shaderProgramHandle = shaderProgramHandle;

  glGenVertexArrays(1, &vao);
  glGenBuffers(1, vbo);

  glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
  glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(float), vertices, GL_STATIC_DRAW);

  colorHandle = glGetUniformLocation(*this->shaderProgramHandle, "u_color");
}

ColoredRectangle::~ColoredRectangle() {

}

void ColoredRectangle::render() {
  glUseProgram(*shaderProgramHandle);

  glBindVertexArray(vao);

  glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
  glEnableVertexAttribArray(0);

  glUniform1i(colorHandle, 0);

  glDrawArrays(GL_TRIANGLES, 0, 6);
}