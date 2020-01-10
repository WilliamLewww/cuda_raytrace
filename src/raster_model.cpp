#include "raster_model.h"

RasterModel::RasterModel(GLuint* shaderProgramHandle, const Model& model) : Model(model) {
  this->shaderProgramHandle = shaderProgramHandle;

  glGenVertexArrays(1, &vao);
  glGenBuffers(2, vbo);

  glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
  glBufferData(GL_ARRAY_BUFFER, vertexList.size() * sizeof(float), &vertexList[0], GL_STATIC_DRAW);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo[1]);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, vertexIndexList.size() * sizeof(float), &vertexIndexList[0], GL_STATIC_DRAW);

  modelMatrixLocationHandle = glGetUniformLocation(*shaderProgramHandle, "modelMatrix");
}

RasterModel::~RasterModel() {

}

void RasterModel::render() {
  glUseProgram(*shaderProgramHandle);

  glBindVertexArray(vao);

  glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);
  glEnableVertexAttribArray(0);

  glUniformMatrix4fv(modelMatrixLocationHandle, 1, GL_FALSE, modelMatrix);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo[1]);
  glDrawElements(GL_TRIANGLES, vertexIndexList.size(), GL_UNSIGNED_INT, 0);
}