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
  viewMatrixLocationHandle = glGetUniformLocation(*shaderProgramHandle, "viewMatrix");
  projectionMatrixLocationHandle = glGetUniformLocation(*shaderProgramHandle, "projectionMatrix");
}

RasterModel::~RasterModel() {

}

void RasterModel::render(float* viewMatrix, float* projectionMatrix) {
  glEnable(GL_DEPTH_TEST);
  glUseProgram(*shaderProgramHandle);

  glBindVertexArray(vao);

  glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);
  glEnableVertexAttribArray(0);

  glUniformMatrix4fv(modelMatrixLocationHandle, 1, GL_FALSE, modelMatrix);
  glUniformMatrix4fv(viewMatrixLocationHandle, 1, GL_FALSE, viewMatrix);
  glUniformMatrix4fv(projectionMatrixLocationHandle, 1, GL_FALSE, projectionMatrix);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo[1]);
  glDrawElements(GL_TRIANGLES, vertexIndexList.size(), GL_UNSIGNED_INT, 0);
  
  glDisable(GL_DEPTH_TEST);
}