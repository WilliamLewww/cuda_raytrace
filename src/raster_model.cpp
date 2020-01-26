#include "raster_model.h"

RasterModel::RasterModel(GLuint* shaderProgramHandle, Model* model) {
  this->model = model;

  this->shaderProgramHandle = shaderProgramHandle;

  glGenVertexArrays(1, &vao);
  glGenBuffers(2, vbo);

  glBindVertexArray(vao);

  glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
  glBufferData(GL_ARRAY_BUFFER, model->getVertexArraySize() * sizeof(Tuple), model->getVertexArray(), GL_STATIC_DRAW);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo[1]);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, model->getVertexIndexArraySize() * sizeof(int), model->getVertexIndexArray(), GL_STATIC_DRAW);

  modelMatrixLocationHandle = glGetUniformLocation(*shaderProgramHandle, "u_modelMatrix");
  viewMatrixLocationHandle = glGetUniformLocation(*shaderProgramHandle, "u_viewMatrix");
  projectionMatrixLocationHandle = glGetUniformLocation(*shaderProgramHandle, "u_projectionMatrix");
}

RasterModel::~RasterModel() {

}

Model* RasterModel::getModel() {
  return model;
}

void RasterModel::render(Camera* camera, DirectionalLight* directionalLight) {
  glEnable(GL_DEPTH_TEST);
  glUseProgram(*shaderProgramHandle);

  glBindVertexArray(vao);

  glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Tuple), nullptr);
  glEnableVertexAttribArray(0);

  glUniformMatrix4fv(modelMatrixLocationHandle, 1, GL_TRUE, model->getModelMatrix());
  glUniformMatrix4fv(viewMatrixLocationHandle, 1, GL_TRUE, camera->getViewMatrix());
  glUniformMatrix4fv(projectionMatrixLocationHandle, 1, GL_TRUE, camera->getProjectionMatrix());

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo[1]);
  glDrawElements(GL_TRIANGLES, model->getVertexIndexArraySize(), GL_UNSIGNED_INT, nullptr);

  glDisable(GL_DEPTH_TEST);
  glUseProgram(0);
}

void RasterModel::renderProvidedModelMatrix(Camera* camera, DirectionalLight* directionalLight, float* modelMatrix) {
  glEnable(GL_DEPTH_TEST);
  glUseProgram(*shaderProgramHandle);

  glBindVertexArray(vao);

  glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Tuple), nullptr);
  glEnableVertexAttribArray(0);

  glUniformMatrix4fv(modelMatrixLocationHandle, 1, GL_TRUE, modelMatrix);
  glUniformMatrix4fv(viewMatrixLocationHandle, 1, GL_TRUE, camera->getViewMatrix());
  glUniformMatrix4fv(projectionMatrixLocationHandle, 1, GL_TRUE, camera->getProjectionMatrix());

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo[1]);
  glDrawElements(GL_TRIANGLES, model->getVertexIndexArraySize(), GL_UNSIGNED_INT, nullptr);

  glDisable(GL_DEPTH_TEST);
  glUseProgram(0);
}