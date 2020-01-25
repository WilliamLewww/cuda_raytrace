#include "phong_raster_model.h"

PhongRasterModel::PhongRasterModel(GLuint* shaderProgramHandle, Model* model) : RasterModel(shaderProgramHandle, model) {
  vertexListUnwrapped = model->getVertexListUnwrapped();
  normalListUnwrapped = model->getNormalListUnwrapped();

  glGenVertexArrays(1, &vao);
  glGenBuffers(2, vbo);

  glBindVertexArray(vao);

  glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
  glBufferData(GL_ARRAY_BUFFER, vertexListUnwrapped.size() * sizeof(Tuple), &vertexListUnwrapped[0], GL_STATIC_DRAW);

  glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
  glBufferData(GL_ARRAY_BUFFER, normalListUnwrapped.size() * sizeof(Tuple), &normalListUnwrapped[0], GL_STATIC_DRAW);

  modelMatrixLocationHandle = glGetUniformLocation(*shaderProgramHandle, "u_modelMatrix");
  viewMatrixLocationHandle = glGetUniformLocation(*shaderProgramHandle, "u_viewMatrix");
  projectionMatrixLocationHandle = glGetUniformLocation(*shaderProgramHandle, "u_projectionMatrix");

  lightPositionLocationHandle = glGetUniformLocation(*shaderProgramHandle, "u_lightPosition");
  lightColorLocationHandle = glGetUniformLocation(*shaderProgramHandle, "u_lightColor");
  viewPositionLocationHandle = glGetUniformLocation(*shaderProgramHandle, "u_viewPosition");
}

PhongRasterModel::~PhongRasterModel() {

}

Model* PhongRasterModel::getModel() {
  return model;
}

void PhongRasterModel::render(Camera* camera, DirectionalLight* directionalLight) {
  glEnable(GL_DEPTH_TEST);
  glUseProgram(*shaderProgramHandle);

  glBindVertexArray(vao);

  glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Tuple), nullptr);
  glEnableVertexAttribArray(0);

  glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Tuple), nullptr);
  glEnableVertexAttribArray(1);

  glUniformMatrix4fv(modelMatrixLocationHandle, 1, GL_TRUE, model->getModelMatrix());
  glUniformMatrix4fv(viewMatrixLocationHandle, 1, GL_TRUE, camera->getViewMatrix());
  glUniformMatrix4fv(projectionMatrixLocationHandle, 1, GL_TRUE, camera->getProjectionMatrix());

  glUniform3f(lightPositionLocationHandle, directionalLight->position.x, directionalLight->position.y, directionalLight->position.z);
  glUniform3f(lightColorLocationHandle, directionalLight->intensity.x, directionalLight->intensity.y, directionalLight->intensity.z);

  Tuple viewPosition = camera->getPosition();
  glUniform3f(viewPositionLocationHandle, viewPosition.x, viewPosition.y, viewPosition.z);

  glDrawArrays(GL_TRIANGLES, 0, vertexListUnwrapped.size());

  glDisable(GL_DEPTH_TEST);
  glUseProgram(0);
}