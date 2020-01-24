#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "model.h"
#include "camera.h"

class PhongRasterModel {
private:
  Model* model;

  std::vector<Tuple> vertexListUnwrapped;
  std::vector<Tuple> normalListUnwrapped;

  GLuint* shaderProgramHandle;
  GLuint vao, vbo[2];

  GLuint modelMatrixLocationHandle;
  GLuint viewMatrixLocationHandle;
  GLuint projectionMatrixLocationHandle;

  GLuint lightPositionLocationHandle;
  GLuint lightColorLocationHandle;
  GLuint viewPositionLocationHandle;
public:
  PhongRasterModel(GLuint* shaderProgramHandle, Model* model);
  ~PhongRasterModel();

  Model* getModel();

  void render(Camera* camera, DirectionalLight* directionalLight);
};