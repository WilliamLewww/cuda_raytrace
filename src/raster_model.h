#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "model.h"

class RasterModel : public Model {
private:
  GLuint* shaderProgramHandle;
  GLuint vao, vbo[2];

  GLuint modelMatrixLocationHandle;
  GLuint viewMatrixLocationHandle;
  GLuint projectionMatrixLocationHandle;
public:
  RasterModel(GLuint* shaderProgramHandle, const Model& model);
  ~RasterModel();

  void render(float* viewMatrix, float* projectionMatrix);
};