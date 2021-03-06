#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "model.h"
#include "camera.h"

enum RasterModelType {
  RASTERMODELTYPE_RANDOM,
  RASTERMODELTYPE_RANDOM_PHONG,
};

class RasterModel {
protected:
  Model* model;

  GLuint* shaderProgramHandle;
  GLuint vao, vbo[2];

  GLuint modelMatrixLocationHandle;
  GLuint viewMatrixLocationHandle;
  GLuint projectionMatrixLocationHandle;
public:
  RasterModel(GLuint* shaderProgramHandle, Model* model);
  ~RasterModel();

  Model* getModel();

  virtual void render(Camera* camera, DirectionalLight* directionalLight);
  virtual void renderProvidedModelMatrix(Camera* camera, DirectionalLight* directionalLight, float* modelMatrix);
};