#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "raster_model.h"
#include "camera.h"

class PhongRasterModel : public RasterModel {
private:
  std::vector<Tuple> vertexListUnwrapped;
  std::vector<Tuple> normalListUnwrapped;

  GLuint lightPositionLocationHandle;
  GLuint lightColorLocationHandle;
  GLuint viewPositionLocationHandle;
public:
  PhongRasterModel(GLuint* shaderProgramHandle, Model* model);
  ~PhongRasterModel();

  Model* getModel();

  void render(Camera* camera, DirectionalLight* directionalLight) override;
  void renderProvidedModelMatrix(Camera* camera, DirectionalLight* directionalLight, float* modelMatrix) override;
};