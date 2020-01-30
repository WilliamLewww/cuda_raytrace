#include "model_property_container.h"

ModelPropertyContainer::ModelPropertyContainer(Model* selectedModel, ShaderHandler* shaderHandler, FontHandler* fontHandler) {
  camera = new Camera();
  camera->setPosition(0.0, 0.0, -5.0);
  camera->setPitch(0.0);
  camera->setYaw(0.0);

  rotationX = M_PI / 8.0;
  rotationY = 0.0;

  modelMatrix = multiply(createRotationMatrixX(rotationX), createRotationMatrixY(rotationY));

  this->selectedModel = selectedModel;
  selectedRasterModel = ModelHandler::createRasterModel(RASTERMODELTYPE_RANDOM_PHONG, shaderHandler->getShaderFromName("random_colored_phong_model"), selectedModel);

  upperBackgroundRectangle = new ColoredRectangle(shaderHandler->getShaderFromName("colored_rectangle"), 0.0, 500.0, 500.0, 500.0, 0.2, 0.2, 0.2);
  lowerBackgroundRectangle = new ColoredRectangle(shaderHandler->getShaderFromName("colored_rectangle"), 0.0, 0.0, 1000.0, 500.0, 0.2, 0.2, 0.2);

  textContainer = new TextContainer(shaderHandler->getShaderFromName("textured_rectangle"), fontHandler->getFontFromName("Ubuntu"), "Model Property Editor", 0.0, 32.0);
}

ModelPropertyContainer::~ModelPropertyContainer() {
  delete upperBackgroundRectangle;
  delete lowerBackgroundRectangle;
  delete selectedRasterModel;
  delete textContainer;
  delete [] modelMatrix;
  delete camera;
}

void ModelPropertyContainer::update(float deltaTime) {
  modelMatrix = multiply(createRotationMatrixX(rotationX), createRotationMatrixY(rotationY));
  rotationY += M_PI / 12.0 * deltaTime;
}

void ModelPropertyContainer::render(DirectionalLight* directionalLight) {
  glViewport(500, 500, 500, 500);
  selectedRasterModel->renderProvidedModelMatrix(camera, directionalLight, modelMatrix);

  glViewport(0, 0, 1000, 1000);
  upperBackgroundRectangle->render();
  lowerBackgroundRectangle->render();
  textContainer->render();
}