#include "model_property_container.h"

ModelPropertyContainer::ModelPropertyContainer(Model* selectedModel, ShaderHandler* shaderHandler, FontHandler* fontHandler) {
  this->selectedModel = selectedModel;
  selectedRasterModel = ModelHandler::createRasterModel(RASTERMODELTYPE_RANDOM_PHONG, shaderHandler->getShaderFromName("random_colored_phong_model"), selectedModel);

  textContainer = new TextContainer(shaderHandler->getShaderFromName("textured_rectangle"), fontHandler->getFontFromName("Ubuntu"), "Model Property", -0.95, 0.85);
}

ModelPropertyContainer::~ModelPropertyContainer() {
  delete selectedRasterModel;
  delete textContainer;
}

void ModelPropertyContainer::update() {

}

void ModelPropertyContainer::render(Camera* camera, DirectionalLight* directionalLight) {
  selectedRasterModel->render(camera, directionalLight);

  textContainer->render();
}