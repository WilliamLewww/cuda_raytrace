#include "model_container.h"

ModelContainer::ModelContainer() {
  h_meshDescriptorCount = 0;
  h_meshSegmentCount = 0;
}

ModelContainer::~ModelContainer() {
  for (int x = 0; x < modelList.size(); x++) {
    delete modelList[x];
  }

  for (int x = 0; x < rasterModelList.size(); x++) {
    delete rasterModelList[x];
  }

  cudaFree(d_meshDescriptorBuffer);
  cudaFree(d_meshSegmentBuffer);
}

int ModelContainer::getSize() {
  return modelList.size();
}

int ModelContainer::getModelIndexFromAddress(Model* model) {
  for (int x = 0; x < modelList.size(); x++) {
    if (model == modelList[x]) {
      return x;
    }
  }

  return -1;
}

void ModelContainer::emplaceModel(GLuint* shaderProgramHandle, const char* filename, int reflective = 0) {
  modelList.push_back(ModelHandler::createModel(filename, reflective));
  rasterModelList.push_back(ModelHandler::createRasterModel(shaderProgramHandle, modelList[modelList.size() - 1]));
}

void ModelContainer::emplaceModel(GLuint* shaderProgramHandle, Model* model) {
  modelList.push_back(ModelHandler::createModel(model));
  rasterModelList.push_back(ModelHandler::createRasterModel(shaderProgramHandle, modelList[modelList.size() - 1]));
}

void ModelContainer::deleteModel(int index) {
  delete modelList[index];
  delete rasterModelList[index];

  modelList.erase(modelList.begin() + index);
  rasterModelList.erase(rasterModelList.begin() + index);
}

void ModelContainer::addModel(Model* model) {
  modelList.push_back(model);
}

void ModelContainer::removeModel(int index) {
  modelList.erase(modelList.begin() + index);
}

void ModelContainer::setModelMatrix(int index, float* modelMatrix) {
  modelList[index]->setModelMatrix(modelMatrix);
}

Model* ModelContainer::getModel(int index) {
  return modelList[index];
}

RasterModel* ModelContainer::getRasterModel(int index) {
  return rasterModelList[index];
}

void ModelContainer::addTransformation(int index, float positionX, float positionY, float positionZ, float scaleX, float scaleY, float scaleZ, float pitch, float yaw, float roll) {
  modelList[index]->addTransformation(positionX, positionY, positionZ, scaleX, scaleY, scaleZ, pitch, yaw, roll);
}

void ModelContainer::updateTransformation(int index, float positionX, float positionY, float positionZ, float scaleX, float scaleY, float scaleZ, float pitch, float yaw, float roll) {
  modelList[index]->updateTransformation(positionX, positionY, positionZ, scaleX, scaleY, scaleZ, pitch, yaw, roll);
}

float* ModelContainer::getModelMatrix(int index) {
  modelList[index]->getModelMatrix();
}

std::vector<MeshDescriptor> ModelContainer::getCollectiveMeshDescriptorList() {
  std::vector<MeshDescriptor> collectiveMeshDescriptorList;

  for (int x = 0; x < modelList.size(); x++) {
    collectiveMeshDescriptorList.push_back(modelList[x]->createMeshDescriptor());
  }

  return collectiveMeshDescriptorList;
}

std::vector<MeshSegment> ModelContainer::getCollectiveMeshSegmentList() {
  std::vector<MeshSegment> collectiveMeshSegmentList;

  for (int x = 0; x < modelList.size(); x++) {
    std::vector<MeshSegment> meshSegmentList = modelList[x]->createMeshSegmentList();
    collectiveMeshSegmentList.insert(collectiveMeshSegmentList.end(), meshSegmentList.begin(), meshSegmentList.end());
  }

  return collectiveMeshSegmentList;
}

MeshDescriptor* ModelContainer::getDeviceMeshDescriptorBuffer() {
  return d_meshDescriptorBuffer;
}

MeshSegment* ModelContainer::getDeviceMeshSegmentBuffer() {
  return d_meshSegmentBuffer;
}

int* ModelContainer::getHostMeshDescriptorCount() {
  return &h_meshDescriptorCount;
}

int* ModelContainer::getHostMeshSegmentCount() {
  return &h_meshSegmentCount;
}

void ModelContainer::updateDeviceMesh() {
  cudaFree(d_meshDescriptorBuffer);
  cudaFree(d_meshSegmentBuffer);
  
  std::vector<MeshDescriptor> h_meshDescriptorList = getCollectiveMeshDescriptorList();
  std::vector<MeshSegment> h_meshSegmentList = getCollectiveMeshSegmentList();

  h_meshDescriptorCount = h_meshDescriptorList.size();
  h_meshSegmentCount = h_meshSegmentList.size();

  cudaMalloc(&d_meshDescriptorBuffer, h_meshDescriptorCount*sizeof(MeshDescriptor));
  cudaMalloc(&d_meshSegmentBuffer, h_meshSegmentCount*sizeof(MeshSegment));

  cudaMemcpy(d_meshDescriptorBuffer, &h_meshDescriptorList[0], h_meshDescriptorCount*sizeof(MeshDescriptor), cudaMemcpyHostToDevice);
  cudaMemcpy(d_meshSegmentBuffer, &h_meshSegmentList[0], h_meshSegmentCount*sizeof(MeshSegment), cudaMemcpyHostToDevice);
}

void ModelContainer::renderRasterModels(float* viewMatrix, float* projectionMatrix) {
  for (int x = 0; x < rasterModelList.size(); x++) {
    rasterModelList[x]->render(viewMatrix, projectionMatrix);
  }
}