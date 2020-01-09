#include "model_handler.h"

ModelHandler::ModelHandler() {

}

ModelHandler::~ModelHandler() {
  for (int x = 0; x < modelList.size(); x++) {
    delete modelList[x];
  }

  modelList.clear();
}

void ModelHandler::addModel(const char* filename, int reflective = 0) {
  modelList.push_back(new Model(filename, reflective));
}

void ModelHandler::setModelMatrix(int index, float* modelMatrix) {
  modelList[index]->setModelMatrix(modelMatrix);
}

std::vector<MeshDescriptor> ModelHandler::getCollectiveMeshDescriptorList() {
  std::vector<MeshDescriptor> collectiveMeshDescriptorList;

  for (int x = 0; x < modelList.size(); x++) {
    collectiveMeshDescriptorList.push_back(modelList[x]->createMeshDescriptor());
  }

  return collectiveMeshDescriptorList;
}

std::vector<MeshSegment> ModelHandler::getCollectiveMeshSegmentList() {
  std::vector<MeshSegment> collectiveMeshSegmentList;

  for (int x = 0; x < modelList.size(); x++) {
    std::vector<MeshSegment> meshSegmentList = modelList[x]->createMeshSegmentList();
    collectiveMeshSegmentList.insert(collectiveMeshSegmentList.end(), meshSegmentList.begin(), meshSegmentList.end());
  }

  return collectiveMeshSegmentList;
}