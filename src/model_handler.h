#pragma once
#include <vector>

#include "model.h"

class ModelHandler {
private:
  std::vector<Model*> modelList;
public:
  ModelHandler();
  ~ModelHandler();

  void addModel(const char* filename, int reflective);
  void setModelMatrix(int index, float* modelMatrix);

  std::vector<MeshDescriptor> getCollectiveMeshDescriptorList();
  std::vector<MeshSegment> getCollectiveMeshSegmentList();

  void createReducedOBJ(const char* source, const char* target);
};