#pragma once
#include <vector>

#include "model.h"
#include "raster_model.h"

class ModelHandler {
private:
  std::vector<Model*> modelList;
public:
  ModelHandler();
  ~ModelHandler();

  int getModelListSize();

  void addModel(const char* filename, int reflective);
  void addModel(Model* model);
  void removeModel(int index);

  void setModelMatrix(int index, float* modelMatrix);

  std::vector<MeshDescriptor> getCollectiveMeshDescriptorList();
  std::vector<MeshSegment> getCollectiveMeshSegmentList();

  RasterModel* createRasterModel(GLuint* shaderProgramHandle, int index);
  Model* createReducedModel(int index);
  void createReducedOBJ(const char* source, const char* target);
};