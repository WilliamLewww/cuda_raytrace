#pragma once
#include <vector>

#include <cuda_runtime.h>

#include "model.h"
#include "raster_model.h"

class ModelHandler {
private:
  std::vector<Model*> modelList;

  MeshDescriptor* d_meshDescriptorBuffer;
  MeshSegment* d_meshSegmentBuffer;

  int h_meshDescriptorCount;
  int h_meshSegmentCount;

  std::vector<MeshDescriptor> getCollectiveMeshDescriptorList();
  std::vector<MeshSegment> getCollectiveMeshSegmentList();
public:
  ModelHandler();
  ~ModelHandler();

  int getModelListSize();

  void addModel(const char* filename, int reflective);
  void addModel(Model* model);
  void removeModel(int index);

  Model* getModel(int index);

  void setModelMatrix(int index, float* modelMatrix);
  float* getModelMatrix(int index);

  MeshDescriptor* getDeviceMeshDescriptorBuffer();
  MeshSegment* getDeviceMeshSegmentBuffer();

  int* getHostMeshDescriptorCount();
  int* getHostMeshSegmentCount();

  void updateDeviceMesh();

  RasterModel* createRasterModel(GLuint* shaderProgramHandle, int index);
  Model* createReducedModel(int index);
  void createReducedOBJ(const char* source, const char* target);
};