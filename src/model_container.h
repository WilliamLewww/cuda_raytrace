#pragma once
#include <vector>

#include <cuda_runtime.h>

#include "model.h"
#include "raster_model.h"
#include "model_handler.h"

class ModelContainer {
private:
  std::vector<Model*> modelList;
  std::vector<RasterModel*> rasterModelList;

  MeshDescriptor* d_meshDescriptorBuffer;
  MeshSegment* d_meshSegmentBuffer;

  int h_meshDescriptorCount;
  int h_meshSegmentCount;

  std::vector<MeshDescriptor> getCollectiveMeshDescriptorList();
  std::vector<MeshSegment> getCollectiveMeshSegmentList();
public:
  ModelContainer();
  ~ModelContainer();

  int getSize();
  int getModelIndexFromAddress(Model* model);

  void emplaceModel(GLuint* shaderProgramHandle, const char* filename, int reflective);
  void emplaceModel(GLuint* shaderProgramHandle, Model* model);
  void deleteModel(int index);

  void addModel(Model* model);
  void removeModel(int index);
  
  Model* getModel(int index);
  RasterModel* getRasterModel(int index);

  void updateTransformation(int index, float positionX, float positionY, float positionZ, float scaleX, float scaleY, float scaleZ, float pitch, float yaw, float roll);
  
  void setModelMatrix(int index, float* modelMatrix);
  float* getModelMatrix(int index);

  MeshDescriptor* getDeviceMeshDescriptorBuffer();
  MeshSegment* getDeviceMeshSegmentBuffer();

  int* getHostMeshDescriptorCount();
  int* getHostMeshSegmentCount();

  void updateDeviceMesh();

  void renderRasterModels(float* viewMatrix, float* projectionMatrix);
};