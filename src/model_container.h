#pragma once
#include <vector>

#include <cuda_runtime.h>

#include "model.h"
#include "raster_model.h"

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

  void addModel(Model* model);
  void removeModel(int index);
  Model* getModel(int index);

  void updateTransformation(int index, float positionX, float positionY, float positionZ, float scaleX, float scaleY, float scaleZ, float pitch, float yaw, float roll);
  float* getModelMatrix(int index);

  MeshDescriptor* getDeviceMeshDescriptorBuffer();
  MeshSegment* getDeviceMeshSegmentBuffer();

  int* getHostMeshDescriptorCount();
  int* getHostMeshSegmentCount();

  void updateDeviceMesh();
};