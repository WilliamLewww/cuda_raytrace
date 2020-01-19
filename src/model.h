#pragma once
#include <vector>
#include <fstream>
#include <string>
#include <cmath>

#include "raytrace_structures.h"

class Model {
protected:
  std::vector<Tuple> vertexList;
  std::vector<Tuple> normalList;
  std::vector<int> vertexIndexList;
  std::vector<int> textureIndexList;
  std::vector<int> normalIndexList;

  int reflective;

  float modelMatrix[16];
  float inverseModelMatrix[16];

  void importVertexDataFromFile(const char* filename);
public:
  Model(const char* filename, int reflective);
  Model(const Model& model);
  ~Model();

  Tuple* getVertexArray();
  int getVertexArraySize();

  int* getVertexIndexArray();
  int getVertexIndexArraySize();

  float* getModelMatrix();
  void setModelMatrix(float* modelMatrix);

  Model* createReducedModel();

  MeshDescriptor createMeshDescriptor();
  std::vector<MeshSegment> createMeshSegmentList();
};