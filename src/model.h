#pragma once
#include <vector>
#include <fstream>
#include <string>
#include <cmath>

#include "raytrace_structures.h"

class Model {
private:
  std::vector<Tuple> vertexList;
  std::vector<Tuple> normalList;
  std::vector<Tuple> indexList;

  int reflective;

  float modelMatrix[16];
  float inverseModelMatrix[16];

  void importVertexDataFromFile(const char* filename);
public:
  Model(const char* filename, int reflective);
  ~Model();

  void setModelMatrix(float* modelMatrix);

  MeshDescriptor createMeshDescriptor();
  std::vector<MeshSegment> createMeshSegmentList();
};