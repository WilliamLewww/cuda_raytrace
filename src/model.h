#pragma once
#include <vector>
#include <fstream>
#include <string>
#include <cmath>

#include "raytrace_structures.h"

class Model {
private:
  int reflective;

  std::vector<Tuple> vertexList;
  std::vector<Tuple> normalList;
  std::vector<Tuple> indexList;

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

void createReducedOBJ(const char* source, const char* target);
void initializeModelMatrix(MeshDescriptor* meshDescriptor, float* matrix);