#pragma once
#include <vector>
#include <fstream>
#include <string>
#include <stdio.h>

#include "raytrace_structures.h"

struct MeshDescriptor {
  int segmentCount;
  int reflective;

  float modelMatrix[16];
  float inverseModelMatrix[16];
};

struct MeshSegment {
  Tuple vertexA;
  Tuple vertexB;
  Tuple vertexC;
  Tuple normal;

  Tuple color;
};

struct Model {
  std::vector<Tuple> vertexList;
  std::vector<Tuple> normalList;
  std::vector<Tuple> indexList;

  MeshDescriptor meshDescriptor;
  MeshSegment* meshSegmentArray;
};

Model createModelFromOBJ(const char* filename, int reflective);
void initializeModelMatrix(MeshDescriptor* meshDescriptor, float* matrix);