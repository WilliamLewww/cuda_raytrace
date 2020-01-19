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

  Tuple position;
  Tuple scale;
  float pitch;
  float yaw;

  float modelMatrix[16];
  float inverseModelMatrix[16];

  void updateModelMatrix();
  void importVertexDataFromFile(const char* filename);
public:
  Model(const char* filename, int reflective);
  Model(const Model& model);
  ~Model();

  Tuple* getVertexArray();
  int getVertexArraySize();

  int* getVertexIndexArray();
  int getVertexIndexArraySize();

  void addTransformation(float positionX, float positionY, float positionZ, float scaleX, float scaleY, float scaleZ, float pitch, float yaw);
  void updateTransformation(float positionX, float positionY, float positionZ, float scaleX, float scaleY, float scaleZ, float pitch, float yaw);
  float* getModelMatrix();

  Model* createReducedModel();

  MeshDescriptor createMeshDescriptor();
  std::vector<MeshSegment> createMeshSegmentList();
};