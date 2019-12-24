#pragma once
#include <vector>

#include "structures.h"

struct Model {
  std::vector<Tuple> vertexList;
  std::vector<Tuple> normalList;
  std::vector<Tuple> indexList;

  Triangle* triangleArray;
  int triangleCount;
};

Model createModelFromOBJ(const char* filename);