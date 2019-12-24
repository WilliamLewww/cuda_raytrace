#pragma once
#include <vector>
#include <fstream>
#include <string>
#include <stdio.h>

#include "structures.h"

struct Model {
  std::vector<Tuple> vertexList;
  std::vector<Tuple> normalList;
  std::vector<Tuple> indexList;

  Triangle* triangleArray;
  int triangleCount;
};

Model createModelFromOBJ(const char* filename);