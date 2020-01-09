#include "model.h"

Model::Model(const char* filename, int reflective = 0) {
  importVertexDataFromFile(filename);
  this->reflective = reflective;

  initializeModelMatrix(modelMatrix, createIdentityMatrix());
  initializeInverseModelMatrix(inverseModelMatrix, createIdentityMatrix());
}

Model::~Model() {

}

void Model::importVertexDataFromFile(const char* filename) {
  std::ifstream file(filename);
  std::string line;

  while (std::getline(file, line)) {
    if (line.substr(0, line.find_first_of(' ')) == "v") {
      std::string temp = line.substr(line.find_first_of(' ') + 1);
      float x, y, z;

      if (temp.at(0) == '-') { x = -std::stof(temp.substr(1, temp.find_first_of(' '))); }
      else { x = std::stof(temp.substr(0, temp.find_first_of(' '))); }
      temp = temp.substr(temp.find_first_of(' ') + 1);

      if (temp.at(0) == '-') { y = -std::stof(temp.substr(1, temp.find_first_of(' '))); }
      else { y = std::stof(temp.substr(0, temp.find_first_of(' '))); }
      temp = temp.substr(temp.find_first_of(' ') + 1);

      if (temp.at(0) == '-') { z = -std::stof(temp.substr(1, temp.find_first_of(' '))); }
      else { z = std::stof(temp.substr(0, temp.find_first_of(' '))); }

      vertexList.push_back({x, y, z, 1.0});
    }

    if (line.substr(0, line.find_first_of(' ')) == "vn") {
      std::string temp = line.substr(line.find_first_of(' ') + 1);
      float x, y, z;

      if (temp.at(0) == '-') { x = -std::stof(temp.substr(1, temp.find_first_of(' '))); }
      else { x = std::stof(temp.substr(0, temp.find_first_of(' '))); }
      temp = temp.substr(temp.find_first_of(' ') + 1);

      if (temp.at(0) == '-') { y = -std::stof(temp.substr(1, temp.find_first_of(' '))); }
      else { y = std::stof(temp.substr(0, temp.find_first_of(' '))); }
      temp = temp.substr(temp.find_first_of(' ') + 1);

      if (temp.at(0) == '-') { z = -std::stof(temp.substr(1, temp.find_first_of(' '))); }
      else { z = std::stof(temp.substr(0, temp.find_first_of(' '))); }

      normalList.push_back({x, y, z, 0.0});
    }

    if (line.substr(0, line.find_first_of(' ')) == "f") {
      std::string temp = line.substr(line.find_first_of(' ') + 1);
      Tuple a, b, c;

      a.x = std::stof(temp.substr(0, temp.find_first_of('/')));
      temp = temp.substr(temp.find_first_of('/') + 1);
      // printf("%d\n", std::stof(temp.substr(0, temp.find_first_of('/'))));
      temp = temp.substr(temp.find_first_of('/') + 1);
      a.z = std::stof(temp.substr(0, temp.find_first_of(' ')));

      temp = temp.substr(temp.find_first_of(' ') + 1);

      b.x = std::stof(temp.substr(0, temp.find_first_of('/')));
      temp = temp.substr(temp.find_first_of('/') + 1);
      // printf("%d\n", std::stof(temp.substr(0, temp.find_first_of('/'))));
      temp = temp.substr(temp.find_first_of('/') + 1);
      b.z = std::stof(temp.substr(0, temp.find_first_of(' ')));

      temp = temp.substr(temp.find_first_of(' ') + 1);

      c.x = std::stof(temp.substr(0, temp.find_first_of('/')));
      temp = temp.substr(temp.find_first_of('/') + 1);
      // printf("%d\n", std::stof(temp.substr(0, temp.find_first_of('/'))));
      temp = temp.substr(temp.find_first_of('/') + 1);
      c.z = std::stof(temp.substr(0, temp.find_first_of(' ')));

      indexList.push_back(a);
      indexList.push_back(b);
      indexList.push_back(c);
    }
  }

  file.close();
}

void Model::setModelMatrix(float* modelMatrix) {
  initializeModelMatrix(this->modelMatrix, modelMatrix);
  initializeInverseModelMatrix(inverseModelMatrix, modelMatrix);
}

MeshDescriptor Model::createMeshDescriptor() {
  MeshDescriptor meshDescriptor;
  meshDescriptor.segmentCount = indexList.size() / 3;
  meshDescriptor.reflective = reflective;
  initializeModelMatrix(&meshDescriptor, modelMatrix);

  return meshDescriptor;
}

std::vector<MeshSegment> Model::createMeshSegmentList() {
  std::vector<MeshSegment> meshSegmentList;

  for (int x = 0; x < indexList.size() / 3; x++) {
    MeshSegment segment;
    segment.vertexA = vertexList[indexList[(3 * x)].x - 1];
    segment.vertexB = vertexList[indexList[(3 * x) + 1].x - 1];
    segment.vertexC = vertexList[indexList[(3 * x) + 2].x - 1];

    segment.normal = normalList[indexList[(3 * x)].z - 1];

    segment.color = {float(int(45.0 * x + 87) % 255), float(int(77.0 * x + 102) % 255), float(int(123.0 * x + 153) % 255), 1.0};
  
    meshSegmentList.push_back(segment);
  }

  return meshSegmentList;
}

void createReducedOBJ(const char* source, const char* target) {
  std::ifstream sourceFile(source);
  std::ofstream targetFile(target);
  std::string line;

  bool skipVertex = false;
  bool skipFace = true;
  while (std::getline(sourceFile, line)) {
    int lineType = -1;

    if (line.substr(0, line.find_first_of(' ')) == "v") { lineType = 1; }
    if (line.substr(0, line.find_first_of(' ')) == "f") { lineType = 2; }

    if (lineType == -1) {
      targetFile << line << "\n";
      skipVertex = false;
      skipFace = true;
    }

    if (lineType == 1) {
      if (!skipVertex) {
        targetFile << line << "\n";
      }

      skipVertex = !skipVertex;
    }

    if (lineType == 2) {
      if (!skipFace) {
        std::string tempLine = line;

        std::string reconstructedLine = "f ";

        reconstructedLine += std::to_string((int)ceil(std::stoi(tempLine.substr(2, tempLine.find_first_of("//") - 2)) / 2.0));
        tempLine = tempLine.substr(tempLine.find_first_of("//"));
        reconstructedLine += tempLine.substr(0, tempLine.find_first_of(" "));
        reconstructedLine += " ";

        tempLine = tempLine.substr(tempLine.find_first_of(" ") + 1);
        reconstructedLine += std::to_string((int)ceil(std::stoi(tempLine.substr(0, tempLine.find_first_of("//"))) / 2.0));
        tempLine = tempLine.substr(tempLine.find_first_of("//"));
        reconstructedLine += tempLine.substr(0, tempLine.find_first_of(" "));
        reconstructedLine += " ";

        tempLine = tempLine.substr(tempLine.find_first_of(" ") + 1);
        reconstructedLine += std::to_string((int)ceil(std::stoi(tempLine.substr(0, tempLine.find_first_of("//"))) / 2.0));
        tempLine = tempLine.substr(tempLine.find_first_of("//"));
        reconstructedLine += tempLine;

        targetFile << reconstructedLine << "\n";
      }

      skipFace = !skipFace;
    }
  }

  sourceFile.close();
  targetFile.close();
}