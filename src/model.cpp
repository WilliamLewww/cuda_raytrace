#include "model.h"

Model::Model(const char* filename, int reflective = 0) {
  importVertexDataFromFile(filename);
  this->reflective = reflective;

  initializeModelMatrix(modelMatrix, createIdentityMatrix());
  initializeInverseModelMatrix(inverseModelMatrix, createIdentityMatrix());
}

Model::Model(const Model& model) {
  this->vertexList = model.vertexList;
  this->normalList = model.normalList;
  this->indexList = model.indexList;

  this->reflective = model.reflective;

  for (int x = 0; x < 16; x++) {
    this->modelMatrix[x] = model.modelMatrix[x];
    this->inverseModelMatrix[x] = model.inverseModelMatrix[x];
  }
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

Model* Model::createReducedModel() {
  Model* model = new Model(*this);

  int deleteIndex = 1;
  int reducedSize = model->vertexList.size() / 2;
  for (int x = 0; x < reducedSize; x++) {
    model->vertexList.erase(model->vertexList.begin() + deleteIndex);
    deleteIndex += 1;
  }

  deleteIndex = 0;
  reducedSize = model->indexList.size() / 6;
  for (int x = 0; x < reducedSize; x++) {
    model->indexList.erase(model->indexList.begin() + deleteIndex);
    model->indexList.erase(model->indexList.begin() + deleteIndex);
    model->indexList.erase(model->indexList.begin() + deleteIndex);

    deleteIndex += 3;
  }

  for (int x = 0; x < model->indexList.size(); x++) {
    model->indexList[x].x = ceil(model->indexList[x].x / 2.0);
  }

  return model;
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