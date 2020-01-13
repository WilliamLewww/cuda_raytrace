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
  this->vertexIndexList = model.vertexIndexList;
  this->textureIndexList = model.textureIndexList;
  this->normalIndexList = model.normalIndexList;

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

      vertexIndexList.push_back(a.x - 1);
      textureIndexList.push_back(a.y - 1);
      normalIndexList.push_back(a.z - 1);

      vertexIndexList.push_back(b.x - 1);
      textureIndexList.push_back(b.y - 1);
      normalIndexList.push_back(b.z - 1);

      vertexIndexList.push_back(c.x - 1);
      textureIndexList.push_back(c.y - 1);
      normalIndexList.push_back(c.z - 1);
    }
  }

  file.close();
}

void Model::setModelMatrix(float* modelMatrix) {
  initializeModelMatrix(this->modelMatrix, modelMatrix);
  initializeInverseModelMatrix(inverseModelMatrix, modelMatrix);
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
  reducedSize = model->vertexIndexList.size() / 6;
  for (int x = 0; x < reducedSize; x++) {
    for (int y = 0; y < 3; y++) {
      model->vertexIndexList.erase(model->vertexIndexList.begin() + deleteIndex);
      model->textureIndexList.erase(model->textureIndexList.begin() + deleteIndex);
      model->normalIndexList.erase(model->normalIndexList.begin() + deleteIndex);
    }

    deleteIndex += 3;
  }

  for (int x = 0; x < model->vertexIndexList.size(); x++) {
    model->vertexIndexList[x] = ceil(model->vertexIndexList[x] / 2.0);
  }

  return model;
}

MeshDescriptor Model::createMeshDescriptor() {
  MeshDescriptor meshDescriptor;
  meshDescriptor.segmentCount = vertexIndexList.size() / 3;
  meshDescriptor.reflective = reflective;
  initializeModelMatrix(&meshDescriptor, modelMatrix);

  return meshDescriptor;
}

std::vector<MeshSegment> Model::createMeshSegmentList() {
  std::vector<MeshSegment> meshSegmentList;

  for (int x = 0; x < vertexIndexList.size() / 3; x++) {
    MeshSegment segment;
    segment.vertexA = vertexList[vertexIndexList[(3 * x)]];
    segment.vertexB = vertexList[vertexIndexList[(3 * x) + 1]];
    segment.vertexC = vertexList[vertexIndexList[(3 * x) + 2]];

    segment.normal = normalList[normalIndexList[(3 * x)]];

    segment.color = {float(int(45.0 * x + 87) % 255), float(int(77.0 * x + 102) % 255), float(int(123.0 * x + 153) % 255), 1.0};
  
    meshSegmentList.push_back(segment);
  }

  return meshSegmentList;
}