#include "model.h"

Model::Model(const char* filename, int reflective = 0) {
  importVertexDataFromFile(filename);
  this->reflective = reflective;

  position = {0.0, 0.0, 0.0, 1.0};
  scale = {1.0, 1.0, 1.0, 0.0};
  pitch = 0.0;
  yaw = 0.0;
  roll = 0.0;

  updateModelMatrix();
}

Model::Model(const Model& model) {
  this->vertexList = model.vertexList;
  this->normalList = model.normalList;
  this->vertexIndexList = model.vertexIndexList;
  this->textureIndexList = model.textureIndexList;
  this->normalIndexList = model.normalIndexList;

  this->reflective = model.reflective;

  this->position = {model.position.x, model.position.y, model.position.z, model.position.w};
  this->scale = {model.scale.x, model.scale.y, model.scale.z, model.scale.w};
  this->pitch = model.pitch;
  this->yaw = model.yaw;
  this->roll = model.roll;

  for (int x = 0; x < 16; x++) {
    this->modelMatrix[x] = model.modelMatrix[x];
    this->inverseModelMatrix[x] = model.inverseModelMatrix[x];
  }
}

Model::~Model() {

}

Tuple* Model::getVertexArray() {
  return &vertexList[0];
}

int Model::getVertexArraySize() {
  return vertexList.size();
}

Tuple* Model::getNormalArray() {
  return &normalList[0];
}

int Model::getNormalArraySize() {
  return normalList.size();
}

std::vector<Tuple> Model::getVertexListUnwrapped() {
  std::vector<Tuple> vertexListUnwrapped;

  for (int x = 0; x < vertexIndexList.size(); x++) {
    vertexListUnwrapped.push_back(vertexList[vertexIndexList[x]]);
  }

  return vertexListUnwrapped;
}

std::vector<Tuple> Model::getNormalListUnwrapped() {
  std::vector<Tuple> normalListUnwrapped;

  for (int x = 0; x < normalIndexList.size(); x++) {
    normalListUnwrapped.push_back(normalList[normalIndexList[x]]);
  }

  return normalListUnwrapped;
}

int* Model::getVertexIndexArray() {
  return &vertexIndexList[0];
}

int Model::getVertexIndexArraySize() {
  return vertexIndexList.size();
}

void Model::addTransformation(float positionX, float positionY, float positionZ, float scaleX, float scaleY, float scaleZ, float pitch, float yaw, float roll) {
  position.x += positionX;
  position.y += positionY;
  position.z += positionZ;
  scale.x += scaleX;
  scale.y += scaleY;
  scale.z += scaleZ;

  this->pitch += pitch;
  this->yaw += yaw;
  this->roll += roll;

  updateModelMatrix();
}

void Model::updateTransformation(float positionX, float positionY, float positionZ, float scaleX, float scaleY, float scaleZ, float pitch, float yaw, float roll) {
  position.x = positionX;
  position.y = positionY;
  position.z = positionZ;
  scale.x = scaleX;
  scale.y = scaleY;
  scale.z = scaleZ;

  this->pitch = pitch;
  this->yaw = yaw;
  this->roll = roll;

  updateModelMatrix();
}

void Model::updateModelMatrix() {
  float* transformMatrix = multiply(multiply(multiply(multiply(createTranslateMatrix(position.x, position.y, position.z), createScaleMatrix(scale.x, scale.y, scale.z)), createRotationMatrixX(pitch)), createRotationMatrixY(yaw)), createRotationMatrixZ(roll));

  initializeModelMatrix(modelMatrix, transformMatrix);
  initializeInverseModelMatrix(inverseModelMatrix, transformMatrix);
}

void Model::setModelMatrix(float* modelMatrix) {
  initializeModelMatrix(this->modelMatrix, modelMatrix);
}

float* Model::getModelMatrix() {
  return modelMatrix;
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

Model* Model::createReducedModel() {
  Model* model = new Model(*this);

  int reducedSize = model->vertexList.size() / 2;
  for (int x = 0; x < reducedSize; x++) {
    model->vertexList.erase(model->vertexList.begin() + x + 1);
  }

  reducedSize = model->vertexIndexList.size() / 6;
  for (int x = 0; x < reducedSize; x++) {
    for (int y = 0; y < 3; y++) {
      model->vertexIndexList.erase(model->vertexIndexList.begin() + (x * 3));
      model->textureIndexList.erase(model->textureIndexList.begin() + (x * 3));
      model->normalIndexList.erase(model->normalIndexList.begin() + (x * 3));
    }
  }

  for (int x = 0; x < model->vertexIndexList.size(); x++) {
    model->vertexIndexList[x] = floor(model->vertexIndexList[x] / 2.0);
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

    segment.color = {float(int(45.0 * x + 87.0) % 255), float(int(77.0 * x + 102.0) % 255), float(int(123.0 * x + 153.0) % 255), 1.0};
  
    meshSegmentList.push_back(segment);
  }

  return meshSegmentList;
}