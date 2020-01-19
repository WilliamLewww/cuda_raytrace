#include "model_handler.h"

ModelHandler::ModelHandler() {
  d_meshDescriptorBuffer = nullptr;
  d_meshSegmentBuffer = nullptr;

  h_meshDescriptorCount = 0;
  h_meshSegmentCount = 0;
}

ModelHandler::~ModelHandler() {
  for (int x = 0; x < modelList.size(); x++) {
    delete modelList[x];
  }

  cudaFree(d_meshDescriptorBuffer);
  cudaFree(d_meshSegmentBuffer);
}

int ModelHandler::getModelListSize() {
  return modelList.size();
}

void ModelHandler::addModel(const char* filename, int reflective = 0) {
  modelList.push_back(new Model(filename, reflective));
}

void ModelHandler::addModel(Model* model) {
  modelList.push_back(model);
}

void ModelHandler::removeModel(int index) {
  delete modelList[index];
  modelList.erase(modelList.begin() + index);
}

Model* ModelHandler::getModel(int index) {
  return modelList[index];
}

void ModelHandler::setModelMatrix(int index, float* modelMatrix) {
  modelList[index]->setModelMatrix(modelMatrix);
}

float* ModelHandler::getModelMatrix(int index) {
  modelList[index]->getModelMatrix();
}

std::vector<MeshDescriptor> ModelHandler::getCollectiveMeshDescriptorList() {
  std::vector<MeshDescriptor> collectiveMeshDescriptorList;

  for (int x = 0; x < modelList.size(); x++) {
    collectiveMeshDescriptorList.push_back(modelList[x]->createMeshDescriptor());
  }

  return collectiveMeshDescriptorList;
}

std::vector<MeshSegment> ModelHandler::getCollectiveMeshSegmentList() {
  std::vector<MeshSegment> collectiveMeshSegmentList;

  for (int x = 0; x < modelList.size(); x++) {
    std::vector<MeshSegment> meshSegmentList = modelList[x]->createMeshSegmentList();
    collectiveMeshSegmentList.insert(collectiveMeshSegmentList.end(), meshSegmentList.begin(), meshSegmentList.end());
  }

  return collectiveMeshSegmentList;
}

MeshDescriptor* ModelHandler::getDeviceMeshDescriptorBuffer() {
  return d_meshDescriptorBuffer;
}

MeshSegment* ModelHandler::getDeviceMeshSegmentBuffer() {
  return d_meshSegmentBuffer;
}

int* ModelHandler::getHostMeshDescriptorCount() {
  return &h_meshDescriptorCount;
}

int* ModelHandler::getHostMeshSegmentCount() {
  return &h_meshSegmentCount;
}

void ModelHandler::updateDeviceMesh() {
  std::vector<MeshDescriptor> h_meshDescriptorList = getCollectiveMeshDescriptorList();
  std::vector<MeshSegment> h_meshSegmentList = getCollectiveMeshSegmentList();

  h_meshDescriptorCount = h_meshDescriptorList.size();
  h_meshSegmentCount = h_meshSegmentList.size();

  cudaMalloc(&d_meshDescriptorBuffer, h_meshDescriptorCount*sizeof(MeshDescriptor));
  cudaMalloc(&d_meshSegmentBuffer, h_meshSegmentCount*sizeof(MeshSegment));

  cudaMemcpy(d_meshDescriptorBuffer, &h_meshDescriptorList[0], h_meshDescriptorCount*sizeof(MeshDescriptor), cudaMemcpyHostToDevice);
  cudaMemcpy(d_meshSegmentBuffer, &h_meshSegmentList[0], h_meshSegmentCount*sizeof(MeshSegment), cudaMemcpyHostToDevice);
}

RasterModel* ModelHandler::createRasterModel(GLuint* shaderProgramHandle, int index) {
  RasterModel* rasterModel = new RasterModel(shaderProgramHandle, modelList[index]);
  return rasterModel;
}

Model* ModelHandler::createReducedModel(int index) {
  return modelList[index]->createReducedModel();
}

void ModelHandler::createReducedOBJ(const char* source, const char* target) {
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