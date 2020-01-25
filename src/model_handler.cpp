#include "model_handler.h"

Model* ModelHandler::createModel(const char* filename, int reflective = 0) {
  Model* model = new Model(filename, reflective);
  return model;
}

Model* ModelHandler::createModel(Model* model) {
  Model* cloneModel = new Model(*model);
  return cloneModel;
}

RasterModel* ModelHandler::createRasterModel(RasterModelType rasterModelType, GLuint* shaderProgramHandle, Model* model) {
  RasterModel* rasterModel = nullptr;

  if (rasterModelType == RASTERMODELTYPE_RANDOM) {
    rasterModel = new RasterModel(shaderProgramHandle, model);
  }

  if (rasterModelType == RASTERMODELTYPE_RANDOM_PHONG) {
    rasterModel = new PhongRasterModel(shaderProgramHandle, model);
  }

  return rasterModel;
}

Model* ModelHandler::createReducedModel(Model* model) {
  return model->createReducedModel();
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