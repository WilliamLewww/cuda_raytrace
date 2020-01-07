#include "model.h"

Model createReducedOBJ(const char* source, const char* target) {
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

Model createModelFromOBJ(const char* filename, int reflective) {
  Model model;

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

      model.vertexList.push_back({x, y, z, 1.0});
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

      model.normalList.push_back({x, y, z, 0.0});
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

      model.indexList.push_back(a);
      model.indexList.push_back(b);
      model.indexList.push_back(c);
    }
  }

  file.close();

  model.meshDescriptor.segmentCount = model.indexList.size() / 3;
  model.meshSegmentArray = new MeshSegment[model.meshDescriptor.segmentCount];
  for (int x = 0; x < model.meshDescriptor.segmentCount; x++) {
    model.meshSegmentArray[x].vertexA = model.vertexList[model.indexList[(3 * x)].x - 1];
    model.meshSegmentArray[x].vertexB = model.vertexList[model.indexList[(3 * x) + 1].x - 1];
    model.meshSegmentArray[x].vertexC = model.vertexList[model.indexList[(3 * x) + 2].x - 1];

    model.meshSegmentArray[x].normal = model.normalList[model.indexList[(3 * x)].z - 1];

    model.meshSegmentArray[x].color = {float(int(45.0 * x + 87) % 255), float(int(77.0 * x + 102) % 255), float(int(123.0 * x + 153) % 255), 1.0};
  }

  model.meshDescriptor.reflective = reflective;

  return model;
}

void initializeModelMatrix(MeshDescriptor* meshDescriptor, float* matrix) {
  float* modelMatrix = meshDescriptor->modelMatrix;
  for (int x = 0; x < 16; x++) { modelMatrix[x] = matrix[x]; }

  modelMatrix = meshDescriptor->inverseModelMatrix;
  float* inverseModelMatrix = inverseMatrix(meshDescriptor->modelMatrix);
  for (int x = 0; x < 16; x++) { modelMatrix[x] = inverseModelMatrix[x]; }
}