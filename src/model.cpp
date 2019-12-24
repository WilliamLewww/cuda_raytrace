#include "model.h"

Model createModelFromOBJ(const char* filename) {
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

  model.triangleCount = model.indexList.size() / 3;
  model.triangleArray = new Triangle[model.triangleCount];
  for (int x = 0; x < model.triangleCount; x++) {
    model.triangleArray[x].vertexA = model.vertexList[model.indexList[(3 * x)].x - 1];
    model.triangleArray[x].vertexB = model.vertexList[model.indexList[(3 * x) + 1].x - 1];
    model.triangleArray[x].vertexC = model.vertexList[model.indexList[(3 * x) + 2].x - 1];

    model.triangleArray[x].normal = model.normalList[model.indexList[(3 * x)].z - 1];

    model.triangleArray[x].color = {float(int(45.0 * x + 87) % 255), float(int(77.0 * x + 102) % 255), float(int(123.0 * x + 153) % 255), 1.0};
  }

  return model;
}