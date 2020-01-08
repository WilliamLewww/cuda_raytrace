#include "shader_handler.h"

ShaderHandler::ShaderHandler() {

}

ShaderHandler::~ShaderHandler() {

}

GLuint* ShaderHandler::getShaderFromName(const char* name) {
  for (int x = 0; x < shaderProgramList.size(); x++) {
    if (strcmp(shaderProgramList[x].first.c_str(), name) == 0) {
      return &shaderProgramList[x].second;
    }
  }

  return nullptr;
}

std::string ShaderHandler::readShaderSource(const char* filepath) {
  std::string content;
  std::ifstream fileStream(filepath, std::ios::in);
  std::string line = "";

  while (!fileStream.eof()) {
      getline(fileStream, line);
      content.append(line + "\n");
  }
  fileStream.close();

  return content;
}

GLuint ShaderHandler::createShaderProgram(std::string vertexShaderString, std::string fragmentShaderString) {
  const char* vertexShaderSource = vertexShaderString.c_str();
  const char* fragmentShaderSource = fragmentShaderString.c_str();

  GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
  GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

  glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
  glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);

  glCompileShader(vertexShader);
  glCompileShader(fragmentShader);

  GLuint shaderProgram = glCreateProgram();
  glAttachShader(shaderProgram, vertexShader);
  glAttachShader(shaderProgram, fragmentShader);
  glLinkProgram(shaderProgram);

  return shaderProgram;
}

void ShaderHandler::addShaderProgram(std::string filename) {
  std::string vertexShaderString = readShaderSource((filename + ".vertex").c_str());
  std::string fragmentShaderString = readShaderSource((filename + ".fragment").c_str());

  std::string shaderName = filename.substr(filename.find_last_of("/") + 1);

  shaderProgramList.push_back(std::make_pair(shaderName, createShaderProgram(vertexShaderString, fragmentShaderString)));
}