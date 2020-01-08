#pragma once
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <utility>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

class ShaderHandler {
private:
  std::vector<std::pair<std::string, GLuint>> shaderProgramList;

  std::string readShaderSource(const char* filepath);
  GLuint createShaderProgram(std::string vertexShaderString, std::string fragmentShaderString);
public:
  ShaderHandler();
  ~ShaderHandler();
  
  GLuint* getShaderFromName(const char* name);

  void addShaderProgram(std::string filename);
};