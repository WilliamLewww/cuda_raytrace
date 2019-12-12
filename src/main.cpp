#include <stdlib.h>
#include <fstream>
#include <string>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

extern "C" void renderImage(int blockDimX, int blockDimY, const char* filename);

std::string readShaderSource(const char* filepath);
GLuint createShaderProgram(std::string vertexShaderString, std::string fragmentShaderString);

int main(int argn, char** argv) {
  glfwInit();

  GLFWwindow* window = glfwCreateWindow(1000, 1000, "cuda_raytrace", NULL, NULL);
  glfwMakeContextCurrent(window);
  glewInit();

  GLuint shaderProgramHandle;
  std::string vertexShaderString = readShaderSource("shaders/basic.vertex");
  std::string fragmentShaderString = readShaderSource("shaders/basic.fragment");
  shaderProgramHandle = createShaderProgram(vertexShaderString, fragmentShaderString);

  while (!glfwWindowShouldClose(window)) {
    glClear(GL_COLOR_BUFFER_BIT);
    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glfwTerminate();
  return 0;
}

std::string readShaderSource(const char* filepath) {
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

GLuint createShaderProgram(std::string vertexShaderString, std::string fragmentShaderString) {
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