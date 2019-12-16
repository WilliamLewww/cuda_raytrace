#include "engine.h"

void Engine::initialize() {
  glfwInit();
  window = glfwCreateWindow(1000, 1000, "cuda_raytrace", NULL, NULL);
  glfwMakeContextCurrent(window);
  glewInit();

  input = new Input();
  glfwSetKeyCallback(window, Input::keyCallback);

  std::string vertexShaderString = readShaderSource("shaders/basic.vertex");
  std::string fragmentShaderString = readShaderSource("shaders/basic.fragment");
  shaderProgramHandle = createShaderProgram(vertexShaderString, fragmentShaderString);

  joiner = new Joiner();
  joiner->initialize(&shaderProgramHandle);
}

void Engine::run() {
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    update();
    render();
  }
}

void Engine::exit() {
  glfwDestroyWindow(window);
  glfwTerminate();
}

void Engine::update() {
  joiner->update();
}

void Engine::render() {
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glClear(GL_COLOR_BUFFER_BIT);

  joiner->render();

  glfwSwapBuffers(window);
}

std::string Engine::readShaderSource(const char* filepath) {
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

GLuint Engine::createShaderProgram(std::string vertexShaderString, std::string fragmentShaderString) {
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