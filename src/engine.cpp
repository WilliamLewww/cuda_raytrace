#include "engine.h"

Engine::Engine() {
  glfwInit();

  glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
  window = glfwCreateWindow(RESOLUTION_WIDTH, RESOLUTION_HEIGHT, "cuda_raytrace", NULL, NULL);
  
  glfwMakeContextCurrent(window);
  glewInit();

  glfwSetKeyCallback(window, Input::keyCallback);
  glfwSetCursorPosCallback(window, Input::cursorPositionCallback);
  glfwSetMouseButtonCallback(window, Input::cursorButtonCallback);
  
  fontHandler = new FontHandler();
  fontHandler->addFontFromFile("res/font_ubuntu");

  shaderHandler = new ShaderHandler();
  shaderHandler->addShaderProgram("src/shaders/textured_rectangle");
  shaderHandler->addShaderProgram("src/shaders/random_colored_model");
  shaderHandler->addShaderProgram("src/shaders/colored_rectangle");
  shaderHandler->addShaderProgram("src/shaders/random_colored_phong_model");

  joiner = new Joiner(shaderHandler, fontHandler);

  frameStart = glfwGetTime();
  frameEnd = glfwGetTime();
  deltaTime = 0.0;
}

Engine::~Engine() {
  delete joiner;
  delete shaderHandler;
  delete fontHandler;

  glfwDestroyWindow(window);
  glfwTerminate();
}

void Engine::run() {
  while (!glfwWindowShouldClose(window)) {
    frameStart = glfwGetTime();
    deltaTime = frameStart - frameEnd;

    update(deltaTime);
    render();

    frameEnd = frameStart;
  }
}

void Engine::update(float deltaTime) {
  glfwPollEvents();
  Input::checkControllerPresses();

  joiner->update(deltaTime);
  
  Input::refreshInput();
}

void Engine::render() {
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  joiner->render();

  glfwSwapBuffers(window);
}