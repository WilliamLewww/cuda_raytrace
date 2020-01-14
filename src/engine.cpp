#include "engine.h"

Engine::Engine() {
  glfwInit();
  window = glfwCreateWindow(1000, 1000, "cuda_raytrace", NULL, NULL);
  glfwMakeContextCurrent(window);
  glewInit();

  glfwSetKeyCallback(window, Input::keyCallback);
  
  fontHandler = new FontHandler();
  fontHandler->addFontFromFile("res/font_ubuntu");

  shaderHandler = new ShaderHandler();
  shaderHandler->addShaderProgram("shaders/textured_rectangle");
  shaderHandler->addShaderProgram("shaders/colored_model");

  modelHandler = new ModelHandler();

  joiner = new Joiner(shaderHandler, fontHandler, modelHandler);
}

Engine::~Engine() {
  delete joiner;
  delete modelHandler;
  delete shaderHandler;
  delete fontHandler;

  glfwDestroyWindow(window);
  glfwTerminate();
}

float frameStart, frameEnd, deltaTime;
void Engine::run() {
  frameEnd = glfwGetTime();

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
  joiner->update(deltaTime);
  Input::refreshInput();
}

void Engine::render() {
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  joiner->render();

  glfwSwapBuffers(window);
}