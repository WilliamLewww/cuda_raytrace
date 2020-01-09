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

  modelHandler = new ModelHandler();
  modelHandler->addModel("res/cube.obj", 1);
  modelHandler->addModel("res/donut.obj", 0);
  modelHandler->setModelMatrix(0, createScaleMatrix(5.0, 0.15, 5.0));
  modelHandler->setModelMatrix(1, createTranslateMatrix(0.0, -2.0, 0.0));

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

void Engine::run() {
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    update();
    render();
  }
}

void Engine::update() {
  joiner->update();
  Input::refreshInput();
}

void Engine::render() {
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glClear(GL_COLOR_BUFFER_BIT);

  joiner->render();

  glfwSwapBuffers(window);
}