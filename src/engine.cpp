#include "engine.h"

extern "C" void updateLight(float x, float y, float z);
extern "C" void updateCamera(float x, float y, float z, float rotationX, float rotationY) ;
extern "C" void initializeScene();
extern "C" void renderFrame(int blockDimX, int blockDimY, void* cudaBuffer, cudaGraphicsResource_t* cudaTextureResource);

GLfloat vertices[] = {
  -1.0, -1.0,
   1.0, -1.0,
  -1.0,  1.0,
  -1.0,  1.0,
   1.0, -1.0,
   1.0,  1.0,
};

GLfloat textureCoordinates[] = {
  0.0, 1.0,
  1.0, 1.0,
  0.0, 0.0,
  0.0, 0.0,
  1.0, 1.0,
  1.0, 0.0,
};

struct Tuple {
  float x;
  float y;
  float z;
  float w;
};

Tuple cameraPositionVelocity = {0.0, 0.0, 0.0, 0.0};
Tuple cameraPosition = {5.0, -3.5, -6.0, 1.0};
Tuple cameraRotationVelocity = {0.0, 0.0, 0.0, 0.0};
Tuple cameraRotation = {-M_PI / 12.0, -M_PI / 4.5, 0.0, 0.0};

Tuple lightPositionVelocity = {0.0, 0.0, 0.0, 0.0};
Tuple lightPosition = {10.0, -10.0, -5.0, 1.0};

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

  glGenTextures(1, &textureResource);
  glBindTexture(GL_TEXTURE_2D, textureResource);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT, 1000, 1000, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE, NULL);

  cudaGraphicsGLRegisterImage(&cudaTextureResource, textureResource, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
  cudaMalloc(&cudaBuffer, 1000*1000*4*sizeof(GLubyte));

  initializeScene();

  glGenVertexArrays(1, &vao);
  glGenBuffers(2, vbo);

  textureHandle = glGetUniformLocation(shaderProgramHandle, "u_texture");
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
  cameraPositionVelocity = {0.0, 0.0, 0.0, 0.0};
  cameraRotationVelocity = {0.0, 0.0, 0.0, 0.0};

  lightPositionVelocity = {0.0, 0.0, 0.0, 0.0};

  if (input->checkKeyDown(87)) {
    cameraPositionVelocity.x += cos(-cameraRotation.y + (M_PI / 2)) * 0.1;
    cameraPositionVelocity.z += sin(-cameraRotation.y + (M_PI / 2)) * 0.1;
  }
  if (input->checkKeyDown(83)) {
    cameraPositionVelocity.x += -cos(-cameraRotation.y + (M_PI / 2)) * 0.1;
    cameraPositionVelocity.z += -sin(-cameraRotation.y + (M_PI / 2)) * 0.1;
  }
  if (input->checkKeyDown(65)) {
    cameraPositionVelocity.x += -cos(-cameraRotation.y) * 0.1;
    cameraPositionVelocity.z += -sin(-cameraRotation.y) * 0.1;
  }
  if (input->checkKeyDown(68)) {
    cameraPositionVelocity.x += cos(-cameraRotation.y) * 0.1;
    cameraPositionVelocity.z += sin(-cameraRotation.y) * 0.1;
  }
  if (input->checkKeyDown(341)) {
    cameraPositionVelocity.y += 0.05;
  }
  if (input->checkKeyDown(32)) {
    cameraPositionVelocity.y += -0.05;
  }
  if (input->checkKeyDown(82)) {
    cameraRotationVelocity.x += 0.02;
  }
  if (input->checkKeyDown(70)) {
    cameraRotationVelocity.x += -0.02;
  }
  if (input->checkKeyDown(69)) {
    cameraRotationVelocity.y += 0.02;
  }
  if (input->checkKeyDown(81)) {
    cameraRotationVelocity.y += -0.02;
  }

  if (input->checkKeyDown(265)) {
    lightPositionVelocity.x += -0.05;
  }
  if (input->checkKeyDown(264)) {
    lightPositionVelocity.x += 0.05;
  }
  if (input->checkKeyDown(263)) {
    lightPositionVelocity.z += -0.05;
  }
  if (input->checkKeyDown(262)) {
    lightPositionVelocity.z += 0.05;
  }

  cameraPosition.x += cameraPositionVelocity.x;
  cameraPosition.y += cameraPositionVelocity.y;
  cameraPosition.z += cameraPositionVelocity.z;
  cameraRotation.x += cameraRotationVelocity.x;
  cameraRotation.y += cameraRotationVelocity.y;
  cameraRotation.z += cameraRotationVelocity.z;

  lightPosition.x += lightPositionVelocity.x;
  lightPosition.y += lightPositionVelocity.y;
  lightPosition.z += lightPositionVelocity.z;

  updateCamera(cameraPosition.x, cameraPosition.y, cameraPosition.z, cameraRotation.x, cameraRotation.y);
  updateLight(lightPosition.x, lightPosition.y, lightPosition.z);
}

void Engine::render() {
  renderFrame(16, 16, cudaBuffer, &cudaTextureResource);

  glClearColor(0.0, 0.0, 0.0, 0.0);
  glClear(GL_COLOR_BUFFER_BIT);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, textureResource);

  glUseProgram(shaderProgramHandle);

  glBindVertexArray(vao);

  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
  glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(float), vertices, GL_STATIC_DRAW);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

  glEnableVertexAttribArray(1);
  glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
  glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(float), textureCoordinates, GL_STATIC_DRAW);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);

  glUniform1i(textureHandle, 0);

  glDrawArrays(GL_TRIANGLES, 0, 6);

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