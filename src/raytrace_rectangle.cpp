#include "raytrace_rectangle.h"

extern "C" {
  void initializeScene();
  void renderFrame(int blockDimX, int blockDimY, void* cudaBuffer, cudaGraphicsResource_t* cudaTextureResource);
  void updateCamera(float x, float y, float z, float rotationX, float rotationY);
}

void RaytraceRectangle::initialize(GLuint* shaderProgramHandle) {
  cameraPositionX = 5.0; cameraPositionY = -3.5; cameraPositionZ = -6.0;
  cameraRotationX = -M_PI / 12.0; cameraRotationY = -M_PI / 4.5;

  vertices[0] = -1.0;   vertices[1] = -1.0;
  vertices[2] =  1.0;   vertices[3] = -1.0;
  vertices[4] = -1.0;   vertices[5] =  1.0;
  vertices[6] = -1.0;   vertices[7] =  1.0;
  vertices[8] =  1.0;   vertices[9] = -1.0;
  vertices[10] = 1.0;   vertices[11] = 1.0;

  textureCoordinates[0] =  0.0;   textureCoordinates[1] =  1.0;
  textureCoordinates[2] =  1.0;   textureCoordinates[3] =  1.0;
  textureCoordinates[4] =  0.0;   textureCoordinates[5] =  0.0;
  textureCoordinates[6] =  0.0;   textureCoordinates[7] =  0.0;
  textureCoordinates[8] =  1.0;   textureCoordinates[9] =  1.0;
  textureCoordinates[10] = 1.0;   textureCoordinates[11] = 0.0;

  this->shaderProgramHandle = shaderProgramHandle;

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

  textureHandle = glGetUniformLocation(*this->shaderProgramHandle, "u_texture");
}

void RaytraceRectangle::update() {
  if (abs(Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_X)) > 0.08) {
    cameraPositionX += cos(-cameraRotationY) * Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_X) * 0.05;
    cameraPositionZ += sin(-cameraRotationY) * Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_X) * 0.05;
  }
  if (abs(Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_Y)) > 0.08) {
    cameraPositionX += cos(-cameraRotationY + (M_PI / 2)) * Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_Y) * -0.05;
    cameraPositionZ += sin(-cameraRotationY + (M_PI / 2)) * Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_Y) * -0.05;
  }

  if (abs(Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_X)) > 0.08) {
    cameraRotationY += Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_X) * 0.03;
  }

  if (abs(Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_Y)) > 0.08) {
    cameraRotationX += Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_Y) * -0.03;
  }

  updateCamera(cameraPositionX, cameraPositionY, cameraPositionZ, cameraRotationX, cameraRotationY);
}

void RaytraceRectangle::render() {
  renderFrame(16, 16, cudaBuffer, &cudaTextureResource);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, textureResource);

  glUseProgram(*shaderProgramHandle);

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
}