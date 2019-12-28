#include "raytrace_rectangle.h"

void RaytraceRectangle::initialize(GLuint* shaderProgramHandle) {
  imageResolution = 250;

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

  image = new RaytraceImage();
  image->initialize();
  initializeImage(imageResolution, imageResolution);

  glGenVertexArrays(1, &vao);
  glGenBuffers(2, vbo);

  textureHandle = glGetUniformLocation(*this->shaderProgramHandle, "u_texture");
}

void RaytraceRectangle::update() {
  if (Input::checkGamepadButtonDown(GLFW_GAMEPAD_BUTTON_TRIANGLE) && !shouldIncreaseImageResolution) {
    shouldIncreaseImageResolution = true;
  }

  if (!Input::checkGamepadButtonDown(GLFW_GAMEPAD_BUTTON_TRIANGLE) && shouldIncreaseImageResolution) {
    imageResolution += 50;
    initializeImage(imageResolution, imageResolution);
    shouldIncreaseImageResolution = false;
  }

  if (Input::checkGamepadButtonDown(GLFW_GAMEPAD_BUTTON_CROSS) && !shouldDecreaseImageResolution) {
    shouldDecreaseImageResolution = true;
  }

  if (!Input::checkGamepadButtonDown(GLFW_GAMEPAD_BUTTON_CROSS) && shouldDecreaseImageResolution) {
    if (imageResolution > 50) { imageResolution -= 50; }
    initializeImage(imageResolution, imageResolution);
    shouldDecreaseImageResolution = false;
  }

  image->update();
}

void RaytraceRectangle::render() {
  image->render();

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

  glBindTexture(GL_TEXTURE_2D, 0);
}

void RaytraceRectangle::initializeImage(int width, int height) {
  glGenTextures(1, &textureResource);
  glBindTexture(GL_TEXTURE_2D, textureResource);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT, width, height, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE, NULL);

  image->updateResolution(width, height, textureResource);

  glBindTexture(GL_TEXTURE_2D, 0);
}