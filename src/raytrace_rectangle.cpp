#include "raytrace_rectangle.h"

RaytraceRectangle::RaytraceRectangle(GLuint* shaderProgramHandle, ModelContainer* modelContainer) {
  imageResolution = 50;

  vertices[0] = 0.0;             vertices[1] = 0.0;
  vertices[2] = SCREEN_WIDTH;    vertices[3] = 0.0;
  vertices[4] = 0.0;             vertices[5] = SCREEN_HEIGHT;
  vertices[6] = 0.0;             vertices[7] = SCREEN_HEIGHT;
  vertices[8] = SCREEN_WIDTH;    vertices[9] = 0.0;
  vertices[10] = SCREEN_WIDTH;   vertices[11] = SCREEN_HEIGHT;

  textureCoordinates[0] =  0.0;   textureCoordinates[1] =  1.0;
  textureCoordinates[2] =  1.0;   textureCoordinates[3] =  1.0;
  textureCoordinates[4] =  0.0;   textureCoordinates[5] =  0.0;
  textureCoordinates[6] =  0.0;   textureCoordinates[7] =  0.0;
  textureCoordinates[8] =  1.0;   textureCoordinates[9] =  1.0;
  textureCoordinates[10] = 1.0;   textureCoordinates[11] = 0.0;

  this->shaderProgramHandle = shaderProgramHandle;

  image = new RaytraceImage(modelContainer);
  initializeImage(imageResolution, imageResolution);

  glGenVertexArrays(1, &vao);
  glGenBuffers(2, vbo);

  glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
  glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(float), vertices, GL_STATIC_DRAW);

  glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
  glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(float), textureCoordinates, GL_STATIC_DRAW);

  textureLocationHandle = glGetUniformLocation(*this->shaderProgramHandle, "u_texture");
  resolutionLocationHandle = glGetUniformLocation(*this->shaderProgramHandle, "u_resolution");
}

RaytraceRectangle::~RaytraceRectangle() {
  delete image;
}

int RaytraceRectangle::getImageResolution() {
  return imageResolution;
}

void RaytraceRectangle::update(Camera* camera, DirectionalLight* directionalLight) {
  image->update(camera, directionalLight);
}

void RaytraceRectangle::incrementResolution() {
  imageResolution += 50;
  initializeImage(imageResolution, imageResolution);
}

void RaytraceRectangle::decrementResolution() {
  if (imageResolution > 50) { imageResolution -= 50; }
  initializeImage(imageResolution, imageResolution);
}

void RaytraceRectangle::render() {
  image->render();

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, textureResource);

  glUseProgram(*shaderProgramHandle);

  glBindVertexArray(vao);

  glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
  glEnableVertexAttribArray(0);

  glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);
  glEnableVertexAttribArray(1);

  glUniform1i(textureLocationHandle, 0);
  glUniform2f(resolutionLocationHandle, SCREEN_WIDTH, SCREEN_HEIGHT);

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