#include "raytrace_image.h"

extern "C" {
  void initializeScene();
  
  void updateCamera(float x, float y, float z, float rotationX, float rotationY);
  void updateScene();

  void renderFrame(int blockDimX, int blockDimY, void* cudaBuffer, cudaGraphicsResource_t* cudaTextureResource, int frameWidth, int frameHeight, Tuple* lightingBuffer, Tuple* reflectionsBuffer);
  void renderImage(int blockDimX, int blockDimY, const char* filename);
}

void RaytraceImage::initialize() {
  frameWidth = 250;
  frameHeight = 250;

  cameraPositionX = 5.0; cameraPositionY = -3.5; cameraPositionZ = -6.0;
  cameraRotationX = -M_PI / 12.0; cameraRotationY = -M_PI / 4.5;

  initializeScene();
}

void RaytraceImage::updateResolution(int width, int height, GLuint textureResource) {
  frameWidth = width;
  frameHeight = height;

  cudaGraphicsGLRegisterImage(&cudaTextureResource, textureResource, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

  cudaFree(cudaBuffer);
  cudaMalloc(&cudaBuffer, frameWidth*frameHeight*4*sizeof(GLubyte));

  cudaFree(lightingBuffer);
  cudaFree(reflectionsBuffer);
  cudaMalloc(&lightingBuffer, frameWidth*frameHeight*sizeof(Tuple));
  cudaMalloc(&reflectionsBuffer, frameWidth*frameHeight*sizeof(Tuple));
}

void RaytraceImage::update() {
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

  if (Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_TRIGGER) > -0.92) {
    cameraPositionY += (Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_LEFT_TRIGGER) + 1.0) * -0.03;
  }

  if (Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER) > -0.92) {
    cameraPositionY += (Input::checkGamepadAxis(GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER) + 1.0) * 0.03;
  }

  if (Input::checkGamepadButtonDown(GLFW_GAMEPAD_BUTTON_CIRCLE) && !shouldTakePhoto) {
    shouldTakePhoto = true;
  }

  if (!Input::checkGamepadButtonDown(GLFW_GAMEPAD_BUTTON_CIRCLE) && shouldTakePhoto) {
    renderImage(16, 16, "image.ppm");
    shouldTakePhoto = false;
  }
  
  updateCamera(cameraPositionX, cameraPositionY, cameraPositionZ, cameraRotationX, cameraRotationY);
  updateScene();
}

void RaytraceImage::render() {
  renderFrame(16, 16, cudaBuffer, &cudaTextureResource, frameWidth, frameHeight, lightingBuffer, reflectionsBuffer);
}