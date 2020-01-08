#include "raytrace_image.h"

extern "C" {
  void initializeScene(int* h_meshDescriptorCount, int* h_meshSegmentCount);
  
  void updateCamera(float x, float y, float z, float rotationX, float rotationY);
  void updateScene();

  void renderFrame(int blockDimX, int blockDimY, void* colorBuffer, cudaGraphicsResource_t* cudaTextureResource, int frameWidth, int frameHeight, Tuple* lightingBuffer, Tuple* reflectionsBuffer, MeshDescriptor* meshDescriptorBuffer, MeshSegment* meshSegmentBuffer);
  void renderImage(int blockDimX, int blockDimY, const char* filename, MeshDescriptor* meshDescriptorBuffer, MeshSegment* meshSegmentBuffer);
}

void RaytraceImage::initialize() {
  frameWidth = 250;
  frameHeight = 250;

  cameraPositionX = 5.0; cameraPositionY = -3.5; cameraPositionZ = -6.0;
  cameraRotationX = -M_PI / 12.0; cameraRotationY = -M_PI / 4.5;

  modelList.push_back(createModelFromOBJ("res/cube.obj", 1));
  modelList.push_back(createModelFromOBJ("res/donut.obj", 0));
  initializeModelMatrix(&modelList[0].meshDescriptor, createScaleMatrix(5.0, 0.15, 5.0));
  initializeModelMatrix(&modelList[1].meshDescriptor, createTranslateMatrix(0.0, -2.0, 0.0));

  std::vector<MeshDescriptor> h_meshDescriptorList;
  std::vector<MeshSegment> h_meshSegmentList;

  for (int x = 0; x < modelList.size(); x++) {
    h_meshDescriptorList.push_back(modelList[x].meshDescriptor);

    for (int y = 0; y < h_meshDescriptorList[x].segmentCount; y++) {
      h_meshSegmentList.push_back(modelList[x].meshSegmentArray[y]);
    }
  }

  h_meshDescriptorCount = h_meshDescriptorList.size();
  h_meshSegmentCount = h_meshSegmentList.size();

  cudaMalloc(&meshDescriptorBuffer, h_meshDescriptorCount*sizeof(MeshDescriptor));
  cudaMalloc(&meshSegmentBuffer, h_meshSegmentCount*sizeof(MeshSegment));

  cudaMemcpy(meshDescriptorBuffer, &h_meshDescriptorList[0], h_meshDescriptorCount*sizeof(MeshDescriptor), cudaMemcpyHostToDevice);
  cudaMemcpy(meshSegmentBuffer, &h_meshSegmentList[0], h_meshSegmentCount*sizeof(MeshSegment), cudaMemcpyHostToDevice);

  initializeScene(&h_meshDescriptorCount, &h_meshSegmentCount);
}

void RaytraceImage::updateResolution(int width, int height, GLuint textureResource) {
  frameWidth = width;
  frameHeight = height;

  cudaGraphicsGLRegisterImage(&cudaTextureResource, textureResource, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

  cudaFree(colorBuffer);
  cudaMalloc(&colorBuffer, frameWidth*frameHeight*4*sizeof(GLubyte));

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
    renderImage(16, 16, "image.ppm", meshDescriptorBuffer, meshSegmentBuffer);
    shouldTakePhoto = false;
  }
  
  updateCamera(cameraPositionX, cameraPositionY, cameraPositionZ, cameraRotationX, cameraRotationY);
  updateScene();
}

void RaytraceImage::render() {
  renderFrame(16, 16, colorBuffer, &cudaTextureResource, frameWidth, frameHeight, lightingBuffer, reflectionsBuffer, meshDescriptorBuffer, meshSegmentBuffer);
}