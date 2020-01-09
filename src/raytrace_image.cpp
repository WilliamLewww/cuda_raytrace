#include "raytrace_image.h"

extern "C" {
  void initializeScene(int* h_meshDescriptorCount, int* h_meshSegmentCount);
  
  void updateCamera(float x, float y, float z, float rotationX, float rotationY);
  void updateScene();

  void renderFrame(int blockDimX, int blockDimY, void* d_colorBuffer, cudaGraphicsResource_t* cudaTextureResource, int frameWidth, int frameHeight, Tuple* d_lightingBuffer, Tuple* d_reflectionsBuffer, MeshDescriptor* d_meshDescriptorBuffer, MeshSegment* d_meshSegmentBuffer);
  void renderImage(int blockDimX, int blockDimY, const char* filename, int imageWidth, int imageHeight, MeshDescriptor* d_meshDescriptorBuffer, MeshSegment* d_meshSegmentBuffer);
}

RaytraceImage::RaytraceImage() {
  frameWidth = 250;
  frameHeight = 250;
  
  imageWidth = 5000;
  imageHeight = 5000;

  cameraPositionX = 5.0; cameraPositionY = -3.5; cameraPositionZ = -6.0;
  cameraRotationX = -M_PI / 12.0; cameraRotationY = -M_PI / 4.5;

  modelList.push_back(new Model("res/cube.obj", 1));
  modelList.push_back(new Model("res/donut.obj", 0));
  // initializeModelMatrix(&modelList[0]->meshDescriptor, createScaleMatrix(5.0, 0.15, 5.0));
  // initializeModelMatrix(&modelList[1]->meshDescriptor, createTranslateMatrix(0.0, -2.0, 0.0));

  std::vector<MeshDescriptor> h_meshDescriptorList;
  std::vector<MeshSegment> h_meshSegmentList;

  for (int x = 0; x < modelList.size(); x++) {
    h_meshDescriptorList.push_back(modelList[x]->createMeshDescriptor());
    
    std::vector<MeshSegment> tempMeshSegmentList = modelList[x]->createMeshSegmentList();
    h_meshSegmentList.insert(h_meshSegmentList.end(), tempMeshSegmentList.begin(), tempMeshSegmentList.end());
  }

  h_meshDescriptorCount = h_meshDescriptorList.size();
  h_meshSegmentCount = h_meshSegmentList.size();

  cudaMalloc(&d_meshDescriptorBuffer, h_meshDescriptorCount*sizeof(MeshDescriptor));
  cudaMalloc(&d_meshSegmentBuffer, h_meshSegmentCount*sizeof(MeshSegment));

  cudaMemcpy(d_meshDescriptorBuffer, &h_meshDescriptorList[0], h_meshDescriptorCount*sizeof(MeshDescriptor), cudaMemcpyHostToDevice);
  cudaMemcpy(d_meshSegmentBuffer, &h_meshSegmentList[0], h_meshSegmentCount*sizeof(MeshSegment), cudaMemcpyHostToDevice);

  initializeScene(&h_meshDescriptorCount, &h_meshSegmentCount);
}

RaytraceImage::~RaytraceImage() {
  cudaFree(d_colorBuffer);
  cudaFree(d_reflectionsBuffer);
  cudaFree(d_lightingBuffer);

  cudaFree(d_meshSegmentBuffer);
  cudaFree(d_meshDescriptorBuffer);
}

void RaytraceImage::updateResolution(int width, int height, GLuint textureResource) {
  frameWidth = width;
  frameHeight = height;

  cudaGraphicsGLRegisterImage(&cudaTextureResource, textureResource, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

  cudaFree(d_colorBuffer);
  cudaMalloc(&d_colorBuffer, frameWidth*frameHeight*4*sizeof(GLubyte));

  cudaFree(d_lightingBuffer);
  cudaFree(d_reflectionsBuffer);
  cudaMalloc(&d_lightingBuffer, frameWidth*frameHeight*sizeof(Tuple));
  cudaMalloc(&d_reflectionsBuffer, frameWidth*frameHeight*sizeof(Tuple));
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
    renderImage(16, 16, "image.ppm", imageWidth, imageHeight, d_meshDescriptorBuffer, d_meshSegmentBuffer);
    shouldTakePhoto = false;
  }
  
  updateCamera(cameraPositionX, cameraPositionY, cameraPositionZ, cameraRotationX, cameraRotationY);
  updateScene();
}

void RaytraceImage::render() {
  renderFrame(16, 16, d_colorBuffer, &cudaTextureResource, frameWidth, frameHeight, d_lightingBuffer, d_reflectionsBuffer, d_meshDescriptorBuffer, d_meshSegmentBuffer);
}