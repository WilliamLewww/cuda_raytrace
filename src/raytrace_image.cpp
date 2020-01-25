#include "raytrace_image.h"

extern "C" {
  void initializeScene(int* h_meshDescriptorCount, int* h_meshSegmentCount);
  
  void updateDirectionalLight(float x, float y, float z, float red, float green, float blue);
  void updateCudaCamera(float x, float y, float z, float pitch, float yaw);
  void updateScene();

  void renderFrame(int blockDimX, int blockDimY, void* d_colorBuffer, cudaGraphicsResource_t* cudaTextureResource, int frameWidth, int frameHeight, Tuple* d_lightingBuffer, Tuple* d_reflectionsBuffer, MeshDescriptor* d_meshDescriptorBuffer, MeshSegment* d_meshSegmentBuffer);
  void renderImage(int blockDimX, int blockDimY, const char* filename, int imageWidth, int imageHeight, MeshDescriptor* d_meshDescriptorBuffer, MeshSegment* d_meshSegmentBuffer);
}

RaytraceImage::RaytraceImage(ModelContainer* modelContainer) {
  this->modelContainer = modelContainer;
  modelContainer->updateDeviceMesh();

  initializeScene(modelContainer->getHostMeshDescriptorCount(), modelContainer->getHostMeshSegmentCount());

  frameWidth = 250;
  frameHeight = 250;
  
  imageWidth = 1000;
  imageHeight = 1000;
}

RaytraceImage::~RaytraceImage() {
  cudaFree(d_colorBuffer);
  cudaFree(d_reflectionsBuffer);
  cudaFree(d_lightingBuffer);
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

void RaytraceImage::update(Camera* camera, DirectionalLight* directionalLight) {
  if (Input::checkCirclePressed()) {
    renderImage(16, 16, "image.png", imageWidth, imageHeight, modelContainer->getDeviceMeshDescriptorBuffer(), modelContainer->getDeviceMeshSegmentBuffer());
  }

  Tuple cameraPosition = camera->getPosition();
  updateDirectionalLight(directionalLight->position.x, directionalLight->position.y, directionalLight->position.z, directionalLight->intensity.x, directionalLight->intensity.y, directionalLight->intensity.z);
  updateCudaCamera(cameraPosition.x, cameraPosition.y, cameraPosition.z, camera->getPitch(), camera->getYaw());
  updateScene();
}

void RaytraceImage::render() {
  renderFrame(16, 16, d_colorBuffer, &cudaTextureResource, frameWidth, frameHeight, d_lightingBuffer, d_reflectionsBuffer, modelContainer->getDeviceMeshDescriptorBuffer(), modelContainer->getDeviceMeshSegmentBuffer());
}