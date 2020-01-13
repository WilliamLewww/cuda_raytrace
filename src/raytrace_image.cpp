#include "raytrace_image.h"

extern "C" {
  void initializeScene(int* h_meshDescriptorCount, int* h_meshSegmentCount);
  
  void updateCudaCamera(float x, float y, float z, float pitch, float yaw);
  void updateScene();

  void renderFrame(int blockDimX, int blockDimY, void* d_colorBuffer, cudaGraphicsResource_t* cudaTextureResource, int frameWidth, int frameHeight, Tuple* d_lightingBuffer, Tuple* d_reflectionsBuffer, MeshDescriptor* d_meshDescriptorBuffer, MeshSegment* d_meshSegmentBuffer);
  void renderImage(int blockDimX, int blockDimY, const char* filename, int imageWidth, int imageHeight, MeshDescriptor* d_meshDescriptorBuffer, MeshSegment* d_meshSegmentBuffer);
}

RaytraceImage::RaytraceImage(ModelHandler* modelHandler) {
  frameWidth = 250;
  frameHeight = 250;
  
  imageWidth = 5000;
  imageHeight = 5000;

  shouldTakePhoto = false;
  
  std::vector<MeshDescriptor> h_meshDescriptorList = modelHandler->getCollectiveMeshDescriptorList();
  std::vector<MeshSegment> h_meshSegmentList = modelHandler->getCollectiveMeshSegmentList();

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

void RaytraceImage::update(Camera* camera) {
  Tuple cameraPosition = camera->getPosition();
  updateCudaCamera(cameraPosition.x, cameraPosition.y, cameraPosition.z, camera->getPitch(), camera->getYaw());
  updateScene();
}

void RaytraceImage::render() {
  renderFrame(16, 16, d_colorBuffer, &cudaTextureResource, frameWidth, frameHeight, d_lightingBuffer, d_reflectionsBuffer, d_meshDescriptorBuffer, d_meshSegmentBuffer);
}