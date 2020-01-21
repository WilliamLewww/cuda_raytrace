#pragma once
#include <vector>
#include <math.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "raytrace_structures.h"
#include "camera.h"
#include "model_handler.h"
#include "model_container.h"
#include "model.h"
#include "input.h"

class RaytraceImage {
private:
  ModelContainer* modelContainer;

  int frameWidth, frameHeight;
  int imageWidth, imageHeight;

  struct cudaGraphicsResource* cudaTextureResource;

  Tuple* d_lightingBuffer;
  Tuple* d_reflectionsBuffer;
  void* d_colorBuffer;
public:
  RaytraceImage(ModelContainer* modelContainer);
  ~RaytraceImage();

  void update(Camera* camera);
  void render();

  void updateResolution(int width, int height, GLuint textureResource);
};