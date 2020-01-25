#pragma once
#include <vector>

#include <cuda_runtime.h>

#include "model.h"
#include "raster_model.h"
#include "phong_raster_model.h"

class ModelHandler {
public:
  static Model* createModel(const char* filename, int reflective);
  static Model* createModel(Model* model);
  static RasterModel* createRasterModel(RasterModelType rasterModelType, GLuint* shaderProgramHandle, Model* model);
  static Model* createReducedModel(Model* model);
  static void createReducedOBJ(const char* source, const char* target);
};