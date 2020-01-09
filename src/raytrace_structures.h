#pragma once
#include "math.h"

struct Tuple {
  float x;
  float y;
  float z;
  float w;
};

struct Ray {
  Tuple origin;
  Tuple direction;
};

struct Light {
  Tuple position;
  Tuple intensity;
};

struct Sphere {
  Tuple origin;
  float radius;

  Tuple color;

  float modelMatrix[16];
  float inverseModelMatrix[16];
};

struct Plane {
  Tuple origin;
  
  Tuple color;

  float modelMatrix[16];
  float inverseModelMatrix[16];
};

struct Triangle {
  Tuple vertexA;
  Tuple vertexB;
  Tuple vertexC;
  Tuple normal;

  Tuple color;

  float modelMatrix[16];
  float inverseModelMatrix[16];
};

struct MeshDescriptor {
  int segmentCount;
  int reflective;

  float modelMatrix[16];
  float inverseModelMatrix[16];
};

struct MeshSegment {
  Tuple vertexA;
  Tuple vertexB;
  Tuple vertexC;
  Tuple normal;

  Tuple color;
};

struct Camera {
  Tuple position;
  Tuple direction;

  float modelMatrix[16];
  float inverseModelMatrix[16];
};

float* inverseMatrix(float* matrix);
float* multiply(float* a, float* b);
float* createIdentityMatrix();
float* createTranslateMatrix(float x, float y, float z);
float* createScaleMatrix(float x, float y, float z);
float* createRotationMatrixX(float radians);
float* createRotationMatrixY(float radians);
float* createRotationMatrixZ(float radians);
void initializeModelMatrix(float* dst, float* src);
void initializeInverseModelMatrix(float* dst, float* src);
void initializeModelMatrix(Sphere* sphere, float* matrix);
void initializeModelMatrix(Plane* plane, float* matrix);
void initializeModelMatrix(Triangle* triangle, float* matrix);
Tuple multiplyMatrixTuple(float* matrix, Tuple tuple);