#pragma once

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
	Tuple normal;

	Tuple color;

	float modelMatrix[16];
	float inverseModelMatrix[16];
};

struct Camera {
	Tuple position;
	Tuple direction;
};

float* inverseMatrix(float* matrix) {
    float* inverse = (float*)malloc(16*sizeof(float));

    inverse[0] = matrix[5]  * matrix[10] * matrix[15] - matrix[5]  * matrix[11] * matrix[14] - matrix[9]  * matrix[6]  * matrix[15] + matrix[9]  * matrix[7]  * matrix[14] +matrix[13] * matrix[6]  * matrix[11] - matrix[13] * matrix[7]  * matrix[10];
    inverse[4] = -matrix[4]  * matrix[10] * matrix[15] + matrix[4]  * matrix[11] * matrix[14] + matrix[8]  * matrix[6]  * matrix[15] - matrix[8]  * matrix[7]  * matrix[14] - matrix[12] * matrix[6]  * matrix[11] + matrix[12] * matrix[7]  * matrix[10];
    inverse[8] = matrix[4]  * matrix[9] * matrix[15] - matrix[4]  * matrix[11] * matrix[13] - matrix[8]  * matrix[5] * matrix[15] + matrix[8]  * matrix[7] * matrix[13] + matrix[12] * matrix[5] * matrix[11] - matrix[12] * matrix[7] * matrix[9];
    inverse[12] = -matrix[4]  * matrix[9] * matrix[14] + matrix[4]  * matrix[10] * matrix[13] +matrix[8]  * matrix[5] * matrix[14] - matrix[8]  * matrix[6] * matrix[13] - matrix[12] * matrix[5] * matrix[10] + matrix[12] * matrix[6] * matrix[9];
    inverse[1] = -matrix[1]  * matrix[10] * matrix[15] + matrix[1]  * matrix[11] * matrix[14] + matrix[9]  * matrix[2] * matrix[15] - matrix[9]  * matrix[3] * matrix[14] - matrix[13] * matrix[2] * matrix[11] + matrix[13] * matrix[3] * matrix[10];
    inverse[5] = matrix[0]  * matrix[10] * matrix[15] - matrix[0]  * matrix[11] * matrix[14] - matrix[8]  * matrix[2] * matrix[15] + matrix[8]  * matrix[3] * matrix[14] + matrix[12] * matrix[2] * matrix[11] - matrix[12] * matrix[3] * matrix[10];
    inverse[9] = -matrix[0]  * matrix[9] * matrix[15] + matrix[0]  * matrix[11] * matrix[13] + matrix[8]  * matrix[1] * matrix[15] - matrix[8]  * matrix[3] * matrix[13] - matrix[12] * matrix[1] * matrix[11] + matrix[12] * matrix[3] * matrix[9];
    inverse[13] = matrix[0]  * matrix[9] * matrix[14] - matrix[0]  * matrix[10] * matrix[13] - matrix[8]  * matrix[1] * matrix[14] + matrix[8]  * matrix[2] * matrix[13] + matrix[12] * matrix[1] * matrix[10] - matrix[12] * matrix[2] * matrix[9];
    inverse[2] = matrix[1]  * matrix[6] * matrix[15] - matrix[1]  * matrix[7] * matrix[14] - matrix[5]  * matrix[2] * matrix[15] + matrix[5]  * matrix[3] * matrix[14] + matrix[13] * matrix[2] * matrix[7] - matrix[13] * matrix[3] * matrix[6];
    inverse[6] = -matrix[0]  * matrix[6] * matrix[15] + matrix[0]  * matrix[7] * matrix[14] + matrix[4]  * matrix[2] * matrix[15] - matrix[4]  * matrix[3] * matrix[14] - matrix[12] * matrix[2] * matrix[7] + matrix[12] * matrix[3] * matrix[6];
    inverse[10] = matrix[0]  * matrix[5] * matrix[15] - matrix[0]  * matrix[7] * matrix[13] - matrix[4]  * matrix[1] * matrix[15] + matrix[4]  * matrix[3] * matrix[13] + matrix[12] * matrix[1] * matrix[7] - matrix[12] * matrix[3] * matrix[5];
    inverse[14] = -matrix[0]  * matrix[5] * matrix[14] + matrix[0]  * matrix[6] * matrix[13] + matrix[4]  * matrix[1] * matrix[14] - matrix[4]  * matrix[2] * matrix[13] - matrix[12] * matrix[1] * matrix[6] + matrix[12] * matrix[2] * matrix[5];
    inverse[3] = -matrix[1] * matrix[6] * matrix[11] + matrix[1] * matrix[7] * matrix[10] + matrix[5] * matrix[2] * matrix[11] - matrix[5] * matrix[3] * matrix[10] - matrix[9] * matrix[2] * matrix[7] + matrix[9] * matrix[3] * matrix[6];
    inverse[7] = matrix[0] * matrix[6] * matrix[11] - matrix[0] * matrix[7] * matrix[10] - matrix[4] * matrix[2] * matrix[11] + matrix[4] * matrix[3] * matrix[10] + matrix[8] * matrix[2] * matrix[7] - matrix[8] * matrix[3] * matrix[6];
    inverse[11] = -matrix[0] * matrix[5] * matrix[11] + matrix[0] * matrix[7] * matrix[9] + matrix[4] * matrix[1] * matrix[11] - matrix[4] * matrix[3] * matrix[9] - matrix[8] * matrix[1] * matrix[7] + matrix[8] * matrix[3] * matrix[5];
    inverse[15] = matrix[0] * matrix[5] * matrix[10] - matrix[0] * matrix[6] * matrix[9] - matrix[4] * matrix[1] * matrix[10] + matrix[4] * matrix[2] * matrix[9] + matrix[8] * matrix[1] * matrix[6] - matrix[8] * matrix[2] * matrix[5];

    float determinant = 1.0 / (matrix[0] * inverse[0] + matrix[1] * inverse[4] + matrix[2] * inverse[8] + matrix[3] * inverse[12]);
    for (int x = 0; x < 16; x++) {
        inverse[x] *= determinant;
    }

    return inverse;
}

float* createIdentityMatrix() {
	float* matrix = (float*)malloc(16*sizeof(float));
	matrix[0] = 1.0;  matrix[1] = 0.0;  matrix[2] = 0.0;  matrix[3] = 0.0;
	matrix[4] = 0.0;  matrix[5] = 1.0;  matrix[6] = 0.0;  matrix[7] = 0.0;
	matrix[8] = 0.0;  matrix[9] = 0.0;  matrix[10] = 1.0; matrix[11] = 0.0;
	matrix[12] = 0.0; matrix[13] = 0.0; matrix[14] = 0.0; matrix[15] = 1.0;

	return matrix;
}

float* createTranslateMatrix(float x, float y, float z) {
	float* matrix = createIdentityMatrix();
	matrix[3] = x;
	matrix[7] = y;
	matrix[11] = z;

	return matrix;
}

float* createScaleMatrix(float x, float y, float z) {
	float* matrix = createIdentityMatrix();
	matrix[0] = x;
	matrix[5] = y;
	matrix[10] = z;

	return matrix;
}

void initializeModelMatrix(Sphere* sphere, float* matrix) {
	float* modelMatrix = sphere->modelMatrix;
	for (int x = 0; x < 16; x++) { modelMatrix[x] = matrix[x]; }

	modelMatrix = sphere->inverseModelMatrix;
	float* inverseModelMatrix = inverseMatrix(sphere->modelMatrix);
	for (int x = 0; x < 16; x++) { modelMatrix[x] = inverseModelMatrix[x]; }
}

void initializeModelMatrix(Plane* plane, float* matrix) {
	float* modelMatrix = plane->modelMatrix;
	for (int x = 0; x < 16; x++) { modelMatrix[x] = matrix[x]; }

	modelMatrix = plane->inverseModelMatrix;
	float* inverseModelMatrix = inverseMatrix(plane->modelMatrix);
	for (int x = 0; x < 16; x++) { modelMatrix[x] = inverseModelMatrix[x]; }
}

__device__ Tuple multiply(float* matrix, Tuple tuple) {
	return { 
			(matrix[0] * tuple.x) + (matrix[1] * tuple.y) + (matrix[2] * tuple.z) + (matrix[3] * tuple.w),
			(matrix[4] * tuple.x) + (matrix[5] * tuple.y) + (matrix[6] * tuple.z) + (matrix[7] * tuple.w),
			(matrix[8] * tuple.x) + (matrix[9] * tuple.y) + (matrix[10] * tuple.z) + (matrix[11] * tuple.w),
			(matrix[12] * tuple.x) + (matrix[13] * tuple.y) + (matrix[14] * tuple.z) + (matrix[15] * tuple.w)
		};
}

__device__ Ray transform(Ray ray, float* matrix) { return {multiply(matrix, ray.origin), multiply(matrix, ray.direction)}; }
__device__ Tuple operator+(Tuple tupleA, Tuple tupleB) { return {tupleA.x + tupleB.x, tupleA.y + tupleB.y, tupleA.z + tupleB.z, tupleA.w + tupleB.w}; }
__device__ Tuple operator-(Tuple tupleA, Tuple tupleB) { return {tupleA.x - tupleB.x, tupleA.y - tupleB.y, tupleA.z - tupleB.z, tupleA.w - tupleB.w}; }
__device__ Tuple operator*(Tuple tuple, float scalar) { return {tuple.x * scalar, tuple.y * scalar, tuple.z * scalar, tuple.w * scalar}; }
__device__ Tuple operator*(float scalar, Tuple tuple) { return {tuple.x * scalar, tuple.y * scalar, tuple.z * scalar, tuple.w * scalar}; }
__device__ Tuple hadamardProduct(Tuple tupleA, Tuple tupleB) { return {tupleA.x * tupleB.x, tupleA.y * tupleB.y, tupleA.z * tupleB.z, tupleA.w * tupleB.w}; }
__device__ float magnitude(Tuple tuple) { return sqrt(tuple.x * tuple.x + tuple.y * tuple.y + tuple.z * tuple.z + tuple.w * tuple.w); }
__device__ Tuple normalize(Tuple tuple) { return {tuple.x / magnitude(tuple), tuple.y / magnitude(tuple), tuple.z / magnitude(tuple), tuple.w / magnitude(tuple)}; }
__device__ Tuple negate(Tuple tuple) { return {-tuple.x, -tuple.y, -tuple.z, -tuple.w}; }
__device__ Tuple project(Ray ray, float t) { return ray.origin + (ray.direction * t); }
__device__ float dot(Tuple tupleA, Tuple tupleB) { return (tupleA.x * tupleB.x) + (tupleA.y * tupleB.y) + (tupleA.z * tupleB.z) + (tupleA.w * tupleB.w); }