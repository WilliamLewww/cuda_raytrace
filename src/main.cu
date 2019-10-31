#include <fstream>
#include <stdio.h>
#include "analysis.h"
#include "precomputed.h"
#include "shape.h"
#include "ray.h"

#define IMAGE_WIDTH 1000
#define IMAGE_HEIGHT 1000
#define SPHERE_COUNT 1

__constant__ Sphere sphereArray[1];

__device__
Tuple normalAtSphere(Sphere sphere, Tuple point) {
	return point - sphere.origin;
}

__device__
Precomputed prepareComputations(float intersectionPoint, Sphere sphere, Ray ray) {
	Precomputed precomputed;

	precomputed.intersectionPoint = intersectionPoint;

	precomputed.point = project(ray, precomputed.intersectionPoint);
	precomputed.eyeV = negate(ray.direction);
	precomputed.normalV = normalAtSphere(sphere, precomputed.point);

	bool isNegative = dot(precomputed.normalV, precomputed.eyeV) < 0;
	precomputed.inside = isNegative;
	precomputed.normalV = (precomputed.normalV * !isNegative) + (negate(precomputed.normalV) * isNegative);

	precomputed.overPoint = precomputed.point + precomputed.normalV * 0.01;

	return precomputed;
}

__device__
int intersectSphere(float* intersectionPoint, Sphere sphere, Ray ray) {
	Tuple sphereToRay = ray.origin - sphere.origin;
	float a = dot(ray.direction, ray.direction);
	float b = 2.0 * dot(sphereToRay, ray.direction);
	float c = dot(sphereToRay, sphereToRay) - 1;

	float discriminant = (b * b) - (4 * a * c);
	float pointA = (-b - sqrt(discriminant)) / (2 * a);
	float pointB = (-b + sqrt(discriminant)) / (2 * a);

	*intersectionPoint = (pointA * (pointA <= pointB)) + (pointB * (pointB < pointA));

	return (discriminant >= 0) * (2 - (pointA == pointB));
}

__global__
void colorFromRay(Tuple* colorOut) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (idx >= IMAGE_WIDTH || idy >= IMAGE_HEIGHT) { return; }

	Tuple origin = {0.0, 0.0, 0.0, 1.0};
	Tuple pixel = {float(idx) - (IMAGE_WIDTH / 2), float(idy) - (IMAGE_HEIGHT / 2), 100.0, 1.0};
	Tuple direction = normalize(pixel - origin);

	Ray ray = {origin, direction};

	float intersectionPoint = 0;
	int intersectionCount = intersectSphere(&intersectionPoint, sphereArray[0], ray);

	Precomputed precomputed = prepareComputations(intersectionPoint * (intersectionCount > 0), sphereArray[0], ray);

	if (intersectionCount > 0) {
		colorOut[(idy*IMAGE_WIDTH)+idx] = {255, 255, 255};
	}
	else {
		colorOut[(idy*IMAGE_WIDTH)+idx] = {0, 0, 0};
	}
}

void writeColorDataToFile(const char* filename, Tuple* colorData) {
	std::ofstream file;
	file.open(filename);
	file << "P3\n" << IMAGE_WIDTH << " " << IMAGE_HEIGHT << "\n255\n";

	for (int x = 0; x < IMAGE_WIDTH * IMAGE_HEIGHT; x++) {
		file << colorData[x].x << " ";
		file << colorData[x].y << " ";
		file << colorData[x].z << "\n";

	}

	file.close();
}

int main(void) {
	printf("\n");

	Analysis::setAbsoluteStart();
	Analysis::createLabel(0, "allocate_memory");
	Analysis::createLabel(1, "execute_kernel");
	Analysis::createLabel(2, "copy_device");
	Analysis::createLabel(3, "create_image");

	Analysis::begin();
	const Sphere h_sphereArray[] = {{{0.0, 0.0, 5.0, 1.0}}};
	cudaMemcpyToSymbol(sphereArray, h_sphereArray, SPHERE_COUNT*sizeof(Sphere));

	Tuple* h_colorData = (Tuple*)malloc(IMAGE_WIDTH*IMAGE_HEIGHT*sizeof(Tuple));
	Tuple* d_colorData;
	cudaMalloc((Tuple**)&d_colorData, IMAGE_WIDTH*IMAGE_HEIGHT*sizeof(Tuple));
	Analysis::end(0);

	dim3 block(32, 32);
	dim3 grid((IMAGE_WIDTH + block.x - 1) / block.x, (IMAGE_HEIGHT + block.y - 1) / block.y);

	Analysis::begin();
	printf("rendering ray traced image...\n");
	colorFromRay<<<grid, block>>>(d_colorData);
	cudaDeviceSynchronize();
	printf("finished rendering\n");
	Analysis::end(1);

	Analysis::begin();
	cudaMemcpy(h_colorData, d_colorData, IMAGE_WIDTH*IMAGE_HEIGHT*sizeof(Tuple), cudaMemcpyDeviceToHost);
	cudaFree(d_colorData);
	Analysis::end(2);

	Analysis::begin();
	const char* filename = "image.ppm";
	writeColorDataToFile(filename, h_colorData);
	printf("saved image as: [%s]\n", filename);
	Analysis::end(3);

	Analysis::printAll(IMAGE_WIDTH, IMAGE_HEIGHT);

	cudaDeviceReset();
	free(h_colorData);
	printf("\n");
	return 0;
}