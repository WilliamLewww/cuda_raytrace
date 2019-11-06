#include <fstream>
#include <stdio.h>
#include "analysis.h"
#include "structures.h"

#define IMAGE_WIDTH 500
#define IMAGE_HEIGHT 500
#define SPHERE_COUNT 1
#define LIGHT_COUNT 1

__constant__ Sphere sphereArray[1];
__constant__ Light lightArray[1];

__device__
int intersectSphere(float* intersectionPoint, Sphere sphere, Ray ray) {
	Tuple sphereToRay = ray.origin - sphere.origin;
	float a = dot(ray.direction, ray.direction);
	float b = 2.0f * dot(sphereToRay, ray.direction);
	float c = dot(sphereToRay, sphereToRay) - 1.0f;

	float discriminant = (b * b) - (4.0f * a * c);
	float pointA = (-b - sqrt(discriminant)) / (2.0f * a);
	float pointB = (-b + sqrt(discriminant)) / (2.0f * a);

	*intersectionPoint = (pointB * (pointA <= pointB)) + (pointA * (pointB < pointA));

	return (discriminant >= 0) * (2 - (pointA == pointB));
}

__global__
void colorFromRay(Tuple* colorOut) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (idx >= IMAGE_WIDTH || idy >= IMAGE_HEIGHT) { return; }

	Tuple origin = {0.0f, 0.0f, 0.0f, 1.0f};
	Tuple pixel = {float(idx) - (IMAGE_WIDTH / 2.0f), float(idy) - (IMAGE_HEIGHT / 2.0f), 100.0f, 1.0f};
	Tuple direction = normalize(pixel - origin);

	Ray ray = {origin, direction};

	float intersectionPoint = 0.0f;
	int intersectionCount = intersectSphere(&intersectionPoint, sphereArray[0], ray);

	if (intersectionCount > 0) {
		Tuple direction = normalize(lightArray[0].position - sphereArray[0].origin);
		Tuple normal = normalize(sphereArray[0].origin - project(ray, intersectionPoint));
		float angleDifference = dot(normal, direction);
		float color = ((angleDifference > 0) * angleDifference) * 255.0f;

		colorOut[(idy*IMAGE_WIDTH)+idx] = {color, 0.0f, 0.0f, 1.0f};
	}
	else {
		colorOut[(idy*IMAGE_WIDTH)+idx] = {0.0f, 0.0f, 0.0f, 1.0f};
	}
}

void writeColorDataToFile(const char* filename, Tuple* colorData) {
	std::ofstream file;
	file.open(filename);
	file << "P3\n" << IMAGE_WIDTH << " " << IMAGE_HEIGHT << "\n255\n";

	for (int x = 0; x < IMAGE_WIDTH * IMAGE_HEIGHT; x++) {
		file << int(colorData[x].x) << " ";
		file << int(colorData[x].y) << " ";
		file << int(colorData[x].z) << "\n";

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
	const Sphere h_sphereArray[] = {{{0.0, 0.0, 3.0, 1.0}}};
	cudaMemcpyToSymbol(sphereArray, h_sphereArray, SPHERE_COUNT*sizeof(Sphere));

	const Light h_lightArray[] = {{{10, 10, -3, 1}, {1, 1, 1, 1}}};
	cudaMemcpyToSymbol(lightArray, h_lightArray, LIGHT_COUNT*sizeof(Light));

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