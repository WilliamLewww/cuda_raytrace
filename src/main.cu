#include <fstream>
#include <stdio.h>
#include "shape.h"
#include "ray.h"

#define IMAGE_WIDTH 100
#define IMAGE_HEIGHT 100

__device__
bool intersectSphere(Sphere sphere, Ray ray) {
	Tuple sphereToRay = ray.origin - sphere.origin;
	float a = dot(ray.direction, ray.direction);
	float b = 2.0 * dot(sphereToRay, ray.direction);
	float c = dot(sphereToRay, sphereToRay) - 1;

	float discriminant = (b * b) - (4 * a * c);

	return discriminant > 0;
}

__global__
void colorFromRay(Tuple* colorData) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (idx >= IMAGE_WIDTH || idy >= IMAGE_HEIGHT) { return; }

	Tuple origin = {0.0, 0.0, 0.0, 1.0};
	Tuple pixel = {float(idx) - (IMAGE_WIDTH / 2), float(idy) - (IMAGE_HEIGHT / 2), 100.0, 0.0};
	Tuple direction = normalize(pixel - origin);
	Ray ray = {origin, direction};

	Sphere sphere = {{0.0, 0.0, 5.0, 1.0}};

	if (intersectSphere(sphere, ray)) {
		colorData[(idy*IMAGE_WIDTH)+idx] = {255, 255, 255};
	}
	else {
		colorData[(idy*IMAGE_WIDTH)+idx] = {0, 0, 255};
	}
}

void writeColorDataToFile(Tuple* colorData) {
	std::ofstream file;
	file.open("image.ppm");
	file << "P3\n" << IMAGE_WIDTH << " " << IMAGE_HEIGHT << "\n255\n";

	for (int x = 0; x < IMAGE_WIDTH * IMAGE_HEIGHT; x++) {
		file << colorData[x].x << " ";
		file << colorData[x].y << " ";
		file << colorData[x].z << "\n";

	}

	file.close();
}

int main(void) {
	Tuple* h_colorData = (Tuple*)malloc(IMAGE_WIDTH*IMAGE_HEIGHT*sizeof(Tuple));
	Tuple* d_colorData;
	cudaMalloc((Tuple**)&d_colorData, IMAGE_WIDTH*IMAGE_HEIGHT*sizeof(Tuple));

	dim3 block(32, 32);
	dim3 grid((IMAGE_WIDTH + block.x - 1) / block.x, (IMAGE_HEIGHT + block.y - 1) / block.y);
	colorFromRay<<<grid, block>>>(d_colorData);
	cudaDeviceSynchronize();

	cudaMemcpy(h_colorData, d_colorData, IMAGE_WIDTH*IMAGE_HEIGHT*sizeof(Tuple), cudaMemcpyDeviceToHost);
	cudaFree(d_colorData);

	writeColorDataToFile(h_colorData);

	cudaDeviceReset();
	free(h_colorData);
	return 0;
}