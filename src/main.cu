#include <fstream>
#include <stdio.h>
#include "shape.h"
#include "ray.h"

#define IMAGE_WIDTH 1000
#define IMAGE_HEIGHT 1000

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

	Sphere sphere_A = {{0.0, 0.0, 5.0, 1.0}};
	Sphere sphere_B = {{1.0, 1.0, 4.0, 1.0}};
	float intersection = intersectSphere(sphere_A, ray) + intersectSphere(sphere_B, ray);

	float color = intersection * 255.0;
	colorData[(idy*IMAGE_WIDTH)+idx] = {color, color, color};
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
	Tuple* h_colorData = (Tuple*)malloc(IMAGE_WIDTH*IMAGE_HEIGHT*sizeof(Tuple));
	Tuple* d_colorData;
	cudaMalloc((Tuple**)&d_colorData, IMAGE_WIDTH*IMAGE_HEIGHT*sizeof(Tuple));

	dim3 block(32, 32);
	dim3 grid((IMAGE_WIDTH + block.x - 1) / block.x, (IMAGE_HEIGHT + block.y - 1) / block.y);
	printf("rendering ray traced image...\n");
	colorFromRay<<<grid, block>>>(d_colorData);
	cudaDeviceSynchronize();
	printf("finished rendering\n");

	cudaMemcpy(h_colorData, d_colorData, IMAGE_WIDTH*IMAGE_HEIGHT*sizeof(Tuple), cudaMemcpyDeviceToHost);
	cudaFree(d_colorData);

	const char* filename = "image.ppm";
	writeColorDataToFile(filename, h_colorData);
	printf("saved image as: [%s]\n", filename);

	cudaDeviceReset();
	free(h_colorData);
	printf("\n");
	return 0;
}