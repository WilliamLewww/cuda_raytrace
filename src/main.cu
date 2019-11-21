#include <fstream>
#include <stdio.h>
#include "structures.h"

#define IMAGE_WIDTH 1000
#define IMAGE_HEIGHT 1000

#define LIGHT_COUNT 1
#define SPHERE_COUNT 2

__constant__ Light lightArray[LIGHT_COUNT];
__constant__ Sphere sphereArray[SPHERE_COUNT];

__device__
int intersectSphere(float* intersectionPoint, Sphere sphere, Ray ray) {
	Tuple sphereToRay = ray.origin - sphere.origin;
	float a = dot(ray.direction, ray.direction);
	float b = 2.0f * dot(sphereToRay, ray.direction);
	float c = dot(sphereToRay, sphereToRay) - (sphere.radius * sphere.radius);

	float discriminant = (b * b) - (4.0f * a * c);
	float pointA = (-b - sqrt(discriminant)) / (2.0f * a);
	float pointB = (-b + sqrt(discriminant)) / (2.0f * a);

	*intersectionPoint = (pointA * (pointA <= pointB)) + (pointB * (pointB < pointA));

	return (discriminant >= 0) * (2 - (pointA == pointB)) * (pointA > 0 && pointB > 0);
}

__global__
void colorFromRay(Tuple* colorOut) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (idx >= IMAGE_WIDTH || idy >= IMAGE_HEIGHT) { return; }

	Tuple origin = {0.0f, 0.0f, 0.0f, 1.0f};
	Tuple pixel = {idx - (IMAGE_WIDTH / 2.0f), idy - (IMAGE_HEIGHT / 2.0f), 200.0f, 1.0f};
	Tuple direction = normalize(pixel - origin);

	Ray ray = {origin, direction};

	int intersectionIndex = -1;
	float intersectionPoint = 0.0f;

	#pragma unroll
	for (int x = 0; x < SPHERE_COUNT; x++) {
		float point;
		int count = intersectSphere(&point, sphereArray[x], ray);

		intersectionIndex = (x * (count > 0 && (point < intersectionPoint || intersectionPoint == 0))) + (intersectionIndex * (count <= 0 || (point >= intersectionPoint && intersectionPoint != 0)));
		intersectionPoint = (point * (count > 0 && (point < intersectionPoint || intersectionPoint == 0))) + (intersectionPoint * (count <= 0 || (point >= intersectionPoint && intersectionPoint != 0)));
	}

	if (intersectionIndex != -1) {
		Tuple intersectionPosition = project(ray, intersectionPoint);
		Ray lightRay = {intersectionPosition, normalize(lightArray[0].position - intersectionPosition)};

		int intersecionCount = 0;

		#pragma unroll
		for (int x = 0; x < SPHERE_COUNT; x++) {
			if (x != intersectionIndex) {
				float point;
				intersecionCount += intersectSphere(&point, sphereArray[x], lightRay);
			}
		}

		Tuple direction = normalize(lightArray[0].position - project(ray, intersectionPoint));
		Tuple normal = negate(normalize(sphereArray[intersectionIndex].origin - project(ray, intersectionPoint)));
		float angleDifference = dot(normal, direction);
		Tuple color = (0.1f * sphereArray[intersectionIndex].color) + ((angleDifference > 0) * angleDifference) * sphereArray[intersectionIndex].color * (intersecionCount == 0);

		colorOut[(idy*IMAGE_WIDTH)+idx] = color;
	}
	else {
		colorOut[(idy*IMAGE_WIDTH)+idx] = {0.0, 0.0, 0.0, 1.0};
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

int main(int argn, char** argv) {
	printf("\n");

	Tuple* h_colorData = (Tuple*)malloc(IMAGE_WIDTH*IMAGE_HEIGHT*sizeof(Tuple));
	Tuple* d_colorData;
	cudaMalloc((Tuple**)&d_colorData, IMAGE_WIDTH*IMAGE_HEIGHT*sizeof(Tuple));

	const Light h_lightArray[] = {{{-10.0, -10.0, 0.0, 1.0}, {1.0, 1.0, 1.0, 1.0}}};
	cudaMemcpyToSymbol(lightArray, h_lightArray, LIGHT_COUNT*sizeof(Light));

	const Sphere h_sphereArray[] = {
									{{0.0, 0.0, 3.0, 1.0}, 2.0, {255.0, 0.0, 0.0}},
									{{5.0, 5.0, 5.0, 1.0}, 4.0, {0.0, 255.0, 0.0}}
								};
	cudaMemcpyToSymbol(sphereArray, h_sphereArray, SPHERE_COUNT*sizeof(Sphere));

	dim3 block(32, 32);
	dim3 grid((IMAGE_WIDTH + block.x - 1) / block.x, (IMAGE_HEIGHT + block.y - 1) / block.y);

	colorFromRay<<<grid, block>>>(d_colorData);
	cudaDeviceSynchronize();

	cudaMemcpy(h_colorData, d_colorData, IMAGE_WIDTH*IMAGE_HEIGHT*sizeof(Tuple), cudaMemcpyDeviceToHost);
	cudaFree(d_colorData);

	const char* filename = argv[1];
	writeColorDataToFile(filename, h_colorData);
	printf("saved image as: [%s]\n", filename);

	cudaDeviceReset();
	free(h_colorData);
	printf("\n");
	return 0;
}