#include <fstream>
#include <stdio.h>
#include "analysis.h"
#include "structures.h"

#define IMAGE_WIDTH 1000
#define IMAGE_HEIGHT 1000
#define FOV 1.0471975512

#define SHADE_OFFSET 0.01
#define PLANE_COMPARISON 0.00001

#define LIGHT_COUNT 1

#define SPHERE_COUNT 2
#define PLANE_COUNT 1

__constant__ Camera camera[1];

__constant__ Light lightArray[LIGHT_COUNT];

__constant__ Sphere sphereArray[SPHERE_COUNT];
__constant__ Plane planeArray[PLANE_COUNT];

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

__device__
int intersectPlane(float* intersectionPoint, Plane plane, Ray ray) {
	int intersectionCount = 0;

	float denom = dot(plane.normal, ray.direction);
	if (fabsf(denom) > PLANE_COMPARISON) {
		float t = dot(plane.origin - ray.origin, plane.normal);

		if (t >= 0) {
			intersectionCount = 1;
			*intersectionPoint = t;
		}
	}

	return intersectionCount;
}

__global__
void colorFromRay(Tuple* colorOut) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (idx >= IMAGE_WIDTH || idy >= IMAGE_HEIGHT) { return; }

	Tuple origin = {0.0f, 0.0f, 0.0f, 1.0f};
	Tuple pixel = {(float(idx) * camera[0].pixelSize) - (camera[0].pixelSize * IMAGE_WIDTH / 2.0f), 
				   (float(idy) * camera[0].pixelSize) - (camera[0].pixelSize * IMAGE_HEIGHT / 2.0f), 100.0f, 1.0f};
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

		float lightIntersectionPoint = 0.0f;

		#pragma unroll
		for (int x = 0; x < SPHERE_COUNT; x++) {
			float point;
			int count = intersectSphere(&point, sphereArray[x], lightRay);

			lightIntersectionPoint = (point * (count > 0 && (point < lightIntersectionPoint || lightIntersectionPoint == 0))) + (lightIntersectionPoint * (count <= 0 || (point >= lightIntersectionPoint && lightIntersectionPoint != 0)));
		}

		Tuple projectedPosition = project(lightRay, lightIntersectionPoint);
		int inShade = (magnitude(projectedPosition) + SHADE_OFFSET) < magnitude(intersectionPosition);

		Tuple direction = normalize(lightArray[0].position - project(ray, intersectionPoint));
		Tuple normal = normalize(sphereArray[intersectionIndex].origin - project(ray, intersectionPoint));
		float angleDifference = dot(normal, direction);
		float color = (0.1f * 255.0f) + ((angleDifference > 0) * angleDifference) * 255.0f * inShade;

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

int main(int argn, char** argv) {
	printf("\n");

	Analysis::setAbsoluteStart();
	Analysis::createLabel(0, "allocate_memory");
	Analysis::createLabel(1, "execute_kernel");
	Analysis::createLabel(2, "copy_device");
	Analysis::createLabel(3, "create_image");

	Analysis::begin();
	float halfView = tan(FOV / 2);
	float aspect = float(IMAGE_WIDTH) / float(IMAGE_HEIGHT);

	float halfWidth = ((aspect >= 1) * halfView) + ((aspect < 1) * halfView * aspect);
	float halfHeight = ((aspect >= 1) * halfView / aspect) + ((aspect < 1) * halfView);
	float pixelSize = ((halfWidth * 2) / IMAGE_WIDTH) * 500;

	const Camera h_camera[] = {{halfWidth, halfHeight, pixelSize}};
	cudaMemcpyToSymbol(camera, h_camera, sizeof(Camera));

	const Light h_lightArray[] = {{{-10, -10, 5, 1}, {1, 1, 1, 1}}};
	cudaMemcpyToSymbol(lightArray, h_lightArray, LIGHT_COUNT*sizeof(Light));

	const Sphere h_sphereArray[] = {{{0.0, 0.0, 5.0, 1.0}},{{2.0, 2.0, 5.0, 1.0}}};
	cudaMemcpyToSymbol(sphereArray, h_sphereArray, SPHERE_COUNT*sizeof(Sphere));

	const Plane h_planeArray[] = {{{0, 0, 5},{0, 1, 1}}};
	cudaMemcpyToSymbol(planeArray, h_planeArray, PLANE_COUNT*sizeof(Plane));

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
	const char* filename = argv[1];
	writeColorDataToFile(filename, h_colorData);
	printf("saved image as: [%s]\n", filename);
	Analysis::end(3);

	Analysis::printAll(IMAGE_WIDTH, IMAGE_HEIGHT);

	cudaDeviceReset();
	free(h_colorData);
	printf("\n");
	return 0;
}