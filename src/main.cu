#include <fstream>
#include <stdio.h>
#include "ray.h"

#define IMAGE_WIDTH 100
#define IMAGE_HEIGHT 100

__global__
void colorFromRay(Tuple* colorData) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (idx < IMAGE_WIDTH && idy < IMAGE_HEIGHT) {
		colorData[(idy*IMAGE_WIDTH)+idx] = {255, 255, 255};
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

	free(h_colorData);
	return 0;
}