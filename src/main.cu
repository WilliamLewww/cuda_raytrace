#include <fstream>
#include <stdio.h>
#include "structures.h"

#define IMAGE_WIDTH 1000
#define IMAGE_HEIGHT 1000

__global__
void colorFromRay(Tuple* colorOut) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (idx >= IMAGE_WIDTH || idy >= IMAGE_HEIGHT) { return; }
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
	return 0;
}