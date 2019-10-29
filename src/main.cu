#include <fstream>
#include "ray.h"

#define IMAGE_WIDTH 100
#define IMAGE_HEIGHT 100

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
	return 0;
}