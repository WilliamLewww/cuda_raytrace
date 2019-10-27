#include <fstream>
#define IMAGE_WIDTH 100
#define IMAGE_HEIGHT 100

int main(void) {
	std::ofstream file;
	file.open("image.ppm");
	file << "P3\n" << IMAGE_WIDTH << " " << IMAGE_HEIGHT << "\n255\n";

	file.close();

	return 0;
}