#include <fstream>

int main(void) {
	std::ofstream file;
	file.open("image.ppm");
	file << "P3\n" << 100 << " " << 100 << "\n255\n";

	return 0;
}