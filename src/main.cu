extern "C" void renderImage(int blockDimX, int blockDimY, const char* filename);

int main(int argn, char** argv) {
  renderImage(atoi(argv[2]), atoi(argv[3]), argv[1]);
  return 0;
}