#include "engine.h"

int main(int argn, char** argv) {
  Engine engine;
  engine.initialize();
  engine.run();
  engine.exit();
  
  return 0;
}