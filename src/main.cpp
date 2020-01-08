#include "engine.h"

int main(int argn, char** argv) {
  Engine* engine = new Engine();
  engine->run();
  
  delete engine;
  return 0;
}