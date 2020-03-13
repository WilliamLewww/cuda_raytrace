#include "engine.h"

int main(int argn, const char** argv) {
  Engine* engine = new Engine();
  engine->run();

  delete engine;
  return 0;
}