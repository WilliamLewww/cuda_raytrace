#pragma once
#include <cstring>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "character.h"

struct Font {
  const char* name;

  int size;
  int bold, italic;
  int width, height;
  int characterCount;

  Character* characters;
};