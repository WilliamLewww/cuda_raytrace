#pragma once
#include <vector>
#include <string>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

struct Character {
  char symbol;

  int x, y;
  int width, height;
  int originX, originY;
};

struct Font {
  std::string name;

  int size;
  int bold, italic;
  int width, height;
  int characterCount;

  std::vector<Character> characterList;
  GLuint textureResource;
};