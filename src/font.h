#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>

struct Character {
  char symbol;
  int x, y;
  int width, height;
  int originX, originY;
};

struct Font {
  const char* name;
  int size;
  int bold, italic;
  int width, height;
  int characterCount;

  Character* characters;
};

class FontHolder {
private:
  static Character charactersUbuntu[];
public:
  static Font fontUbuntu;
  static GLuint fontUbuntuTextureResource;

  static void initialize();
};