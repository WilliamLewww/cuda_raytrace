#pragma once
#include <cstring>

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
  static Font fontArray[];

  static int fontCount;
public:
  static GLuint fontUbuntuTextureResource;

  static Font* findFontFromName(const char* name);
  static int findIndexFromSymbol(Font font, char symbol);

  static void initialize();
};