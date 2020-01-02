#pragma once
#include <cstring>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "font.h"

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