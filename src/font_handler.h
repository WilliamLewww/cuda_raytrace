#pragma once
#include <vector>
#include <string>
#include <fstream>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "font_structures.h"

class FontHandler {
private:
  static std::vector<Font> fontList;

  static Character charactersUbuntu[];
  static Font fontArray[];

  static int fontCount;
public:
  static GLuint fontUbuntuTextureResource;

  static void createFontFromFile(std::string filename);

  static Font* findFontFromName(const char* name);
  static int findIndexFromSymbol(Font font, char symbol);

  static void initialize();
};