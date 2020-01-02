#pragma once
#include <vector>
#include <string>
#include <fstream>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "font_structures.h"

class FontHandler {
private:
  std::vector<Font> fontList;
public:
  void createFontFromFile(std::string filename);

  Font* getFontFromName(const char* name);
  static int findIndexFromSymbol(Font font, char symbol);
};