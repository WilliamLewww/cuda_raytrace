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
  FontHandler();
  ~FontHandler();
  
  static int getIndexFromSymbol(Font font, char symbol);
  Font* getFontFromName(const char* name);
  
  void addFontFromFile(std::string filename);
};