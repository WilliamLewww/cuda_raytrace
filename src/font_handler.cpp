#include "font_handler.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

Font* FontHandler::getFontFromName(const char* name) {
  for (int x = 0; x < fontList.size(); x++) {
    if (strcmp(fontList[x].name.c_str(), name) == 0) {
      return &fontList[x];
    }
  }

  return nullptr;
}

int FontHandler::findIndexFromSymbol(Font font, char symbol) {
  for (int x = 0; x < font.characterList.size(); x++) {
    if (font.characterList[x].symbol == symbol) {
      return x;
    }
  }

  return -1;
}

void FontHandler::createFontFromFile(std::string filename) {
  std::string dateFile = filename + ".glyph";
  std::string imageFile = filename + ".png";

  std::ifstream file(dateFile);
  std::string line;

  Font font;

  while (std::getline(file, line)) {
    std::string temp = line;

    if (temp.at(0) == '\"') {
      temp = temp.substr(1);
      font.name = temp.substr(0, temp.find_first_of("\"")).c_str();
      temp = temp.substr(temp.find_first_of("\"") + 3);
      font.size = std::stoi(temp.substr(0, temp.find_first_of(",")));
      temp = temp.substr(temp.find_first_of(" ") + 1);
      font.bold = std::stoi(temp.substr(0, temp.find_first_of(",")));
      temp = temp.substr(temp.find_first_of(" ") + 1);
      font.italic = std::stoi(temp.substr(0, temp.find_first_of(",")));
      temp = temp.substr(temp.find_first_of(" ") + 1);
      font.width = std::stoi(temp.substr(0, temp.find_first_of(",")));
      temp = temp.substr(temp.find_first_of(" ") + 1);
      font.height = std::stoi(temp.substr(0, temp.find_first_of(",")));
      temp = temp.substr(temp.find_first_of(" ") + 1);
      font.characterCount = std::stoi(temp);
    }

    if (temp.at(0) == '\'') {
      Character character;

      temp = temp.substr(1);
      character.symbol = temp.substr(0, temp.find_first_of("\'")).c_str()[0];
      temp = temp.substr(temp.find_first_of("\'") + 3);
      character.x = std::stoi(temp.substr(0, temp.find_first_of(",")));
      temp = temp.substr(temp.find_first_of(" ") + 1);
      character.y = std::stoi(temp.substr(0, temp.find_first_of(",")));
      temp = temp.substr(temp.find_first_of(" ") + 1);
      character.width = std::stoi(temp.substr(0, temp.find_first_of(",")));
      temp = temp.substr(temp.find_first_of(" ") + 1);
      character.height = std::stoi(temp.substr(0, temp.find_first_of(",")));
      temp = temp.substr(temp.find_first_of(" ") + 1);
      character.originX = std::stoi(temp.substr(0, temp.find_first_of(",")));
      temp = temp.substr(temp.find_first_of(" ") + 1);
      character.originY = std::stoi(temp.substr(0, temp.find_first_of(",")));

      font.characterList.push_back(character);
    }
  }

  int w, h, comp;
  unsigned char* image = stbi_load(imageFile.c_str(), &w, &h, &comp, STBI_rgb_alpha);

  glGenTextures(1, &font.textureResource);
  glBindTexture(GL_TEXTURE_2D, font.textureResource);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, image);
  glBindTexture(GL_TEXTURE_2D, 0);

  stbi_image_free(image);

  fontList.push_back(font);
}