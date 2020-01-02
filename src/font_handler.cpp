#include "font_handler.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

Character FontHandler::charactersUbuntu[] = {
  {' ', 245, 106, 3, 3, 1, 1},
  {'!', 138, 82, 7, 24, -1, 23},
  {'"', 156, 106, 12, 11, -1, 25},
  {'#', 68, 58, 21, 24, 0, 23},
  {'$', 140, 0, 16, 30, -1, 26},
  {'%', 281, 0, 27, 25, 0, 24},
  {'&', 0, 33, 22, 25, 0, 24},
  {'\'', 175, 106, 6, 11, -1, 25},
  {'(', 58, 0, 11, 33, -1, 26},
  {')', 69, 0, 10, 33, 1, 26},
  {'*', 104, 106, 16, 16, 0, 24},
  {'+', 50, 106, 18, 18, 0, 19},
  {',', 168, 106, 7, 11, 0, 5},
  {'-', 234, 106, 11, 5, 1, 12},
  {'.', 207, 106, 8, 7, 0, 6},
  {'/', 0, 0, 17, 33, 2, 26},
  {'0', 64, 33, 18, 25, 0, 24},
  {'1', 119, 82, 11, 24, -2, 23},
  {'2', 190, 33, 17, 25, 0, 24},
  {'3', 207, 33, 17, 25, 0, 24},
  {'4', 309, 58, 18, 24, 0, 23},
  {'5', 87, 82, 16, 24, -1, 23},
  {'6', 36, 82, 17, 24, -1, 23},
  {'7', 0, 82, 18, 24, 0, 23},
  {'8', 82, 33, 18, 25, 0, 24},
  {'9', 100, 33, 18, 25, 0, 24},
  {':', 42, 106, 8, 19, 0, 18},
  {';', 130, 82, 8, 24, 0, 18},
  {'<', 68, 106, 18, 17, 0, 18},
  {'=', 138, 106, 18, 12, 0, 16},
  {'>', 86, 106, 18, 17, 0, 18},
  {'?', 241, 33, 15, 25, 1, 24},
  {'@', 156, 0, 29, 29, -1, 24},
  {'A', 0, 58, 23, 24, 1, 23},
  {'B', 252, 58, 19, 24, -1, 23},
  {'C', 22, 33, 21, 25, 0, 24},
  {'D', 89, 58, 21, 24, -1, 23},
  {'E', 18, 82, 18, 24, -1, 23},
  {'F', 53, 82, 17, 24, -1, 23},
  {'G', 43, 33, 21, 25, 0, 24},
  {'H', 152, 58, 20, 24, -1, 23},
  {'I', 145, 82, 6, 24, -1, 23},
  {'J', 103, 82, 16, 24, 1, 23},
  {'K', 172, 58, 20, 24, -1, 23},
  {'L', 70, 82, 17, 24, -1, 23},
  {'M', 300, 33, 26, 24, -1, 23},
  {'N', 110, 58, 21, 24, -1, 23},
  {'O', 308, 0, 25, 25, 0, 24},
  {'P', 271, 58, 19, 24, -1, 23},
  {'Q', 104, 0, 25, 31, 0, 24},
  {'R', 192, 58, 20, 24, -1, 23},
  {'S', 118, 33, 18, 25, 0, 24},
  {'T', 212, 58, 20, 24, 1, 23},
  {'U', 232, 58, 20, 24, -1, 23},
  {'V', 23, 58, 23, 24, 1, 23},
  {'W', 269, 33, 31, 24, 1, 23},
  {'X', 46, 58, 22, 24, 1, 23},
  {'Y', 131, 58, 21, 24, 1, 23},
  {'Z', 290, 58, 19, 24, 0, 23},
  {'[', 79, 0, 10, 33, -2, 26},
  {'\\', 17, 0, 17, 33, 2, 26},
  {']', 89, 0, 10, 33, 1, 26},
  {'^', 120, 106, 18, 14, 0, 23},
  {'_', 215, 106, 19, 5, 2, -2},
  {'`', 181, 106, 8, 9, -1, 26},
  {'a', 277, 82, 16, 19, 0, 18},
  {'b', 202, 0, 18, 27, -1, 26},
  {'c', 293, 82, 16, 19, 0, 18},
  {'d', 220, 0, 18, 27, 0, 26},
  {'e', 223, 82, 18, 19, 0, 18},
  {'f', 254, 0, 13, 27, -1, 26},
  {'g', 224, 33, 17, 25, 0, 18},
  {'h', 238, 0, 16, 27, -1, 26},
  {'i', 275, 0, 6, 26, -1, 25},
  {'j', 129, 0, 11, 31, 4, 24},
  {'k', 185, 0, 17, 28, -1, 27},
  {'l', 267, 0, 8, 27, -1, 26},
  {'m', 178, 82, 26, 19, -1, 18},
  {'n', 309, 82, 16, 19, -1, 18},
  {'o', 204, 82, 19, 19, 0, 18},
  {'p', 136, 33, 18, 25, -1, 18},
  {'q', 154, 33, 18, 25, 0, 18},
  {'r', 30, 106, 12, 19, -1, 18},
  {'s', 0, 106, 15, 19, 0, 18},
  {'t', 256, 33, 13, 25, -1, 24},
  {'u', 325, 82, 16, 19, -1, 18},
  {'v', 241, 82, 18, 19, 1, 18},
  {'w', 151, 82, 27, 19, 1, 18},
  {'x', 259, 82, 18, 19, 1, 18},
  {'y', 172, 33, 18, 25, 1, 18},
  {'z', 15, 106, 15, 19, 0, 18},
  {'{', 34, 0, 12, 33, 0, 26},
  {'|', 99, 0, 5, 33, -2, 26},
  {'}', 46, 0, 12, 33, 1, 26},
  {'~', 189, 106, 18, 7, 0, 13},
};
GLuint FontHandler::fontUbuntuTextureResource;

Font FontHandler::fontArray[] = {{"Ubuntu", 32, 0, 0, 341, 125, 95, charactersUbuntu}};
int FontHandler::fontCount = 1;

std::vector<Font> FontHandler::fontList;

Font* FontHandler::findFontFromName(const char* name) {
  for (int x = 0; x < fontCount; x++) {
    if (strcmp(fontArray[x].name, name) == 0) {
      return &fontArray[x];
    }
  }

  return nullptr;
}

int FontHandler::findIndexFromSymbol(Font font, char symbol) {
  for (int x = 0; x < font.characterCount; x++) {
    if (font.characters[x].symbol == symbol) {
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

void FontHandler::initialize() {
  int w, h, comp;
  unsigned char* image = stbi_load("res/font_ubuntu.png", &w, &h, &comp, STBI_rgb_alpha);

  glGenTextures(1, &fontUbuntuTextureResource);
  glBindTexture(GL_TEXTURE_2D, fontUbuntuTextureResource);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, image);
  glBindTexture(GL_TEXTURE_2D, 0);

  stbi_image_free(image);
}