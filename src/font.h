#pragma once

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

extern Character charactersUbuntu[];
extern Font fontUbuntu;