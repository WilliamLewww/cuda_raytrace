#pragma once
#include <vector>

#include "character_rectangle.h"

class SquiggleAnimationText {
private:
  std::vector<CharacterRectangle>* characterRectangleList;
public:
  void initialize(std::vector<CharacterRectangle>* characterRectangleList);
  void animate();
};