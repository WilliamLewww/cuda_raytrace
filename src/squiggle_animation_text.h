#pragma once
#include <vector>

#include "character_rectangle.h"

class SquiggleAnimationText {
private:
  std::vector<CharacterRectangle>* characterRectangleList;

  int movingDownLower;
  int movingDownUpper;
  int counter;
public:
  void initialize(std::vector<CharacterRectangle>* characterRectangleList);
  void animate();
};