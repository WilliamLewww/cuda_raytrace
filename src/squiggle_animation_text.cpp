#include "squiggle_animation_text.h"

int movingDownLower = 0;
int movingDownUpper = 0;

int movingUpLower = 0;
int movingUpUpper = 0;

void SquiggleAnimationText::initialize(std::vector<CharacterRectangle>* characterRectangleList) {
  this->characterRectangleList = characterRectangleList;

  movingUpLower = characterRectangleList->size();
  movingUpUpper = characterRectangleList->size();
}

int counter = 0;
void SquiggleAnimationText::animate() {
  for (int x = movingDownLower; x < movingDownUpper; x++) {
    (*characterRectangleList)[x].addPosition(0.0, -0.0002);
  }

  for (int x = movingUpLower; x < movingUpUpper; x++) {
    (*characterRectangleList)[x].addPosition(0.0, 0.0002);
  }
}