#include "squiggle_animation_text.h"
#include <stdio.h>

int movingDownLower = 0;
int movingDownUpper = 0;

void SquiggleAnimationText::initialize(std::vector<CharacterRectangle>* characterRectangleList) {
  this->characterRectangleList = characterRectangleList;

  movingDownLower = 0;
  movingDownUpper = characterRectangleList->size();
}

int counter = 0;
void SquiggleAnimationText::animate() {
  if (counter < 15) {
    counter += 1;
  }
  else {
    counter = 0;

    if (movingDownLower < characterRectangleList->size() * 2 + 1) { movingDownLower += 1; }
    else { movingDownLower = 0; }
    if (movingDownUpper < characterRectangleList->size() * 2 + 1) { movingDownUpper += 1; }
    else { movingDownUpper = 0; }
  }

  for (int x = 0; x < characterRectangleList->size(); x++) {
    if (movingDownLower < movingDownUpper) {
      if (x >= movingDownLower && x <= movingDownUpper) {
        (*characterRectangleList)[x].addPosition(0.0, -0.0001);
      }
      if (x < movingDownLower || x > movingDownUpper) {
        (*characterRectangleList)[x].addPosition(0.0, 0.0001);
      }
    }
    else {
      if (x >= movingDownLower || x <= movingDownUpper) {
        (*characterRectangleList)[x].addPosition(0.0, -0.0001);
      }
      if (x < movingDownLower && x > movingDownUpper) {
        (*characterRectangleList)[x].addPosition(0.0, 0.0001);
      }
    }
  }
}