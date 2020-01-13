#include "camera.h"

Camera::Camera() {
  position = {5.0, -3.5, -6.0, 1.0};
  pitch = -M_PI / 12.0;
  yaw = -M_PI / 4.5;
}

Camera::~Camera() {

}