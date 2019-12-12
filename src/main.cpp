#include <stdlib.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

extern "C" void renderImage(int blockDimX, int blockDimY, const char* filename);

int main(int argn, char** argv) {
  glfwInit();

  GLFWwindow* window = glfwCreateWindow(1000, 1000, "cuda_raytrace", NULL, NULL);
  glfwMakeContextCurrent(window);
  glewInit();

  while (!glfwWindowShouldClose(window)) {
    glClear(GL_COLOR_BUFFER_BIT);
    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glfwTerminate();
  return 0;
}