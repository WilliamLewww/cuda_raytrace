#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <string>
#include <math.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

struct Tuple {
  float x;
  float y;
  float z;
  float w;
};

extern "C" void updateCamera(double x, double y, double z, double rotation) ;
extern "C" void initializeScene();
extern "C" void renderFrame(int blockDimX, int blockDimY, void* cudaBuffer, cudaGraphicsResource_t* cudaTextureResource);

extern "C" void renderImage(int blockDimX, int blockDimY, const char* filename);

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
std::string readShaderSource(const char* filepath);
GLuint createShaderProgram(std::string vertexShaderString, std::string fragmentShaderString);

GLfloat vertices[] = {
  -1.0, -1.0,
   1.0, -1.0,
  -1.0,  1.0,
  -1.0,  1.0,
   1.0, -1.0,
   1.0,  1.0,
};

GLfloat textureCoordinates[] = {
  0.0, 0.0,
  1.0, 0.0,
  0.0, 1.0,
  0.0, 1.0,
  1.0, 0.0,
  1.0, 1.0,
};

Tuple cameraPositionVelocity = {0.0, 0.0, 0.0, 0.0};
Tuple cameraPosition = {5.0, -3.5, -6.0, 1.0};
double cameraRotationVelocity = 0.0;
double cameraRotation = -M_PI / 4.5;

int main(int argn, char** argv) {
  glfwInit();

  GLFWwindow* window = glfwCreateWindow(1000, 1000, "cuda_raytrace", NULL, NULL);
  glfwMakeContextCurrent(window);
  glewInit();

  glfwSetKeyCallback(window, keyCallback);

  GLuint shaderProgramHandle;
  std::string vertexShaderString = readShaderSource("shaders/basic.vertex");
  std::string fragmentShaderString = readShaderSource("shaders/basic.fragment");
  shaderProgramHandle = createShaderProgram(vertexShaderString, fragmentShaderString);

  struct cudaGraphicsResource* cudaTextureResource;
  GLuint textureResource;

  glGenTextures(1, &textureResource);
  glBindTexture(GL_TEXTURE_2D, textureResource);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT, 1000, 1000, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE, NULL);

  cudaGraphicsGLRegisterImage(&cudaTextureResource, textureResource, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

  void* cudaBuffer;
  cudaMalloc(&cudaBuffer, 1000*1000*4*sizeof(GLubyte));

  initializeScene();

  GLuint vao, vbo[2];
  glGenVertexArrays(1, &vao);
  glGenBuffers(2, vbo);

  GLuint textureHandle = glGetUniformLocation(shaderProgramHandle, "u_texture");

  while (!glfwWindowShouldClose(window)) {
    cameraPosition.x += cameraPositionVelocity.x;
    cameraPosition.y += cameraPositionVelocity.y;
    cameraPosition.z += cameraPositionVelocity.z;
    cameraRotation += cameraRotationVelocity;

    updateCamera(cameraPosition.x, cameraPosition.y, cameraPosition.z, cameraRotation);
    renderFrame(16, 16, cudaBuffer, &cudaTextureResource);
    glfwPollEvents();

    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textureResource);

    glUseProgram(shaderProgramHandle);

    glBindVertexArray(vao);

    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
    glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(float), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
    glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(float), textureCoordinates, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);

    glUniform1i(textureHandle, 0);

    glDrawArrays(GL_TRIANGLES, 0, 6);

    glfwSwapBuffers(window);
  }

  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
  if (action == GLFW_PRESS) {
    if (key == 87) {
      cameraPositionVelocity.x = cos(-cameraRotation + (M_PI / 2)) * 0.1;
      cameraPositionVelocity.z = sin(-cameraRotation + (M_PI / 2)) * 0.1;
    }
    if (key == 83) {
      cameraPositionVelocity.x = -cos(-cameraRotation + (M_PI / 2)) * 0.1;
      cameraPositionVelocity.z = -sin(-cameraRotation + (M_PI / 2)) * 0.1;
    }
    if (key == 65) {
      cameraPositionVelocity.x = -cos(-cameraRotation) * 0.1;
      cameraPositionVelocity.z = -sin(-cameraRotation) * 0.1;
    }
    if (key == 68) {
      cameraPositionVelocity.x = cos(-cameraRotation) * 0.1;
      cameraPositionVelocity.z = sin(-cameraRotation) * 0.1;
    }
    if (key == 82) {
      cameraPositionVelocity.y = -0.05;
    }
    if (key == 70) {
      cameraPositionVelocity.y = 0.05;
    }
    if (key == 69) {
      cameraRotationVelocity = 0.01;
    }
    if (key == 81) {
      cameraRotationVelocity = -0.01;
    }
  }

  if (action == GLFW_RELEASE) {
    if (key == 87 || key == 83 ||key == 65 || key == 68) {
      cameraPositionVelocity.x = 0.0;
      cameraPositionVelocity.z = 0.0;
    }
    if (key == 82 || key == 70) {
      cameraPositionVelocity.y = 0.0;
    }
    if (key == 69 || key == 81) {
      cameraRotationVelocity = 0.0;
    }
  }
}

std::string readShaderSource(const char* filepath) {
    std::string content;
    std::ifstream fileStream(filepath, std::ios::in);
    std::string line = "";

    while (!fileStream.eof()) {
        getline(fileStream, line);
        content.append(line + "\n");
    }
    fileStream.close();

    return content;
}

GLuint createShaderProgram(std::string vertexShaderString, std::string fragmentShaderString) {
    const char* vertexShaderSource = vertexShaderString.c_str();
    const char* fragmentShaderSource = fragmentShaderString.c_str();

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);

    glCompileShader(vertexShader);
    glCompileShader(fragmentShader);

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    return shaderProgram;
}