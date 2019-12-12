#include <stdlib.h>
#include <fstream>
#include <string>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

extern "C" void renderImage(int blockDimX, int blockDimY, const char* filename);

std::string readShaderSource(const char* filepath);
GLuint createShaderProgram(std::string vertexShaderString, std::string fragmentShaderString);

GLfloat vertices[] = {
  1.0f, 1.0f, 0.5f, 1.0f, 1.0f,
  1.0f, -1.0f, 0.5f, 1.0f, 0.0f,
  -1.0f, -1.0f, 0.5f, 0.0f, 0.0f,
  -1.0f, 1.0f, 0.5f, 0.0f, 1.0f,
};

int main(int argn, char** argv) {
  glfwInit();

  GLFWwindow* window = glfwCreateWindow(1000, 1000, "cuda_raytrace", NULL, NULL);
  glfwMakeContextCurrent(window);
  glewInit();

  GLuint opengl_tex_cuda;
  void* cuda_dev_render_buffer;
  struct cudaGraphicsResource* cuda_tex_resource;

  glGenTextures(1, &opengl_tex_cuda);
  glBindTexture(GL_TEXTURE_2D, opengl_tex_cuda);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT, 1000, 1000, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE, NULL);
  cudaGraphicsGLRegisterImage(&cuda_tex_resource, opengl_tex_cuda, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

  GLuint shaderProgramHandle;
  std::string vertexShaderString = readShaderSource("shaders/basic.vertex");
  std::string fragmentShaderString = readShaderSource("shaders/basic.fragment");
  shaderProgramHandle = createShaderProgram(vertexShaderString, fragmentShaderString);

  cudaMalloc(&cuda_dev_render_buffer, 1000*1000*4*sizeof(GLubyte));

  GLuint vao, vbo;
  glGenVertexArrays(1, &vao);
  glGenBuffers(1, &vbo);

  glBindVertexArray(vao);

  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)0);
  glEnableVertexAttribArray(0);

  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
  glEnableVertexAttribArray(1);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(shaderProgramHandle);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, opengl_tex_cuda);
    glUniform1i(glGetUniformLocation(shaderProgramHandle, "u_textureSampler"), 0);

    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);

    glfwSwapBuffers(window);
  }

  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
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