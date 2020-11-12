# cuda_raytrace

[![Video](https://img.youtube.com/vi/6RQwBniXa-M/hqdefault.jpg)](https://www.youtube.com/watch?v=6RQwBniXa-M)

cuda_raytrace contains a real-time ray-tracer engine that uses OpenGL's interoperability with CUDA. The ray-traced graphics are generated via CUDA which are then copied into a OpenGL texture. The texture is then applied to a quad that encompass the entire view-port.

## Dependencies
- CUDA (cuda-10.2)
- GLFW (glfw-3.3.2)
- GLEW (glew-2.1.0)
- GNU make

## Building

Edit the dependencies' path in the makefile. Use the makefile to compile or run the program.

```bash
# makefile
CUDA_PATH=/usr/local/cuda-10.2
GLFW_PATH=/usr/local/glfw-3.3.2
GLEW_PATH=/usr/local/glew-2.1.0
```
