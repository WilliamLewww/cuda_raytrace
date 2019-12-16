BIN_PATH=./bin
CUDA_PATH=/usr/local/cuda-10.1

GLFW_PATH=/usr/local/glfw-3.3
GLFW_LIBRARY_PATH=$(GLFW_PATH)/glfw-build/src
GLFW_INCLUDE_PATH=$(GLFW_PATH)/include

GLEW_PATH=/usr/local/glew-2.1.0
GLEW_LIBRARY_PATH=$(GLEW_PATH)/lib
GLEW_INCLUDE_PATH=$(GLEW_PATH)/include

CURRENT_PATH=$(shell pwd)

CC=g++
NVCC=$(CUDA_PATH)/bin/nvcc
NVPROF=$(CUDA_PATH)/bin/nvprof
NSIGHT_CLI=$(CUDA_PATH)/bin/nv-nsight-cu-cli
NVVP=$(CUDA_PATH)/bin/nvvp
CUDA_GDB=$(CUDA_PATH)/bin/cuda-gdb
MEMCHECK=$(CUDA_PATH)/bin/cuda-memcheck

CUDA_FLAGS=--gpu-architecture=sm_30
LIBRARIES=-lglfw3 -lGLEW -lGL -lGLU -lXrandr -lXext -lX11
LINKER_ARGUMENTS=-L$(GLFW_LIBRARY_PATH) -L$(GLEW_LIBRARY_PATH) -I$(GLFW_INCLUDE_PATH) -I$(GLEW_INCLUDE_PATH) $(LIBRARIES)

EXEC=raytrace_renderer
EXEC_ARGS=bin/image.ppm 16 16

all: clean $(EXEC) run

$(EXEC): main.o engine.o input.o joiner.o renderer.o
	$(NVCC) $(CUDA_FLAGS) $(BIN_PATH)/*.o -o $(BIN_PATH)/$(EXEC) $(LINKER_ARGUMENTS)

main.o: ./src/main.cpp
	$(NVCC) $(CUDA_FLAGS) --device-c $^ -o $(BIN_PATH)/main.o $(LINKER_ARGUMENTS)

engine.o: ./src/engine.cpp
	$(NVCC) $(CUDA_FLAGS) --device-c $^ -o $(BIN_PATH)/engine.o $(LINKER_ARGUMENTS)

joiner.o: ./src/joiner.cpp
	$(NVCC) $(CUDA_FLAGS) --device-c $^ -o $(BIN_PATH)/joiner.o $(LINKER_ARGUMENTS)

input.o: ./src/input.cpp
	$(NVCC) $(CUDA_FLAGS) --device-c $^ -o $(BIN_PATH)/input.o $(LINKER_ARGUMENTS)

renderer.o: ./src/renderer.cu
	$(NVCC) $(CUDA_FLAGS) --device-c $^ -o $(BIN_PATH)/renderer.o $(LINKER_ARGUMENTS)

run:
	$(BIN_PATH)/$(EXEC) $(EXEC_ARGS)

profile:
	sudo $(NVPROF) $(BIN_PATH)/$(EXEC) $(EXEC_ARGS) 2>$(BIN_PATH)/profile.log; cat $(BIN_PATH)/profile.log;

nvvp:
	sudo $(NVVP) $(CURRENT_PATH)/bin/$(EXEC) $(EXEC_ARGS) -vm /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java

cuda-gdb:
	$(CUDA_GDB) $(BIN_PATH)/$(EXEC)

memory-check:
	$(MEMCHECK) $(BIN_PATH)/$(EXEC) $(EXEC_ARGS) 2>$(BIN_PATH)/memory-check.log; cat $(BIN_PATH)/memory-check.log;

open:
	xdg-open bin/image.ppm

clean:
	rm -rf $(BIN_PATH)/*