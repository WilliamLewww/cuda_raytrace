CUDA_PATH=/usr/local/cuda-10.1
GLFW_PATH=/usr/local/glfw-3.3
GLEW_PATH=/usr/local/glew-2.1.0

CURRENT_PATH=$(shell pwd)

BIN_PATH=$(CURRENT_PATH)/bin
SRC_PATH=$(CURRENT_PATH)/src

GLFW_LIBRARY_PATH=$(GLFW_PATH)/glfw-build/src
GLFW_INCLUDE_PATH=$(GLFW_PATH)/include

GLEW_LIBRARY_PATH=$(GLEW_PATH)/lib
GLEW_INCLUDE_PATH=$(GLEW_PATH)/include

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

SRCS := main.cpp engine.cpp joiner.cpp input.cpp raytrace_rectangle.cpp raytrace_image.cpp structures.cpp model.cpp
SRCS += character_rectangle.cpp text_container.cpp squiggle_animation_text.cpp font_handler.cpp
OBJS := $(SRCS:%.cpp=%.o)

CUDA_SRCS := renderer_triangles.cu
CUDA_OBJS := $(CUDA_SRCS:%.cu=%.o)

$(EXEC): $(OBJS) $(CUDA_OBJS)
	$(NVCC) $(CUDA_FLAGS) $(BIN_PATH)/*.o -o $(BIN_PATH)/$(EXEC) $(LINKER_ARGUMENTS)

%.o: $(SRC_PATH)/%.cpp
	$(NVCC) $(CUDA_FLAGS) --device-c $^ -o $(BIN_PATH)/$@ $(LINKER_ARGUMENTS)

%.o: $(SRC_PATH)/%.cu
	$(NVCC) $(CUDA_FLAGS) --device-c $^ -o $(BIN_PATH)/$@ $(LINKER_ARGUMENTS)

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
	mkdir -p bin