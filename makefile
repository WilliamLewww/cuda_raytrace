CUDA_PATH=/usr/local/cuda-10.2
GLFW_PATH=/usr/local/glfw-3.3.2
GLEW_PATH=/usr/local/glew-2.1.0

CURRENT_PATH=$(shell pwd)

BIN_PATH=$(CURRENT_PATH)/bin
BUILD_PATH=$(CURRENT_PATH)/build
DUMP_PATH=$(CURRENT_PATH)/dump
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
GDB=gdb
CUDA_GDB=$(CUDA_PATH)/bin/cuda-gdb
MEMCHECK=$(CUDA_PATH)/bin/cuda-memcheck

CUDA_FLAGS=--gpu-architecture=sm_30
LIBRARIES=-lglfw3 -lGLEW -lGL -lGLU -lXrandr -lXext -lX11
LINKER_ARGUMENTS=-L$(GLFW_LIBRARY_PATH) -L$(GLEW_LIBRARY_PATH) -I$(GLFW_INCLUDE_PATH) -I$(GLEW_INCLUDE_PATH) $(LIBRARIES)

EXEC=raytrace_renderer.out
EXEC_ARGS=dump/image.ppm 16 16

all: clean $(EXEC) run

SRCS := $(wildcard src/*.cpp)
OBJS := $(SRCS:src/%.cpp=%.o)

CUDA_SRCS := $(wildcard src/*.cu)
CUDA_OBJS := $(CUDA_SRCS:src/%.cu=%.o)

$(EXEC): $(OBJS) $(CUDA_OBJS)
	$(NVCC) $(CUDA_FLAGS) $(BUILD_PATH)/*.o -o $(BIN_PATH)/$(EXEC) $(LINKER_ARGUMENTS)

%.o: $(SRC_PATH)/%.cpp
	$(NVCC) $(CUDA_FLAGS) --device-c $^ -o $(BUILD_PATH)/$@ $(LINKER_ARGUMENTS)

%.o: $(SRC_PATH)/%.cu
	$(NVCC) $(CUDA_FLAGS) --device-c $^ -o $(BUILD_PATH)/$@ $(LINKER_ARGUMENTS)

run:
	$(BIN_PATH)/$(EXEC) $(EXEC_ARGS)

profile:
	sudo $(NVPROF) $(BUILD_PATH)/$(EXEC) $(EXEC_ARGS) 2>$(BUILD_PATH)/profile.log; cat $(BUILD_PATH)/profile.log;

nvvp:
	sudo $(NVVP) $(CURRENT_PATH)/bin/$(EXEC) $(EXEC_ARGS) -vm /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java

gdb:
	gdb $(CURRENT_PATH)/bin/$(EXEC)

cuda-gdb:
	$(CUDA_GDB) $(BUILD_PATH)/$(EXEC)

memory-check:
	$(MEMCHECK) $(BUILD_PATH)/$(EXEC) $(EXEC_ARGS) 2>$(BUILD_PATH)/memory-check.log; cat $(BUILD_PATH)/memory-check.log;

clean:
	rm -rf $(BUILD_PATH)/*
	mkdir -p $(BUILD_PATH)
	mkdir -p $(BIN_PATH)
	mkdir -p $(DUMP_PATH)
