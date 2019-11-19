BIN_PATH=./bin/
CUDA_PATH=/usr/local/cuda-10.1
CURRENT_PATH=$(shell pwd)

CC=g++
NVCC=$(CUDA_PATH)/bin/nvcc
NVPROF=$(CUDA_PATH)/bin/nvprof
MEMCHECK=$(CUDA_PATH)/bin/cuda-memcheck
NSIGHTCLI=$(CUDA_PATH)/bin/nv-nsight-cu-cli
NVVP=$(CUDA_PATH)/bin/nvvp

CUDA_FLAGS=--gpu-architecture=sm_50

all: clean raytrace_renderer run

raytrace_renderer: main.o
	$(NVCC) $(CUDA_FLAGS) $(BIN_PATH)*.o -o $(BIN_PATH)raytrace_renderer

main.o: ./src/main.cu
	$(NVCC) $(CUDA_FLAGS) --device-c $^ -o $(BIN_PATH)main.o

run:
	$(BIN_PATH)raytrace_renderer bin/image.ppm

open:
	xdg-open bin/image.ppm

clean:
	rm -rf $(BIN_PATH)*