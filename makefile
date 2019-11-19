BIN_PATH=./bin/
CUDA_PATH=/usr/local/cuda-10.1
CURRENT_PATH=$(shell pwd)

CC=g++
NVCC=$(CUDA_PATH)/bin/nvcc
NVPROF=$(CUDA_PATH)/bin/nvprof
MEMCHECK=$(CUDA_PATH)/bin/cuda-memcheck
NSIGHT_CLI=$(CUDA_PATH)/bin/nv-nsight-cu-cli
NVVP=$(CUDA_PATH)/bin/nvvp

CUDA_FLAGS=--gpu-architecture=sm_50

EXEC=raytrace_renderer
EXEC_ARGS=bin/image.ppm

all: clean $(EXEC) run

$(EXEC): main.o
	$(NVCC) $(CUDA_FLAGS) $(BIN_PATH)*.o -o $(BIN_PATH)$(EXEC)

main.o: ./src/main.cu
	$(NVCC) $(CUDA_FLAGS) --device-c $^ -o $(BIN_PATH)main.o

run:
	$(BIN_PATH)$(EXEC) $(EXEC_ARGS)

memory-check:
	$(MEMCHECK) $(BIN_PATH)$(EXEC) $(EXEC_ARGS)

profile:
	sudo $(NVPROF) $(BIN_PATH)$(EXEC) $(EXEC_ARGS) 2>$(BIN_PATH)profile.log; cat $(BIN_PATH)profile.log;

open:
	xdg-open bin/image.ppm

clean:
	rm -rf $(BIN_PATH)*