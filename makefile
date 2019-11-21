BIN_PATH=./bin/
CUDA_PATH=/usr/local/cuda-10.1
CURRENT_PATH=$(shell pwd)

CC=g++
NVCC=$(CUDA_PATH)/bin/nvcc
NVPROF=$(CUDA_PATH)/bin/nvprof
NSIGHT_CLI=$(CUDA_PATH)/bin/nv-nsight-cu-cli
NVVP=$(CUDA_PATH)/bin/nvvp
CUDA_GDB=$(CUDA_PATH)/bin/cuda-gdb
MEMCHECK=$(CUDA_PATH)/bin/cuda-memcheck

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

profile:
	sudo $(NVPROF) $(BIN_PATH)$(EXEC) $(EXEC_ARGS) 2>$(BIN_PATH)profile.log; cat $(BIN_PATH)profile.log;

nvvp:
	sudo $(NVVP) $(CURRENT_PATH)/bin/$(EXEC) $(EXEC_ARGS) -vm /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java

cuda-gdb:
	$(CUDA_GDB) $(BIN_PATH)$(EXEC)

memory-check:
	$(MEMCHECK) $(BIN_PATH)$(EXEC) $(EXEC_ARGS) 2>$(BIN_PATH)memory-check.log; cat $(BIN_PATH)memory-check.log;

open:
	xdg-open bin/image.ppm

clean:
	rm -rf $(BIN_PATH)*