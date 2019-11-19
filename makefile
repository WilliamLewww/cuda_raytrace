CUDA_PATH=/usr/local/cuda-10.1

NVCC=$(CUDA_PATH)/bin/nvcc
CUDA_FLAGS=--gpu-architecture=sm_50

BIN_PATH=./bin/

raytrace_renderer: main.o
	$(NVCC) $(CUDA_FLAGS) $(BIN_PATH)*.o -o $(BIN_PATH)raytrace_renderer

main.o: ./src/main.cu
	$(NVCC) $(CUDA_FLAGS) --device-c $^ -o $(BIN_PATH)main.o

clean:
	rm -rf $(BIN_PATH)*