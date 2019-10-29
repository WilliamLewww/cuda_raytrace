CUDAPATH=/usr/local/cuda-10.1
CURRENTPATH=$(shell pwd)

CC=g++
NVCC=$(CUDAPATH)/bin/nvcc
NVPROF=$(CUDAPATH)/bin/nvprof
MEMCHECK=$(CUDAPATH)/bin/cuda-memcheck
NSIGHTCLI=$(CUDAPATH)/bin/nv-nsight-cu-cli
NVVP=$(CUDAPATH)/bin/nvvp

CUDAFLAGS=--gpu-architecture=sm_50 -rdc=true

all: compile run

clean:
	rm -rf bin
	rm -rf dump

compile:
	mkdir -p bin
	cd bin; $(NVCC) $(CUDAFLAGS) --device-c ../src/*.cu
	cd bin; $(NVCC) $(CUDAFLAGS) *.o -o raytrace_renderer.out

run:
	mkdir -p dump
	cd dump; ../bin/raytrace_renderer.out

open:
	cd dump; xdg-open image.ppm

memory-check:
	mkdir -p dump
	cd dump; $(MEMCHECK) ../bin/raytrace_renderer.out

profile:
	mkdir -p dump
	cd dump; sudo $(NVPROF) ../bin/raytrace_renderer.out 2>profile.log; cat profile.log;

profile-metrics:
	mkdir -p dump
	cd dump; sudo $(NVPROF) --metrics all ../bin/raytrace_renderer.out 2>profile-metrics.log; cat profile-metrics.log;

profile-events:
	mkdir -p dump
	cd dump; sudo $(NVPROF) --events all ../bin/raytrace_renderer.out 2>profile-events.log; cat profile-events.log;

nsight-cli:
	mkdir -p dump
	cd dump; sudo $(NSIGHTCLI) ../bin/raytrace_renderer.out > nsight-cli.log; cat nsight-cli.log;

nvvp:
	sudo $(NVVP) $(CURRENTPATH)/bin/$(CURRENTFILE).out -vm /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java