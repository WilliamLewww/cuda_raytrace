CUDAPATH=/usr/local/cuda-10.1
CURRENTPATH=$(shell pwd)

CC=g++
NVCC=$(CUDAPATH)/bin/nvcc
NVPROF=$(CUDAPATH)/bin/nvprof
MEMCHECK=$(CUDAPATH)/bin/cuda-memcheck
NSIGHTCLI=$(CUDAPATH)/bin/nv-nsight-cu-cli
NVVP=$(CUDAPATH)/bin/nvvp

CUDAFLAGS=--gpu-architecture=sm_50 -rdc=true

OUPUTFILE=raytrace_renderer.out
IMAGEFILE=image.ppm

all: compile run

clean:
	rm -rf bin
	rm -rf dump

compile:
	mkdir -p bin
	cd bin; $(NVCC) $(CUDAFLAGS) --device-c ../src/*.cu
	cd bin; $(NVCC) $(CUDAFLAGS) *.o -o $(OUPUTFILE)

run:
	mkdir -p dump
	cd dump; ../bin/$(OUPUTFILE)

open:
	cd dump; xdg-open $(IMAGEFILE)

memory-check:
	mkdir -p dump
	cd dump; $(MEMCHECK) ../bin/$(OUPUTFILE)

profile:
	mkdir -p dump
	cd dump; sudo $(NVPROF) ../bin/$(OUPUTFILE) 2>profile.log; cat profile.log;

profile-metrics:
	mkdir -p dump
	cd dump; sudo $(NVPROF) --metrics all ../bin/$(OUPUTFILE) 2>profile-metrics.log; cat profile-metrics.log;

profile-events:
	mkdir -p dump
	cd dump; sudo $(NVPROF) --events all ../bin/$(OUPUTFILE) 2>profile-events.log; cat profile-events.log;

nsight-cli:
	mkdir -p dump
	cd dump; sudo $(NSIGHTCLI) ../bin/$(OUPUTFILE) > nsight-cli.log; cat nsight-cli.log;

nvvp:
	sudo $(NVVP) $(CURRENTPATH)/bin/$(OUPUTFILE) -vm /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java