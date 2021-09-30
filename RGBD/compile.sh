#!/bin/bash
nvcc -std=c++11 -O3 -o demo demo.cu -I/usr/local/cuda/include -L$CUDA_LIB_DIR -lcudart -lcublas -lcurand -D_MWAITXINTRIN_H_INCLUDED `pkg-config --cflags --libs opencv`
