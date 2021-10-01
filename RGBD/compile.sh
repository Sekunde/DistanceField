#!/bin/bash
# install opencv: sudo apt-get install libopencv-dev
/usr/local/cuda/bin/nvcc -std=c++11 -O3 -o fusion cpp/fusion.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib  `pkg-config --cflags --libs opencv`
