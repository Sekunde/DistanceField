#!/bin/bash
# install opencv: sudo apt-get install libopencv-dev
#/usr/local/cuda/bin/nvcc -o fusion_gpu cpp/fusion.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib -I./dependencies/opencv ./dependencies/libopencv_highgui.so ./dependencies/libopencv_core.so
g++ -o fusion_cpu cpp/fusion.cpp -I./dependencies/include -L./dependencies/lib -lopencv_highgui -lopencv_core

