#!/bin/bash
# install opencv: sudo apt-get install libopencv-dev or conda install opencv
nvcc -o fusion_gpu src/fusion.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib -I/rhome/jhou/.local/anaconda3/envs/sparseconv043/include -L/rhome/jhou/.local/anaconda3/envs/sparseconv043/lib -lopencv_core -lopencv_imgcodecs
g++ -o  fusion_cpu src/fusion.cpp -I/rhome/jhou/.local/anaconda3/envs/sparseconv043/include -L/rhome/jhou/.local/anaconda3/envs/sparseconv043/lib -lopencv_core -lopencv_imgcodecs -fopenmp

