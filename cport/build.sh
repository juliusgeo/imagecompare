#!/bin/bash
g++ -g $(pkg-config --cflags --libs opencv) keypoint.cpp -o keypoint.out
time ./keypoint.out