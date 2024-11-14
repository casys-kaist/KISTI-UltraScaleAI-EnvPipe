#!/bin/bash

mkdir -p nsight
OUTFILE='nsight/'$(date '+%Y-%m-%d_%H-%M-%S')
OUTFILE_SQL='nsight/'$(date '+%Y-%m-%d_%H-%M-%S')
nsys profile -t cuda,osrt,nvtx,cudnn,cublas --gpu-metrics-device=0,1,3 --output=${OUTFILE} $@
#nsys profile -t cuda,osrt,nvtx,cudnn,cublas --gpu-metrics-device=0 --output=${OUTFILE} --export=sqlite $@
