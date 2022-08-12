#!/bin/bash

echo "this script will install all the prerequisites and create a conda virtual environment named 'openmmlab'"
echo "there are specific steps for cpu-only or for gpu installation, which one do you want?"
echo "cpu/gpu"

read device

echo "be careful: you need conda installed, and be sure you run this script at the root of this git repo"
echo "continue ?"
echo "y/n"

read continue

if [ "$continue" != "y" ] || [ "$device" != "cpu" -a "$device" != "gpu" ]; then
  echo "install cancelled"
else
  conda create --name openmmlab python=3.8
  conda activate openmmlab

  if [ "$device" == "cpu"]; then
    conda install pytorch torchvision cpuonly -c pytorch
  else
    conda install pytorch torchvision -c pytorch
  fi

  pip install -U openmim
  mim install mmcv-full

  git clone https://github.com/open-mmlab/mmpose.git
  cd mmpose
  pip install -r requirements.txt
  pip install -v -e .

  cd ..
  git clone https://github.com/open-mmlab/mmdetection.git
  cd mmdetection
  pip install -v -e .

  cd ..

fi