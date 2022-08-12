#!/bin/bash

echo "this script will install all the prerequisites and create a conda virtual environment named 'openmmlab'"
echo "be careful: you need conda installed, and be sure you run this script at the root of this git repo"
echo "continue ?"
echo "y/n"

read continue

if [ "$continue" != "y" ]; then
  echo "install cancelled"
else
  conda create --name openmmlab python=3.8 -y
  conda activate openmmlab
  conda install pytorch torchvision -c pytorch

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

  echo "and that should be done now."

fi