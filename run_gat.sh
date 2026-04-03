#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
echo "SCRIPT START"
/home/levos/miniconda3/envs/baikal/bin/python3 -u /home/levos/experiments/2026-03-31_gat_marathon/train_gat.py
