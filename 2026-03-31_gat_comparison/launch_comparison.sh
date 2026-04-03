#!/bin/bash
# Останавливаем старые, если вдруг есть
pkill -f train_deep.py
pkill -f train_wide.py
sleep 2

# Запуск Deep на GPU 0
nohup /home/levos/miniconda3/envs/baikal/bin/python3 /home/levos/experiments/2026-03-31_gat_comparison/train_deep.py > /home/levos/experiments/2026-03-31_gat_comparison/deep.out 2>&1 &
echo "Deep launched with PID $!"

sleep 5

# Запуск Wide на GPU 0
nohup /home/levos/miniconda3/envs/baikal/bin/python3 /home/levos/experiments/2026-03-31_gat_comparison/train_wide.py > /home/levos/experiments/2026-03-31_gat_comparison/wide.out 2>&1 &
echo "Wide launched with PID $!"
