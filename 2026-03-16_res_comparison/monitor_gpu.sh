#!/bin/bash
LOG_FILE="2026-03-16_res_comparison/gpu_monitor.log"
echo "Timestamp, GPU_ID, Temp, Power, Usage, Mem_Used" > $LOG_FILE
while true; do
    nvidia-smi --query-gpu=timestamp,index,temperature.gpu,power.draw,utilization.gpu,memory.used --format=csv,noheader,nounits >> $LOG_FILE
    sleep 30
done
