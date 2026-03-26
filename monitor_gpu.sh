#!/bin/bash
# Скрипт для фонового мониторинга GPU в CSV формат
OUT_FILE="gpu_monitor.csv"

# Записываем заголовок, если файл пустой
if [ ! -s "$OUT_FILE" ]; then
    echo "timestamp,gpu_id,name,utilization_gpu_pct,utilization_mem_pct,memory_used_mib,memory_total_mib,temperature_gpu_c,power_draw_w" > "$OUT_FILE"
fi

echo "Starting GPU monitoring to $OUT_FILE..."

while true; do
    nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | sed 's/, /,/g' >> "$OUT_FILE"
    sleep 30
done
