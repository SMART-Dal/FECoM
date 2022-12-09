#!/bin/sh -x

# Set default value for count_interval_ms if undefined by user
count_interval_ms=${1:-500}

# Collect the energy consumption of the GPU
nvidia-smi -i 0 --loop-ms=$count_interval_ms --format=csv,noheader --query-gpu=timestamp,power.draw > nvidia_smi.txt &

# Get nvidia-smi's PID
nvidia_smi_PID=$!

# Trap: When the server is closed, terminate the nvidia-smi process
exit_script() {
    kill -9 "$nvidia_smi_PID"
}

trap exit_script SIGINT SIGTERM

perf stat -I $count_interval_ms -e power/energy-pkg/,power/energy-ram/ -o perf.txt -x \;
# python3 server/server.py | ts '[%Y-%m-%d %H:%M:%.S]'
