#!/bin/sh -x

# Collect the energy consumption of the GPU
nvidia-smi -i 0 --loop-ms=500 --format=csv,noheader --query-gpu=timestamp,power.draw > nvidia_smi.txt &

# Get nvidia-smi's PID
nvidia_smi_PID=$!

# Trap: When the server is closed, terminate the nvidia-smi process
exit_script() {
    kill -9 "$nvidia_smi_PID"
}

trap exit_script SIGINT SIGTERM

perf stat -I 500 -e power/energy-pkg/,power/energy-ram/ -o perf.txt python3 dummy_serialisation/server.py | ts '[%Y-%m-%d %H:%M:%.S]'
