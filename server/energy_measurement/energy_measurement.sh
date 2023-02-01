#!/bin/sh -x

# bash energy_measurement.sh 400

# Set to user specified value or to default value 500ms for count_interval_ms if undefined by user
count_interval_ms=${1:-500}

# Collect the energy consumption of the GPU
nvidia-smi -i 0 --loop-ms=$count_interval_ms --format=csv,noheader --query-gpu=timestamp,power.draw > energy_measurement/out/nvidia_smi.txt &

# Get nvidia-smi's PID
nvidia_smi_PID=$!

# Trap: When the server is closed, terminate the nvidia-smi process
exit_script() {
    kill -9 "$nvidia_smi_PID"
}

trap exit_script SIGINT SIGTERM

# ts has not use at the moment
perf stat -I $count_interval_ms -e power/energy-pkg/,power/energy-ram/ -o energy_measurement/out/perf.txt -x \; python3 server.py | ts '[%Y-%m-%d %H:%M:%.S]'
