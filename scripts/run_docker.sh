#!/bin/bash
workspace=$(dirname $(dirname $(realpath $0)))

docker_tag="nvcr.io/nvidia/pytorch:23.10-py3"

# Function to find an available port, starting from a given port
find_available_port() {
  local start_port=$1
  while :; do
    # Check if the port is in use by any process (using lsof) or Docker container
    if ! lsof -i :"$start_port" &>/dev/null && ! docker ps --format '{{.Ports}}' | grep -q ":$start_port->"; then
      echo "$start_port"
      return
    fi
    start_port=$((start_port + 1))
  done
}

# Set the initial port number and find an available port
initial_port=8000
port=$(find_available_port "$initial_port")

# Run Docker container with the selected tag and available port
# Should run docker with root permission for NVML to set GPU clock
docker run --gpus all -it --rm --privileged \
--ipc host \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
--shm-size=12g -p "$port:$port" \
-v "$workspace":"$workspace" \
-w "$workspace" \
"$docker_tag"
