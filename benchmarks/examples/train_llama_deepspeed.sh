#!/bin/bash
# Train script with EnvPipe parameter switching.
set -eux

# Variables to store command-line inputs (no default values)
ENVPIPE_TYPE=""
ENVPIPE_SCHEDULING=""
ENVPIPE_RECONFIGURE=""
GPUS=""

# Display help message
function show_help {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --type TYPE                Set ENVPIPE_TYPE (baseline, uniform, envelope). Required."
    echo "  --scheduling SCHEDULING    Set ENVPIPE_SCHEDULING (1f1b, ours). Required."
    echo "  --reconfig RECONFIGURE     Set ENVPIPE_RECONFIGURE (default, greedy, balanced). Required."
    echo "  --gpus GPUS                Specify GPU numbers (comma-separated, e.g., 0,1,3). Required."
    echo "  -h, --help                 Show this help message."
    exit 0
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --type)
            ENVPIPE_TYPE="$2"
            shift 2
            ;;
        --scheduling)
            ENVPIPE_SCHEDULING="$2"
            shift 2
            ;;
        --reconfig)
            ENVPIPE_RECONFIGURE="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown argument: $1"
            show_help
            ;;
    esac
done

# Check for missing required inputs
if [ -z "$ENVPIPE_TYPE" ] || [ -z "$ENVPIPE_SCHEDULING" ] || [ -z "$ENVPIPE_RECONFIGURE" ] || [ -z "$GPUS" ]; then
    echo "Error: Missing required arguments."
    show_help
fi

# Calculate pipe_parallel_size based on the number of GPUs
IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
PIPE_PARALLEL_SIZE=${#GPU_ARRAY[@]}

export MASTER_PORT=23857
export WORK_DIR=$(pwd)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_IB_DISABLE=1

# Function to convert MB to bytes and set NCCL_BUFFSIZE
set_nccl_buffsize() {
    local mb=$1
    local bytes=$((mb * 1048576))
    export NCCL_BUFFSIZE=$bytes
}

# Pass the desired buffer size in MB as an argument, e.g., 32 for 32MB
set_nccl_buffsize 32

ds_report

OUTPUT=${WORK_DIR}/output
if [ -d "$OUTPUT" ]; then
    echo "${OUTPUT} exists."
else
    mkdir -p "${OUTPUT}"
fi

#CHECKPOINT_PATH="meta-llama/Llama-3.2-1B"
CHECKPOINT_PATH="JackFram/llama-160m"
DATA_PATH="../dataset/json/gsm8k.json"

# Create a copy of the config file and modify it
CONFIG_TEMPLATE="${WORK_DIR}/../configs/ds_config_template.json"
CONFIG_FILE="${WORK_DIR}/../configs/ds_config_${ENVPIPE_TYPE}_${ENVPIPE_SCHEDULING}_${ENVPIPE_RECONFIGURE}.json"

cp "$CONFIG_TEMPLATE" "$CONFIG_FILE"

sed -i "s/ENVPIPE_TYPE/${ENVPIPE_TYPE}/g" "$CONFIG_FILE"
sed -i "s/ENVPIPE_SCHEDULING/${ENVPIPE_SCHEDULING}/g" "$CONFIG_FILE"
sed -i "s/ENVPIPE_RECONFIG/${ENVPIPE_RECONFIGURE}/g" "$CONFIG_FILE"

# Run the training script with DeepSpeed
deepspeed --include "localhost:${GPUS}" --master_port ${MASTER_PORT} ${WORK_DIR}/train.py \
    --output_dir ${OUTPUT} \
    --init_ckpt ${CHECKPOINT_PATH} \
    --data_path ${DATA_PATH} \
    --max_seq_len 8192 \
    --train_steps 10 \
    --eval_steps 10 \
    --save_steps 200 \
    --log_steps 10 \
    --pipe_parallel_size ${PIPE_PARALLEL_SIZE} \
    --deepspeed_config "$CONFIG_FILE"
