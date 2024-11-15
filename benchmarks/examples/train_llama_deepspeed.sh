#!/bin/bash
# Train script.
set -eux


export MASTER_PORT=23857
export WORK_DIR=`pwd`
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_IB_DISABLE=1

# Function to convert MB to bytes and set NCCL_BUFFSIZE
set_nccl_buffsize() {
    local mb=$1
    # Calculate bytes (1 MB = 1048576 bytes)
    local bytes=$((mb * 1048576))
    export NCCL_BUFFSIZE=$bytes
    echo "NCCL_BUFFSIZE set to $NCCL_BUFFSIZE bytes ($mb MB)"
}

# Pass the desired buffer size in MB as an argument, e.g., 32 for 32MB
set_nccl_buffsize 1024

ds_report

OUTPUT=${WORK_DIR}/output
if [ -d $OUTPUT ]; then
    # rm
    echo "${OUTPUT} exist."
else
    mkdir -p ${OUTPUT}
fi

# CHECKPOINT_PATH="meta-llama/Llama-3.2-1B"
CHECKPOINT_PATH="JackFram/llama-160m"
DATA_PATH="../dataset/json/gsm8k.json"

deepspeed --include localhost:0,1,3  --master_port ${MASTER_PORT} ${WORK_DIR}/train.py \
    --output_dir ${OUTPUT} \
    --init_ckpt  ${CHECKPOINT_PATH} \
    --data_path ${DATA_PATH} \
    --max_seq_len 8192 \
    --train_steps 10 \
    --eval_steps 10 \
    --save_steps 200 \
    --log_steps 10 \
    --pipe_parallel_size 3 \
    --deepspeed_config ${WORK_DIR}/../configs/ds_config.json
