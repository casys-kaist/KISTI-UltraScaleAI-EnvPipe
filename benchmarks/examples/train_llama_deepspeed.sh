#!/bin/bash
# Train script.
set -eux


export MASTER_PORT=23857
export WORK_DIR=`pwd`

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

deepspeed --include localhost:0,1,2,3  --master_port ${MASTER_PORT} ${WORK_DIR}/train.py \
    --output_dir ${OUTPUT} \
    --init_ckpt  ${CHECKPOINT_PATH} \
    --data_path ${DATA_PATH} \
    --max_seq_len 8192 \
    --train_steps 1000 \
    --eval_steps 10 \
    --save_steps 200 \
    --log_steps 1 \
    --pipe_parallel_size 4 \
    --deepspeed_config ${WORK_DIR}/../configs/ds_config_zero1.json
