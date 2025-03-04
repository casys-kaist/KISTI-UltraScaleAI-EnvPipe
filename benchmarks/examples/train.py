
import time
import random
import warnings
from dataclasses import dataclass, field
from typing import Optional, Literal

import torch
import transformers
import numpy as np
import deepspeed
from pynvml import *

from models.llama_pipeline_model import get_model
from models.patching import (
    replace_llama_attn_with_flash_attn,
    refine_rope,
)
from feeder import (
    make_prompt_dataloader,
    make_tokenized_dataloader,
)
from utils import jload
from utils import logger_rank0 as logger

warnings.filterwarnings("ignore")

@dataclass
class TrainerArguments:
    init_ckpt: str = field(default="llama-7B-init-test-ckpt")

    rank: int = field(default=None)
    local_rank: int = field(default=None)
    pipe_parallel_size: int = field(default=1)
    model_parallel_size: int = field(default=1)
    world_size: int = field(default=None)
    seed: int = field(default=42)
    deepspeed_config: Optional[str] = field(default=None)

    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    input_format: Literal['raw', 'tokenized'] = 'raw'
    mode: Literal['sft', 'pretrain', 'dialog'] = 'sft'
    num_workers: int = field(default=1)

    cache_dir: Optional[str] = field(default=None)
    output_dir: str = field(default="./output")
    max_seq_len: int = field(default=128)
    train_steps: int = field(default=10)
    eval_steps: int = field(default=100)
    save_steps: int = field(default=100)
    log_steps: int = field(default=1)

    resume_step: int = field(default=-1)
    resume_ckpt: str = field(default="llama-7B-init-test-ckpt")
    ntk : Optional[bool] = field(default=False)

def read_ds_config(config_path):
    config = jload(config_path)
    return config


def main():
    parser = transformers.HfArgumentParser(TrainerArguments)
    args, = parser.parse_args_into_dataclasses()

    # setup deepspeed and other stuff
    deepspeed.init_distributed(dist_backend="nccl")
    args.world_size = torch.distributed.get_world_size()

    ds_config = read_ds_config(args.deepspeed_config)
    args.num_workers = 2 * args.world_size // args.pipe_parallel_size // args.model_parallel_size
    args.batch_size = ds_config.get("train_micro_batch_size_per_gpu", 1)
    activation_checkpointing_config = ds_config.pop("activation_checkpointing", None)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    deepspeed.runtime.utils.set_random_seed(args.seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.init_ckpt,
        model_max_length=args.max_seq_len,
        padding_side="right",
        use_fast=False,
    )
    model_config = transformers.AutoConfig.from_pretrained(args.init_ckpt)

    if args.ntk:
        rope_scaling = {
            "type": "dynamic",
            "factor": 2,
        }
        model_config.rope_scaling = rope_scaling
        logger.info(f"Turn on dynamic rope for llama2")
        
    # pipeline model
    model = get_model(model_config, args, activation_checkpointing_config, partition_method="type:ParallelTransformerLayerPipe")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    engine, _, _, _ = deepspeed.initialize(
        args,
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
    )

    # dataset
    dataloader_maker = make_tokenized_dataloader if args.input_format == 'tokenized' else make_prompt_dataloader
    train_dataloader = dataloader_maker(tokenizer=tokenizer, data_args=args, engine=engine)

    # use `convert2ckpt.py`
    if args.resume_step < 0:
        engine.load_checkpoint(args.init_ckpt,
                            load_module_only=True,
                            load_optimizer_states=False,
                            load_lr_scheduler_states=False,
        )
    else:
        engine.load_checkpoint(args.resume_ckpt)
        
    # Profiling phase 
    while True:
        engine.train_batch(data_iter=train_dataloader)
        if not engine.energy_profiler.is_profiling:
            break
        
    # Reconfigure phase
    while True:
        engine.train_batch(data_iter=train_dataloader)
        if engine.execution_grid.finish_reconfigure():
            break

    device_count = nvmlDeviceGetCount()
    current_energy = [0] * device_count
    total_energy_consumption = [0] * device_count
    
    if args.local_rank == 0:
        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            current_energy[i] = nvmlDeviceGetTotalEnergyConsumption(handle)

        start_time = time.time()
    
    for _ in range(args.train_steps):
        engine.train_batch(data_iter=train_dataloader)
        
    torch.cuda.synchronize()
    
    if args.local_rank == 0:
        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            total_energy_consumption[i] = nvmlDeviceGetTotalEnergyConsumption(handle) - current_energy[i]
        
        throughput = (engine.train_batch_size() * args.train_steps) / (time.time() - start_time)   
    
        print("[RESULT]", round(throughput, 3), ",", round(sum(
            total_energy_consumption) / args.train_steps, 3))

if __name__ == "__main__":
    main()
