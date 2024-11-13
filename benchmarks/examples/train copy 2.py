import time
import random
import warnings
from dataclasses import dataclass, field
from typing import Optional, Literal
import argparse
import sys 

import inspect
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding
import numpy as np
import deepspeed
from deepspeed.pipe import PipelineModule

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
    model_name: str = field(default="facebook/opt-125m")

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
    train_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    save_steps: int = field(default=100)
    log_steps: int = field(default=1)

def read_ds_config(config_path):
    config = jload(config_path)
    return config

class EmbeddingPipe(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, args):        
        input_ids, position_ids, attention_mask = args
        inputs_embeds = self.module(input_ids)
        return (inputs_embeds, position_ids, attention_mask)
    
class TransformerLayerPipe(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        
    def forward(self, args):
        hidden_states, position_ids, attention_mask = args
        outputs = self.module(hidden_states, attention_mask, position_ids)
        
        return (outputs[0], position_ids, attention_mask)

class LlamaRMSNormPipe(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, args):
        print(args)
        hidden_states, *_ = args
        hidden_states = self.module(hidden_states)
        return (hidden_states,)
    
# class LlamaRotaryEmbeddingPipe(torch.nn.Module):
#     def __init__(self, module):
#         super().__init__()
#         self.module = module
    
#     def forward(self, args):
#         x, position_ids = args
#         x = self.module(x, position_ids)
#         return (x, position_ids)
        

def flatten_and_replace_layers(model, whitelist=None, replacement_map=None):
    layers = []
    whitelist = whitelist or []
    replacement_map = replacement_map or {}

    for name, module in model.named_children():
        module_type_name = type(module).__name__

        # Preserve the layer as-is if it's in the whitelist
        if module_type_name in whitelist:
            print(f"Preserving {module_type_name} without further flattening.")
            layers.append(module)
        
        # Replace the layer if it’s in the replacement_map
        elif module_type_name in replacement_map:
            replacement_class = replacement_map[module_type_name]
            print(f"Replacing {module_type_name} with {replacement_class.__name__}")
            
            # Wrap the original module with the replacement class
            new_module = replacement_class(module)
            layers.append(new_module)

        # Flatten further if the layer is not in the whitelist or replacement_map
        else:
            if list(module.children()):
                layers.extend(flatten_and_replace_layers(module, whitelist, replacement_map))
            else:
                layers.append(module)
    return layers

def main():
    parser = transformers.HfArgumentParser(TrainerArguments)
    args, = parser.parse_args_into_dataclasses()
    
    deepspeed.init_distributed(dist_backend="nccl")
    args.world_size = torch.distributed.get_world_size()
    ds_config = read_ds_config(args.deepspeed_config)
    args.batch_size = ds_config.get("train_micro_batch_size_per_gpu", 1)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    deepspeed.runtime.utils.set_random_seed(args.seed)

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name,
                                              model_max_length=args.max_seq_len,)
    tokenizer.pad_token = tokenizer.eos_token

    # Define replacement mapping
    replacement_map = {
        "Embedding": EmbeddingPipe,
        "LlamaDecoderLayer": TransformerLayerPipe,
        "LlamaRMSNorm": LlamaRMSNormPipe,
        # "LlamaRotaryEmbedding": LlamaRotaryEmbeddingPipe,
    }

    # Flatten the model while replacing and preserving specified layers
    flattened_layers = flatten_and_replace_layers(
        model,
        replacement_map=replacement_map
    )

    model = PipelineModule(
        layers=flattened_layers,
        loss_fn=torch.nn.CrossEntropyLoss(),
        num_stages=args.pipe_parallel_size,
        partition_method="type:TransformerLayerPipe",
    )
    
    engine, _, _, _ = deepspeed.initialize(
        args,
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
    )

    # dataset
    train_dataloader = make_prompt_dataloader(tokenizer=tokenizer, data_args=args, engine=engine)

    start = time.time()
    for step in range(1, args.train_steps + 1):
        loss = engine.train_batch(data_iter=train_dataloader)
        if args.local_rank == 0:
            if step % args.log_steps == 0:
                now = time.time()
                avg_time = (now-start) / args.log_steps
                logger.info(f"Step={step:>6}, loss={loss.item():.4f}, {avg_time:.2f} it/s")
                start = now
                
        if step % args.save_steps == 0:
            logger.info(f"Saving at step {step}")
            engine.save_checkpoint(args.output_dir)


if __name__ == "__main__":
    main()
