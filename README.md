# KISTI UltraScaleAI EnvPipe

EnvPipe (Envelope + Pipeline Parallelism) is an energy-efficient DNN training framework designed to reduce energy consumption while maintaining minimal performance impact. This project aims to address the high energy demands and sustainability challenges associated with scaling large language models (LLMs). By leveraging slack time created by bubbles in pipeline parallelism, EnvPipe strategically schedules pipeline units and dynamically adjusts SM frequency, enabling energy savings without compromising training accuracy or hyperparameters.

### Enhancements in This Version
This improved implementation of EnvPipe builds upon the [original EnvPipe repository](https://github.com/casys-kaist/EnvPipe) with the following updates:
- **LLama Model Support**: Enhanced compatibility with the Llama model family.
- **DeepSpeed Upgrade**: Updated for compatibility with the latest DeepSpeed library (v0.15.4).
- **Huggingface Integration**: Refactored code to support Huggingface models seamlessly.
- **Code Refactoring**: Improved code structure for better compatibility and maintainability.
- **Improved P2P Communication**: Redesigned the activation and gradient transfer mechanism to ensure deadlock-free execution aligned with EnvPipe's scheduling method. This improvement resolves the reliance on increased `NCCL_BUFFSIZE` for non-blocking communication, which is not guaranteed as clarified [here](https://github.com/NVIDIA/nccl/issues/1252#issuecomment-2058458352).

For non-blocking to be guaranteed, we'd need to provide something like a MPI_Bsend. 
## Getting Started

### Run the Docker Environment
To set up the environment:
```bash
scripts/run_docker.sh
```

### Install DeepSpeed Library
Once inside the Docker container, install DeepSpeed using one of the following methods:
- Editable mode (recommended for development):
  ```bash
  pip install -e .
  ```
- Normal mode (for production):
  ```bash
  pip install .
  ```

## Running Benchmarks

Navigate to the benchmarks directory and use the provided script to train a model with DeepSpeed and EnvPipe:
```bash
benchmarks/examples/train_llama_deepspeed.sh
```

### Usage
```bash
Usage: train_llama_deepspeed.sh [options]

Options:
  --type TYPE                Set ENVPIPE_TYPE (baseline, uniform, envelope). Default: baseline
  --scheduling SCHEDULING    Set ENVPIPE_SCHEDULING (1f1b, ours). Default: 1f1b
  --reconfig RECONFIGURE     Set ENVPIPE_RECONFIGURE (default, greedy, balanced). Default: default
  -h, --help                 Show this help message.
```

## Additional Information

For more details about DeepSpeed, refer to the [original DeepSpeed README](./README_deepspeed.md).
