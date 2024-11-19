# KISTI UltraScaleAI EnvPipe

EnvPipe (Envelope + Pipeline Parallelism) is an energy-saving DNN training framework designed to maximize energy efficiency while maintaining negligible performance slowdown. It leverages slack time created by bubbles in pipeline parallelism, scheduling pipeline units to place these bubbles strategically. By stretching the execution time of pipeline units through reduced SM frequency, EnvPipe achieves energy savings without compromising the original training accuracy or altering hyperparameters.

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
