# KISTI UltraScaleAI EnvPipe

EnvPipe (Envelope + Pipeline Parallelism) is an energy-efficient DNN training framework designed to reduce energy consumption while maintaining minimal performance impact. This project aims to address the high energy demands and sustainability challenges associated with scaling large language models (LLMs). By leveraging slack time created by bubbles in pipeline parallelism, EnvPipe strategically schedules pipeline units and dynamically adjusts SM frequency, enabling energy savings without compromising training accuracy or hyperparameters.

### Enhancements in This Version
This improved implementation of EnvPipe builds upon the [original EnvPipe repository](https://github.com/casys-kaist/EnvPipe) with the following updates:
- **LLama Model Support**: Enhanced compatibility with the Llama model family.
- **DeepSpeed Upgrade**: Updated for compatibility with the latest DeepSpeed library (v0.15.4).
- **Huggingface Integration**: Refactored code to seamlessly support Huggingface models, incorporating updates based on the [Transpeeder](https://github.com/HuangLK/transpeeder) repository for Llama model compatibility. If a Huggingface model can run with DeepSpeed's pipeline parallelism, it is compatible with EnvPipe.
- **Code Refactoring**: Improved code structure for better compatibility and maintainability.
- **Improved P2P Communication**: Redesigned the activation and gradient transfer mechanism to ensure deadlock-free execution aligned with EnvPipe's scheduling method. This improvement resolves the reliance on increased `NCCL_BUFFSIZE` for non-blocking communication, which is not guaranteed as clarified [here](https://github.com/NVIDIA/nccl/issues/1252#issuecomment-2058458352).

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

## Usage

To run the EnvPipe training script, use the following options:

```bash
Usage: ./train_llama_deepspeed.sh [options]

Options:
  --type TYPE                Set ENVPIPE_TYPE (baseline, uniform, envelope). Required.
  --scheduling SCHEDULING    Set ENVPIPE_SCHEDULING (1f1b, ours). Required.
  --reconfig RECONFIGURE     Set ENVPIPE_RECONFIGURE (default, greedy, balanced). Required.
  --gpus GPUS                Specify GPU numbers (comma-separated, e.g., 0,1,3). Required.
  -h, --help                 Show this help message.
```

| **Parameter** | **Inputs** | **Explanation** |
|---|---|---|
| ENVPIPE_TYPE | baseline | Run all GPUs with maximum SM frequency. |
|  | uniform | Run all GPUs with optimal SM frequency that represents the minimum point in the energy valley curve. |
|  | envelope | Run pipeline units with optimal SM frequency that are inside the outer envelope. |
| ENVPIPE_SCHEDULING | 1f1b | 1F1B scheduling method. |
|  | ours | EnvPipe's scehduling method. |
| ENVPIPE_RECONFIGURE | default | SM frequencies of pipeline units on the critical path are not reconfigured. |
|  | greedy |  SM frequencies of pipeline units on the critical path are greedily reconfigured from the end of the critical path. |
|  | balanced | SM frequencies of pipeline units on the critical path are balanced as much as possible. |

## Example Command

Here’s an example of how to run the script:

```bash
./train_llama_deepspeed.sh --type envelope --scheduling ours --reconfig balanced --gpus 0,1,3
```

### Add New GPU Architecture

EnvPipe supports the following GPU architectures by default: **V100**, **RTX3090**, **A100**, and **A6000**. To add support for a new GPU architecture, you must configure the supported clock frequencies and granularity parameters in the code and update specific files.

#### Steps to Add a New GPU Architecture

1. **Define the GPU Architecture:**
   Add the new GPU architecture name and its clock frequency parameters in the configuration section.

  deepspeed/runtime/constants.py end of file. 

   ```python
   # Add your new GPU architecture name
   ENVPIPE_GPU_NEWARCH = 'newarch'

   # Define clock frequency parameters for the new GPU
   NEWARCH_SM_FREQ_FILTER_MAX = 1800  # Maximum profiled SM frequency (MHz)
   NEWARCH_SM_FREQ_FILTER_MIN = 900   # Minimum profiled SM frequency (MHz)
   NEWARCH_SM_FREQ_GRANULARITY = 90   # Granularity of profiling SM frequency (MHz)
   NEWARCH_RECONFIGURE_GRANULARITY = 30  # Minimum step size for reconfiguration (MHz)
   ```

2. **Update the Profiling Logic:**
   Update `EnvPipe/DeepSpeed/deepspeed/profiling/energy_profiler/profiler.py` to include the new GPU architecture and its parameters.

   ```python
   elif self.config["gpu"] == ENVPIPE_GPU_NEWARCH:
       sm_freq_filter_max = NEWARCH_SM_FREQ_FILTER_MAX
       sm_freq_filter_min = NEWARCH_SM_FREQ_FILTER_MIN
       sm_freq_granularity = NEWARCH_SM_FREQ_GRANULARITY
   ```

3. **Update the Reconfiguration Logic:**
   Modify `EnvPipe/DeepSpeed/deepspeed/runtime/pipe/reconfiguration.py` to incorporate the reconfiguration granularity for the new GPU.

   ```python
   elif self.config["gpu"] == ENVPIPE_GPU_NEWARCH:
       reconfigure_granularity = NEWARCH_RECONFIGURE_GRANULARITY
   ```

4. **Verify Clock Frequencies:**
   Use the provided script `benchmarks/examples/scripts/get_supported_clock_frequencies.py` to determine the supported clock frequencies for the new GPU architecture. This ensures compatibility with your hardware.

   ```bash
   python get_supported_clock_frequencies.py
   ```

## Additional Information

For more details about DeepSpeed, refer to the [original DeepSpeed README](./README_deepspeed.md).
