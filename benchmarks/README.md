# KISTI ULTRA SCALE AI EnvPipe Examples 

The example code is based on the [transpeeder project](https://github.com/HuangLK/transpeeder/tree/c57e1d63e98c74b085506c75492a56174a1dfa92). 

# llama-7B
python -m scripts.convert2ckpt --mp_world_size 4 \
    --model_name_or_path /path/to/llama-7b-hf \
    --output_dir /path/to/llama-7b-init-ckpt

# llama-30B
python -m scripts.convert2ckpt --mp_world_size 8 \
    --model_name_or_path /path/to/llama-30b-hf \
    --output_dir /path/to/llama-30b-init-ckpt