CUDA_VISIBLE_DEVICES=0,1 \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
accelerate launch --config_file deepspeed.yaml trl_facts_training.py