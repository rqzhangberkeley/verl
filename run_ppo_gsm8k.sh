#!/bin/bash

wandb login 363018e9dc8339fae726d3b48a839f262c457194
export CUDA_VISIBLE_DEVICES="4,5,6,7"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=$HOME/project/verl/data/math/train.parquet \
 data.val_files=$HOME/project/verl/data/math/test.parquet \
 data.train_batch_size=256 \
 data.max_prompt_length=1024 \
 data.max_response_length=2048 \
 data.filter_overlong_prompts=True \
 actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 critic.optim.lr=1e-5 \
 critic.model.path=Qwen/Qwen2.5-7B-Instruct \
 critic.ppo_micro_batch_size_per_gpu=4 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=['console','wandb'] \
 +trainer.val_before_train=True \
 trainer.default_hdfs_dir=null \
 trainer.n_gpus_per_node=4 \
 trainer.nnodes=1 \
 trainer.save_freq=400 \
 trainer.test_freq=10 \
 trainer.total_epochs=15 \
 trainer.project_name=grpo \
 trainer.experiment_name=test_run_math_15epochs_7Bit 2>&1 | tee verl_demo.log

# for large-scale dataset, filtering overlong prompts could be timeconsuming. You should disable this and set `truncation='left'

# The key metric val/test_score/openai/gsm8k is computed every trainer.test_freq steps 
# nohup ./run_ppo_gsm8k.sh > ./logs/run_ppo_gsm8k.log 2>&1 &
# nohup ./run_ppo_gsm8k.sh > ./logs/run_ppo_math_7B.log 2>&1 &