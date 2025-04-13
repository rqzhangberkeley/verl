#!/bin/bash

# Default hyperparameters
ACTOR_LR=${1:-1e-6}
CRITIC_LR=${2:-1e-5}
KL_COEF=${3:-0.001}
TRAIN_BATCH_SIZE=${4:-1024}
PPO_MINI_BATCH_SIZE=${5:-256}
PPO_MICRO_BATCH_SIZE=${6:-32}
TOTAL_EPOCHS=${7:-10}
MAX_RESPONSE_LENGTH=${8:-1024}
GPU_MEMORY_UTIL=${9:-0.7}
EXPERIMENT_NAME=${10:-"ppo_experiment_Math1.5B"}

echo "Running with hyperparameters:"
echo "Actor LR: $ACTOR_LR"
echo "Critic LR: $CRITIC_LR"
echo "KL Coefficient: $KL_COEF"
echo "Train Batch Size: $TRAIN_BATCH_SIZE"
echo "PPO Mini Batch Size: $PPO_MINI_BATCH_SIZE"
echo "PPO Micro Batch Size: $PPO_MICRO_BATCH_SIZE"
echo "Total Epochs: $TOTAL_EPOCHS"
echo "Max Response Length: $MAX_RESPONSE_LENGTH"
echo "GPU Memory Utilization: $GPU_MEMORY_UTIL"
echo "Experiment Name: $EXPERIMENT_NAME"

export CUDA_VISIBLE_DEVICES="0"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=$HOME/project/verl/data/math-base/train.parquet \
 data.val_files=$HOME/project/verl/data/math500-base/test.parquet \
 data.train_batch_size=$TRAIN_BATCH_SIZE \
 data.max_prompt_length=512 \
 data.max_response_length=$MAX_RESPONSE_LENGTH \
 data.filter_overlong_prompts=True \
 data.use_chat_template=False \
 actor_rollout_ref.model.path=Qwen/Qwen2.5-Math-1.5B \
 actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTIL \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
 critic.optim.lr=$CRITIC_LR \
 critic.model.path=Qwen/Qwen2.5-Math-1.5B \
 critic.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
 algorithm.kl_ctrl.kl_coef=$KL_COEF \
 trainer.logger=['console'] \
 +trainer.val_before_train=True \
 trainer.default_hdfs_dir=null \
 trainer.n_gpus_per_node=1 \
 trainer.nnodes=1 \
 trainer.save_freq=-1 \
 trainer.test_freq=2 \
 trainer.project_name=grpo \
 trainer.experiment_name=$EXPERIMENT_NAME \
 trainer.total_epochs=$TOTAL_EPOCHS 
 
# nohup ./run_ppo_debug.sh > ./logs/run_ppo_debug.log 2>&1 &