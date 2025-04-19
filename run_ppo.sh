#!/bin/bash

wandb login 363018e9dc8339fae726d3b48a839f262c457194

# Default hyperparameters
ACTOR_LR=${1:-5e-7}
CRITIC_LR=${2:-5e-6}
KL_COEF=${3:-0.001}
NUM_GENERATIONS_VALIDATION=${4:-16}
TRAIN_BATCH_SIZE=${5:-512}
PPO_MINI_BATCH_SIZE=${6:-128}
PPO_MICRO_BATCH_SIZE_PER_GPU=${7:-16}
TOTAL_EPOCHS=${8:-10}
MAX_RESPONSE_LENGTH=${9:-8192}
GPU_MEMORY_UTIL=${10:-0.7}
TEST_FREQ=${11:-5}
N_GPUS=${12:-4}
TOTAL_TRAINING_STEPS=${13:-500}
COMPUTE_PROMPTS_VALUES=${14:-True}
EXPERIMENT_NAME=${15:-"ppo1.5B_math12k_tok8k"}

echo "Running with hyperparameters:"
echo "Actor LR: $ACTOR_LR"
echo "Critic LR: $CRITIC_LR"
echo "KL Coefficient: $KL_COEF"
echo "Number of Generations Validation: $NUM_GENERATIONS_VALIDATION"
echo "Train Batch Size: $TRAIN_BATCH_SIZE"
echo "PPO Mini Batch Size: $PPO_MINI_BATCH_SIZE"
echo "PPO Micro Batch Size: $PPO_MICRO_BATCH_SIZE_PER_GPU"
echo "Total Epochs: $TOTAL_EPOCHS"
echo "Max Response Length: $MAX_RESPONSE_LENGTH"
echo "GPU Memory Utilization: $GPU_MEMORY_UTIL"
echo "Test Frequency: $TEST_FREQ"
echo "Number of GPUs: $N_GPUS"
echo "Total Training Steps: $TOTAL_TRAINING_STEPS"
echo "Compute Prompts Values: $COMPUTE_PROMPTS_VALUES"
echo "Experiment Name: $EXPERIMENT_NAME"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=./data/math500-base/train.parquet \
 data.val_files=./data/math500-base/test.parquet \
 data.train_batch_size=$TRAIN_BATCH_SIZE \
 data.max_prompt_length=512 \
 data.max_response_length=$MAX_RESPONSE_LENGTH \
 data.filter_overlong_prompts=True \
 data.use_chat_template=False \
 actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
 actor_rollout_ref.actor.use_dynamic_bsz=False \
 actor_rollout_ref.actor.use_doctor_grpo=True \
 actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.max_num_batched_tokens=9216 \
 actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTIL \
 actor_rollout_ref.rollout.val_kwargs.n=$NUM_GENERATIONS_VALIDATION \
 actor_rollout_ref.rollout.compute_prompts_values=$COMPUTE_PROMPTS_VALUES \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
 critic.optim.lr=$CRITIC_LR \
 critic.model.path=Qwen/Qwen2.5-1.5B \
 critic.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
 algorithm.kl_ctrl.kl_coef=$KL_COEF \
 trainer.logger=['console','wandb'] \
 +trainer.val_before_train=True\
 trainer.default_hdfs_dir=null \
 trainer.n_gpus_per_node=$N_GPUS \
 trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
 trainer.nnodes=1 \
 trainer.save_freq=-1 \
 trainer.test_freq=$TEST_FREQ \
 trainer.project_name=grpo \
 trainer.experiment_name=$EXPERIMENT_NAME \
 trainer.total_epochs=$TOTAL_EPOCHS 2>&1 | tee ./logs/${EXPERIMENT_NAME}.log
