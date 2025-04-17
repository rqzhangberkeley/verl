#!/bin/bash

# Default hyperparameters
ACTOR_LR=${1:-1e-6}
CRITIC_LR=${2:-1e-5}
KL_COEF=${3:-0.001}
NUM_GENERATIONS_VALIDATION=${4:-32}
TRAIN_BATCH_SIZE=${5:-1024}
PPO_MINI_BATCH_SIZE=${6:-256}
PPO_MICRO_BATCH_SIZE_PER_GPU=${7:-32}
TOTAL_EPOCHS=${8:-10}
MAX_RESPONSE_LENGTH=${9:-1024}
GPU_MEMORY_UTIL=${10:-0.7}
TEST_FREQ=${11:-10}
N_GPUS=${12:-4}
COMPUTE_PROMPTS_VALUES=${13:-True}
EXPERIMENT_NAME=${14:-"ppo_test_Math1.5B"}

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
echo "Compute Prompts Values: $COMPUTE_PROMPTS_VALUES"
echo "Experiment Name: $EXPERIMENT_NAME"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=$HOME/project/verl/data/math-base/train.parquet \
 data.val_files=$HOME/project/verl/data/math500-base-subset50/test.parquet \
 data.train_batch_size=$TRAIN_BATCH_SIZE \
 data.max_prompt_length=512 \
 data.max_response_length=$MAX_RESPONSE_LENGTH \
 data.filter_overlong_prompts=True \
 data.use_chat_template=False \
 actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
 actor_rollout_ref.actor.use_dynamic_bsz=False \
 actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTIL \
 actor_rollout_ref.rollout.val_kwargs.n=$NUM_GENERATIONS_VALIDATION \
 actor_rollout_ref.rollout.compute_prompts_values=$COMPUTE_PROMPTS_VALUES \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
 critic.optim.lr=$CRITIC_LR \
 critic.model.path=Qwen/Qwen2.5-1.5B \
 critic.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
 algorithm.kl_ctrl.kl_coef=$KL_COEF \
 algorithm.use_doctor_grpo=True \
 trainer.logger=['console','wandb'] \
 +trainer.val_before_train=True\
 trainer.default_hdfs_dir=null \
 trainer.n_gpus_per_node=$N_GPUS \
 trainer.nnodes=1 \
 trainer.save_freq=-1 \
 trainer.test_freq=$TEST_FREQ \
 trainer.project_name=grpo \
 trainer.experiment_name=$EXPERIMENT_NAME \
 trainer.total_epochs=$TOTAL_EPOCHS 2>&1 | tee ./logs/${EXPERIMENT_NAME}.log
 
# trainer.project_name=grpo \
# trainer.experiment_name=test_run_7B_math_3epochs_1 2>&1 | tee verl_demo.log

# for large-scale dataset, filtering overlong prompts could be timeconsuming. You should disable this and set `truncation='left'