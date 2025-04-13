#!/bin/bash

# Default hyperparameters
NUM_GENERATIONS=${1:-32}
ACTOR_LR=${2:-3e-6}
KL_COEF=${3:-0.001}
TRAIN_BATCH_SIZE=${4:-8} # number of prompts per RL step
PPO_MINI_BATCH_SIZE=${5:-8} # number of prompts per gradient steps
PPO_MICRO_BATCH_SIZE=${6:-4} # number of completions per gradient accumulationsteps per GPU
TOTAL_EPOCHS=${7:-1}
MAX_RESPONSE_LENGTH=${8:-3584}
GPU_MEMORY_UTIL=${9:-0.7}
TEST_FREQ=${10:-5}
EXPERIMENT_NAME=${11:-"rloo_experiment_test_LIMR"}

echo "Running with hyperparameters:"
echo "Number of Generations: $NUM_GENERATIONS"
echo "Actor LR: $ACTOR_LR"
echo "KL Coefficient: $KL_COEF"
echo "Train Batch Size: $TRAIN_BATCH_SIZE"
echo "PPO Mini Batch Size: $PPO_MINI_BATCH_SIZE"
echo "PPO Micro Batch Size: $PPO_MICRO_BATCH_SIZE"
echo "Total Epochs: $TOTAL_EPOCHS"
echo "Max Response Length: $MAX_RESPONSE_LENGTH"
echo "GPU Memory Utilization: $GPU_MEMORY_UTIL"
echo "Test Frequency: $TEST_FREQ"
echo "Experiment Name: $EXPERIMENT_NAME"

wandb login 363018e9dc8339fae726d3b48a839f262c457194
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export VLLM_ATTENTION_BACKEND=XFORMERS

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=rloo \
    data.train_files=$HOME/project/verl/data/LIMR/train.parquet \
    data.val_files=$HOME/project/verl/data/math500/test.parquet \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=512 \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.use_chat_template=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-Math-1.5B \
    actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTIL \
    actor_rollout_ref.rollout.n=$NUM_GENERATIONS \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=$KL_COEF \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    +trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.project_name=grpo \
    trainer.experiment_name=$EXPERIMENT_NAME 2>&1 | tee ./logs/${EXPERIMENT_NAME}.log

# nohup ./run_rloo.sh > ./logs/run_rloo_test_LIMR.log 2>&1 &