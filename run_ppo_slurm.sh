#!/bin/bash
#SBATCH --job-name=test_ppo_compute_prompts_values       # Job name
#SBATCH --output=./logs/verl_ppo_%j.out  # Output file (%j will be replaced by job ID)
#SBATCH --error=./logs/verl_ppo_%j.err   # Error file
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks-per-node=1       # Number of tasks per node
#SBATCH --cpus-per-task=256         # Number of CPU cores per task
#SBATCH --gres=gpu:4              # Number of GPUs (4 GPUs per node)
#SBATCH --mem-per-gpu=100G                # Memory per node
#SBATCH --time=10:00:00           # Time limit (24 hours)
#SBATCH --account=bdwy-dtai-gh    # Account name (adjust to your account)
#SBATCH --mail-user=rqzhang@berkeley.edu  # Email address to receive notifications
#SBATCH --mail-type=BEGIN,END,FAIL         # Send email at begin, end, or fail of job


# Load modules (Everytime before running any code on DeltaAI, please run this command)
module default
module load cuda/12.6.1
module load gcc/11.4.0
conda activate verl
wandb login 363018e9dc8339fae726d3b48a839f262c457194

# Default hyperparameters
ACTOR_LR=${1:-1e-6}
CRITIC_LR=${2:-1e-5}
KL_COEF=${3:-0.001}
NUM_GENERATIONS_VALIDATION=${4:-32}
TRAIN_BATCH_SIZE=${5:-1024}
PPO_MINI_BATCH_SIZE=${6:-256}
PPO_MICRO_BATCH_SIZE_PER_GPU=${7:-8}
TOTAL_EPOCHS=${8:-10}
MAX_RESPONSE_LENGTH=${9:-8192}
GPU_MEMORY_UTIL=${10:-0.7}
TEST_FREQ=${11:-3}
N_GPUS=${12:-4}
COMPUTE_PROMPTS_VALUES=${13:-True}
EXPERIMENT_NAME=${14:-"ppo_test_Math1.5B_tok8k"}

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
 data.train_files=./data/math-base/train.parquet \
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
 actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
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
