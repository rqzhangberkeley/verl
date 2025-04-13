#!/bin/bash

# Enable wandb logging if available
wandb login 363018e9dc8339fae726d3b48a839f262c457194

# Create logs directory if it doesn't exist
mkdir -p logs

# Function to check if GPUs are available (all 8 GPUs should be available)
function check_gpus_available() {
    # Check if any processes are using GPUs
    num_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    if [ "$num_processes" -eq "0" ]; then
        return 0  # GPUs are available
    else
        return 1  # GPUs are in use
    fi
}

# Function to send notification (can be customized to use email, slack, etc.)
function send_notification() {
    local subject="$1"
    local message="$2"
    echo "[NOTIFICATION] $subject: $message" | tee -a $MASTER_LOG
    # Uncomment and modify to enable email notifications
    # echo "$message" | mail -s "$subject" your.email@example.com
}

# Function to run a single PPO experiment with specific hyperparameters
function run_ppo_experiment() {
    local actor_lr="$1"
    local critic_lr="$2"
    local kl_coef="$3"
    local train_batch_size="$4"
    local ppo_mini_batch_size="$5"
    local ppo_micro_batch_size="$6"
    local total_epochs="$7"
    local max_response_length="$8"
    local gpu_mem_util="$9"
    local test_freq="${10}"
    local experiment_name="${11}"
    
    # Create a hyperparameter-based log filename
    local timestamp=$(date +%Y%m%d-%H%M%S)
    local param_log_name="ppo_actorlr${actor_lr//[.-]/_}_criticlr${critic_lr//[.-]/_}_kl${kl_coef//[.-]/_}_bsz${train_batch_size}_minibsz${ppo_mini_batch_size}_microbsz${ppo_micro_batch_size}_epochs${total_epochs}_maxrl${max_response_length}_testfreq${test_freq}_${timestamp}"
    local param_log="./logs/${param_log_name}.log"
    
    # Print experiment details
    echo "Starting experiment with:" | tee -a $MASTER_LOG
    echo "Actor learning rate: $actor_lr" | tee -a $MASTER_LOG
    echo "Critic learning rate: $critic_lr" | tee -a $MASTER_LOG
    echo "KL coefficient: $kl_coef" | tee -a $MASTER_LOG
    echo "Train batch size: $train_batch_size" | tee -a $MASTER_LOG
    echo "PPO mini batch size: $ppo_mini_batch_size" | tee -a $MASTER_LOG
    echo "PPO micro batch size: $ppo_micro_batch_size" | tee -a $MASTER_LOG
    echo "Total epochs: $total_epochs" | tee -a $MASTER_LOG
    echo "Max response length: $max_response_length" | tee -a $MASTER_LOG
    echo "GPU memory utilization: $gpu_mem_util" | tee -a $MASTER_LOG
    echo "Test frequency: $test_freq" | tee -a $MASTER_LOG
    echo "Experiment name: $experiment_name" | tee -a $MASTER_LOG
    echo "Log file: $param_log" | tee -a $MASTER_LOG
    
    # Initialize the parameter-specific log file
    {
        echo "===================================================="
        echo "PPO Training Run: $experiment_name"
        echo "Started at: $(date)"
        echo "===================================================="
        echo "Parameters:"
        echo "- Actor LR: $actor_lr"
        echo "- Critic LR: $critic_lr"
        echo "- KL Coefficient: $kl_coef"
        echo "- Train Batch Size: $train_batch_size"
        echo "- PPO Mini Batch Size: $ppo_mini_batch_size"
        echo "- PPO Micro Batch Size: $ppo_micro_batch_size"
        echo "- Total Epochs: $total_epochs"
        echo "- Max Response Length: $max_response_length"
        echo "- GPU Memory Utilization: $gpu_mem_util"
        echo "- Test Frequency: $test_freq"
        echo "===================================================="
        echo "TRAINING OUTPUT BELOW:"
        echo "===================================================="
    } > $param_log
    
    # Run the script and capture output
    ./run_ppo.sh "$actor_lr" "$critic_lr" "$kl_coef" "$train_batch_size" "$ppo_mini_batch_size" "$ppo_micro_batch_size" "$total_epochs" "$max_response_length" "$gpu_mem_util" "$test_freq" "$experiment_name" 2>&1 | tee -a $param_log
    
    local exit_code=${PIPESTATUS[0]}
    
    # Record completion status
    {
        echo "===================================================="
        echo "Training finished at: $(date)"
        echo "Exit code: $exit_code"
        echo "===================================================="
    } >> $param_log
    
    # Handle success or failure
    if [ $exit_code -eq 0 ]; then
        echo "Training completed successfully at $(date)" >> $param_log
        echo "Completed configuration: $experiment_name at $(date)" | tee -a $MASTER_LOG
        send_notification "Experiment Completed" "Successfully completed PPO experiment: $experiment_name"
    else
        echo "Training failed at $(date)" >> $param_log
        echo "Failed configuration: $experiment_name at $(date)" | tee -a $MASTER_LOG
        local error_context=$(tail -n 20 $param_log)
        send_notification "Experiment Failed" "PPO experiment failed: $experiment_name with exit code $exit_code"
    fi
    
    echo "-------------------------------------------------" | tee -a $MASTER_LOG
    return $exit_code
}

# Create a file to track master progress
MASTER_LOG="./logs/ppo_hyperparameter_sweep_$(date +%Y%m%d-%H%M%S).log"
echo "Starting PPO hyperparameter sweep at $(date)" | tee $MASTER_LOG
echo "==============================================" | tee -a $MASTER_LOG

# Define hyperparameter configurations
# Format: "actor_lr critic_lr kl_coef train_batch_size ppo_mini_batch_size ppo_micro_batch_size total_epochs max_response_length gpu_mem_util test_freq experiment_name"
CONFIGS=(
    # "1e-5 1e-5 0.0 1024 256 32 1000 3584 0.7 1 ppo_math1024_1000epochs"
    # "1e-5 1e-5 0.0 1024 1024 128 40 3584 0.7 1 ppo_math1024_40epochs_B1024_Bmini1024" # RZ: Do not set the micro-batch-size to 1024 as this will cause OOM
    # "1e-5 1e-5 0.0 256 256 32 40 3584 0.7 4 ppo_math1024_40epochs_B256_Bmini256" That will double the time for each epoch.
    # "1e-5 1e-5 0.0 1024 64 8 40 3584 0.7 1 ppo_math1024_40epochs_B1024_Bmini64" # more offline.
    # "1e-5 1e-5 0.0 64 64 8 40 3584 0.7 16 ppo_math1024_40epochs_B64_Bmini64" # Ecounter an error. Not sure why.
    # "1e-5 1e-5 0.0 256 64 8 40 3584 0.7 4 ppo_math1024_40epochs_B256_Bmini64" # RZ: This also encounter some error. I am not sure why.
)

# Create a directory for job status tracking
mkdir -p job_status

# Count of completed and failed jobs
completed_jobs=0
failed_jobs=0

# Loop through configurations
for i in "${!CONFIGS[@]}"; do
    config=(${CONFIGS[$i]})
    
    # Wait until GPUs are available
    while ! check_gpus_available; do
        echo "GPUs are currently in use. Waiting for 5 minutes before checking again..." | tee -a $MASTER_LOG
        sleep 300  # Wait for 5 minutes
    done
    
    echo "Starting job $((i+1))/${#CONFIGS[@]}: ${config[9]}" | tee -a $MASTER_LOG
    
    # Run the experiment
    run_ppo_experiment "${config[0]}" "${config[1]}" "${config[2]}" "${config[3]}" "${config[4]}" "${config[5]}" "${config[6]}" "${config[7]}" "${config[8]}" "${config[9]}" "${config[10]}"
    
    # Track job status
    if [ $? -eq 0 ]; then
        echo "JOB_STATUS: ${config[9]} - SUCCESS" > job_status/job_${i}_status.txt
        ((completed_jobs++))
    else
        echo "JOB_STATUS: ${config[9]} - FAILED" > job_status/job_${i}_status.txt
        ((failed_jobs++))
    fi
    
    echo "Completed job $((i+1))/${#CONFIGS[@]}" | tee -a $MASTER_LOG
done

# Summary
echo "==============================================" | tee -a $MASTER_LOG
echo "All jobs completed at $(date)" | tee -a $MASTER_LOG
echo "Summary: $completed_jobs succeeded, $failed_jobs failed" | tee -a $MASTER_LOG
echo "==============================================" | tee -a $MASTER_LOG

# Send final notification
send_notification "PPO Sweep Completed" "Completed PPO hyperparameter sweep: $completed_jobs succeeded, $failed_jobs failed" 

# nohup ./launch_ppo.sh > ./logs/launch_ppo_math.log 2>&1 &