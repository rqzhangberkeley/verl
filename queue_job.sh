#!/bin/bash

# Queue a job to run when GPUs are available
# Usage: ./queue_job.sh path_to_your_script.sh [arg1 arg2 ...]

# Check if a script path was provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 path_to_your_script.sh [arg1 arg2 ...]"
    exit 1
fi

# Get the script path and its arguments
SCRIPT_PATH="$1"
shift
SCRIPT_ARGS="$@"

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate a unique job ID
JOB_ID=$(date +%Y%m%d%H%M%S)_$(echo $RANDOM | md5sum | head -c 8)
LOCK_FILE="./job_queue_lock"
QUEUE_DIR="./job_queue"
RUNNING_FILE="./job_queue_running"
LOG_FILE="./logs/job_queue_${JOB_ID}.log"

# Create queue directory if it doesn't exist
mkdir -p "$QUEUE_DIR"

# Register the job in the queue
echo "#!/bin/bash" > "$QUEUE_DIR/$JOB_ID"
echo "# Job queued at $(date)" >> "$QUEUE_DIR/$JOB_ID"
echo "$SCRIPT_PATH $SCRIPT_ARGS" >> "$QUEUE_DIR/$JOB_ID"
chmod +x "$QUEUE_DIR/$JOB_ID"

echo "Job $JOB_ID queued. It will run when GPUs are available and previous jobs have completed."
echo "Logs will be written to $LOG_FILE"

# Check if job processor is already running
if [ -f "$RUNNING_FILE" ] && ps -p "$(cat "$RUNNING_FILE")" > /dev/null; then
    echo "Job processor is already running. Your job has been queued and will run in sequence."
    exit 0
fi

# Start the job processor in background
{
    # Mark processor as running
    echo $$ > "$RUNNING_FILE"
    
    # Function to check if GPUs are available
    function check_gpus_available() {
        # Check if any processes are using GPUs
        num_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
        if [ "$num_processes" -eq "0" ]; then
            return 0  # GPUs are available
        else
            return 1  # GPUs are in use
        fi
    }
    
    # Function to get the next job from the queue
    function get_next_job() {
        # Find the oldest job file
        ls -t "$QUEUE_DIR" | tail -1
    }
    
    # Process jobs from the queue
    while true; do
        # Wait until GPUs are available
        while ! check_gpus_available; do
            echo "[$(date)] GPUs are currently in use. Waiting for 5 minutes before checking again..." | tee -a "$LOG_FILE"
            sleep 300  # Wait for 5 minutes
        done
        
        # Get the next job
        NEXT_JOB=$(get_next_job)
        
        # If there are no more jobs, exit
        if [ -z "$NEXT_JOB" ]; then
            echo "[$(date)] No more jobs in the queue. Exiting." | tee -a "$LOG_FILE"
            rm -f "$RUNNING_FILE"
            exit 0
        fi
        
        # Run the job
        echo "[$(date)] Starting job $NEXT_JOB" | tee -a "$LOG_FILE"
        
        # Execute the job and capture its output
        JOB_LOG_FILE="./logs/job_${NEXT_JOB}.log"
        {
            echo "===================================================="
            echo "Job: $NEXT_JOB"
            echo "Command: $(cat "$QUEUE_DIR/$NEXT_JOB" | grep -v "^#")"
            echo "Started at: $(date)"
            echo "===================================================="
            "$QUEUE_DIR/$NEXT_JOB"
            JOB_EXIT_CODE=$?
            echo "===================================================="
            echo "Job finished at: $(date)"
            echo "Exit code: $JOB_EXIT_CODE"
            echo "===================================================="
        } | tee "$JOB_LOG_FILE"
        
        # Remove the job from the queue
        rm "$QUEUE_DIR/$NEXT_JOB"
        
        echo "[$(date)] Job $NEXT_JOB completed with exit code $JOB_EXIT_CODE" | tee -a "$LOG_FILE"
    done
} &

echo "Job processor started with PID $!" 