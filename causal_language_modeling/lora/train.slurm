#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=00-01:00:00      # Upper time limit for the job (DD-HH:MM:SS)
#SBATCH --nodes=1               # Allocate 1 nodes for the job
#SBATCH --cpus-per-task=32       # Allocate 40 cores
#SBATCH --mem=500G              # Allocate 200 GB of memory per node
#SBATCH --gres=gpu:8            # Allocate 8 GPUs per node
#SBATCH --constraint="v100|a100" # Only run on V100 or A100 GPUs
#SBATCH --job-name="OPT-125M"   # Name of the job
#SBATCH --output=train.out
#SBATCH --mail-user=andr3.storhaug@gmail.com
#SBATCH --mail-type=ALL


cd $SLURM_SUBMIT_DIR

echo "-----------------------------------------------------"
echo "We are running from this directory: $SLURM_SUBMIT_DIR"
echo "The name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"
echo "Total of $SLURM_GPUS_ON_NODE GPUs"
echo "-----------------------------------------------------"

export TMPDIR=~/tmp
export TORCH_EXTENSIONS_DIR=~/tmp/torch_extensions

# Load modules
module purge
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/11.7.0

# Activate environment
source .venv/bin/activate

# Set up environment
export WANDB_LOG_MODEL=false
export WANDB_WATCH=true # BF16 might causes problems
export WANDB_PROJECT=methods2test_small-fm+fc+c+m+f+t+tc
export WANDB_RESUME=allow
#export WANDB_RUN_ID=UNIQUE_ID

# Start training
accelerate launch run_clm_lora.py \
    --model_name_or_path facebook/opt-125m \
    --dataset_name andstor/methods2test_small \
    --dataset_config_name fm+fc+c+m+f+t+tc \
    --text_column_names "source, target" \
    --preprocessing_num_workers 10 \
    \
    --report_to all \
    --logging_first_step --logging_steps 1 \
    --load_best_model_at_end \
    --output_dir .data/output \
    --logging_dir ./logs \
    --log_preditions \
    --log_predition_samples 10 \
    \
    --block_size 2048 \
    \
    --do_train \
    --save_strategy steps \
    --save_total_limit 3 \
    --save_steps 250 \
    --warmup_ratio=0.1 \
    --per_device_train_batch_size 1 \
    \
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps 250 \
    --max_eval_samples 256 \
    --per_device_eval_batch_size 1 \
    \
    --adapter_name default \
    --rank 32 \
    --lora_alpha 64 \
    --target_modules "q_proj, v_proj" \
    --lora_dropout 0.1 \
    --bias none \
    --learning_rate 1e-4
echo "-------------------------- DONE ---------------------------"
