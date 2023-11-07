#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=00-01:00:00      # Upper time limit for the job (DD-HH:MM:SS)
#SBATCH --nodes=1               # Allocate 1 nodes for the job
#SBATCH --cpus-per-task=8       # Allocate 40 cores
#SBATCH --mem=200G              # Allocate 200 GB of memory per node
#SBATCH --gres=gpu:2            # Allocate 4 GPUs per node
#SBATCH --job-name="GPT-J-6B"
#SBATCH --output=train_test.out
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

# Generate hostfile
truncate -s 0 hostfile
scontrol show hostname $SLURM_NODELIST | while read line; do echo ${line}"  slots="${SLURM_GPUS_ON_NODE} >> hostfile; done

# Set up environment
export WANDB_LOG_MODEL=false
export WANDB_WATCH=true # BF16 causes problems
export WANDB_PROJECT=smart-contracts-pt
export WANDB_RESUME=allow
#export WANDB_RUN_ID=UNIQUE_ID

# Start training
accelerate launch run_clm_prompt_tuning.py \
    --model_name_or_path andstor/gpt-j-6B-smart-contract \
    --dataset_name andstor/vulnerable_smart_contracts \
    --dataset_config_name manual_all_s5_embedded \
    --adapter_name default \
    --num_virtual_tokens 4 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --block_size 2048 \
    --report_to all \
    --save_steps 250 \
    --load_best_model_at_end \
    --do_train \
    --do_eval \
    --bf16 \
    --logging_first_step --logging_steps 1 \
    --num_train_epochs 3 \
    --evaluation_strategy steps --eval_steps 5 \
    --output_dir output
echo "-------------------------- DONE ---------------------------"
