#!/usr/bin/env bash

# List all seeds you want to iterate over
SEEDS=("42") #("0" "1" "2" "3" "4" "5" "6" "7" "8" "9")
LRS=("5e-4" "2e-4" "1e-4" "5e-5" "2e-5")
RANKS=("8" "32" "128")
DATA=("mrpc" "cola")

for seed in "${SEEDS[@]}"
do
  for lr in "${LRS[@]}"
  do
    for rank in "${RANKS[@]}"
    do
      for data in "${DATA[@]}"
      do
        echo "Submitting job for seed: ${seed} lr: ${lr} rank: ${rank}"

        # Submit a Slurm job via heredoc
        sbatch <<EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --partition=gpu
#SBATCH --time=00:10:00
#SBATCH --account=su007-fl-gpu

# Load necessary modules (Python, etc.)
module purge
module load GCC/13.2.0  OpenMPI/4.1.6 PIP-PyTorch/2.4.0-CUDA-12.4.0

export HYDRA_FULL_ERROR=1

# Set the Hugging Face token (DO NOT HARDCODE IN THE SCRIPT)
export HF_TOKEN="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"

# Option 1: Use Hugging Face CLI to log in
# echo $HF_TOKEN | huggingface-cli login --token

# Option 2: (Alternative) Use Python to set up the token directly in the environment
python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"

# Print a success message
echo "Successfully logged in to Hugging Face"

wandb login --relogin b1d7e377bf3fc7af261c70d7423b3ff33e146480

#srun python prec_run_exp.py -m ++dataset_name=${data} +init=gradient ++peft.lora_r=${rank} +peft=all wandb.name="prec-s16-norm-r${rank}-lr${lr}" ++init.weight="stable" peft.use_rslora=True peft.lora_alpha=16 ++init.stable_gamma=16 model.learning_rate=${lr} ++seed=${seed}
srun python prec_run_exp.py -m ++dataset_name=${data} +init=gradient ++peft.lora_r=${rank} +peft=all wandb.name="prec-spectral-t-r${rank}-lr${lr}" ++init.weight="stable" peft.use_rslora=True peft.lora_alpha=16 ++init.stable_gamma=16 model.learning_rate=${lr} ++seed=${seed} peft.use_loraplus=False init.direction="OS-LoRA-Full-N"
#srun python prec_run_exp.py -m ++dataset_name=${data} +init=default ++peft.lora_r=${rank} +peft=all wandb.name="p-lora-r${rank}-lr${lr}" model.learning_rate=${lr} ++seed=${seed} peft.use_loraplus=False
EOF

      done
    done
  done
done