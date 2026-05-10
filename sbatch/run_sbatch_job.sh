#!/bin/bash -l 
# (#!) Shebang, specifies Bash as the shell (Starts a login shell)
# Ensures all environment settings are loaded
# Set output directory

# SLURM job configuration
#SBATCH --job-name=lora_stable_diffusion                          # Name of the job
#SBATCH --output=slurm_logs/job_%j_%x.out                             # File where output is stored
#SBATCH --error=slurm_logs/job_%j_%x.err                              # File where error is stored
#SBATCH --time=08:00:00                                     # Maximum runtime (HH:MM:SS)
#SBATCH --partition=a40                                     # Partition to use (A40 GPU nodes)
#SBATCH --gres=gpu:a40:1                                    # Request 1 A40 GPU
#SBATCH --cpus-per-task=16                                  # Request 16 CPU cores per task
#SBATCH --export=NONE                                       # Do not inherit the user’s environment

# Optional: Email notifications (uncomment and update your email)
#SBATCH --mail-user=khaled.gamal@utn.de  # Email for notifications
#SBATCH --mail-type=BEGIN,END,FAIL          # When to receive emails (job start, end, or failure)

# Optional: Array jobs for running multiple tasks in parallel
# #SBATCH --array=1-10%2  # Run 10 jobs (indexed 1-10), with max 2 running at a time

# Optional: Job dependency (only start if JobID 12345 finishes successfully)
# #SBATCH --dependency=afterok:12345

mkdir -p "slurm_logs" # Create the output directory if it doesn't exist


# Load required modules
module load python/3.9-anaconda        # Load Python (Anaconda) module
module load cuda/12.4.1                # Load CUDA 12.4.1 for GPU computations

# Activate Conda environment
source /apps/python/3.9-anaconda/etc/profile.d/conda.sh
conda activate /home/hpc/v123be/v123be29/.conda/envs/lora

# Navigate to the working directory
cd /home/hpc/v123be/v123be29/repos/stable_diffusion_lora_fine_tuning/src


# Run the Python script
python train_lora.py

# Optional: Print resource usage stats after the job finishes
# echo "Job finished at $(date)"
# seff $SLURM_JOB_ID  # Shows job efficiency stats (if available)


# Example command line 
##  sbatch --job-name="" run_sbatch_job.sh