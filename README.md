# Style LoRA Fine-Tuning with Stable Diffusion 1.5

This repository contains code and configurations for fine-tuning Stable Diffusion 1.5 using LoRA (Low-Rank Adaptation) to create a custom "Ghibli Market" style model. LoRA allows efficient model adaptation with minimal computational overhead and parameter updates.

## Overview

The project implements style transfer capabilities through parameter-efficient fine-tuning of Stable Diffusion 1.5. By using LoRA adapters, you can achieve high-quality style transfer while maintaining a relatively small model footprint suitable for various hardware configurations.

## Key Features

- Parameter-efficient fine-tuning using LoRA adapters
- Integration with Hugging Face Diffusers library
- PEFT (Parameter-Efficient Fine-Tuning) framework support
- PyTorch-based training pipeline
- Batch processing with configurable parameters
- Support for GPU acceleration and distributed training
- SLURM integration for high-performance computing environments

## Requirements

The project requires the following Python packages:

- diffusers
- peft
- torch >= 2.0
- torchvision
- safetensors
- transformers
- pillow
- tqdm
- accelerate
- huggingface-hub

For a complete list of dependencies with pinned versions, see `envs/requirements.txt`.

## Installation

Clone the repository and set up the environment:

```bash
git clone https://github.com/Abdullah-Taha9/style-lora-diffusion.git
cd style-lora-diffusion
```

Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r envs/requirements.txt
```

Or use the basic requirements file for a minimal setup:

```bash
pip install -r requirements.txt
```

## Project Structure

The repository is organized as follows:

- `code/` - Main training and inference scripts
- `configs/` - Configuration files for different training scenarios
- `envs/` - Environment specifications and detailed requirements
- `lora_out/` - Output directory for saved LoRA weights and checkpoints
- `samples/` - Sample input images and generated outputs
- `reports/` - Training reports, metrics, and evaluation results
- `sbatch/` - SLURM batch scripts for distributed training on HPC clusters

## Getting Started

Before running training, prepare your dataset and place sample images in the `samples/` directory. Update the configuration files in `configs/` to match your training setup and desired style parameters.

The training pipeline can be executed from the `code/` directory. Check the available scripts and their documentation for specific usage instructions.

## Configuration

Configuration files in the `configs/` directory control various aspects of the training process including:

- Learning rate and optimization parameters
- LoRA adapter configuration (rank, alpha, dropout)
- Training dataset paths and preprocessing options
- Output and checkpoint settings
- Model and tokenizer selection

## Training on HPC Systems

For training on clusters with SLURM job scheduling, use the provided batch scripts in the `sbatch/` directory. These scripts handle environment setup, GPU allocation, and distributed training configurations. Modify them according to your cluster's specifications.

## Output

Trained LoRA weights and model checkpoints are saved to the `lora_out/` directory. These adapters can be loaded onto the base Stable Diffusion 1.5 model for inference or further fine-tuning.


## References

For more information on LoRA and parameter-efficient fine-tuning, refer to:

- PEFT Library: https://github.com/huggingface/peft
- Diffusers Documentation: https://huggingface.co/docs/diffusers
- Stable Diffusion Model Card: https://huggingface.co/CompVis/stable-diffusion-v1-5

## License

This project is provided as-is for research and development purposes.
