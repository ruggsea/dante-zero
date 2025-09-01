#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dante GRPO Training Script

This script uses Generative Pre-trained Transformer Optimization (GRPO) to fine-tune a model
to write Dante-like endecasillabi (11-syllable lines used in Dante's Divine Comedy).

It uses reward functions based on:
1. Endecasillabo verification (using simple_endecasillabo_checker.py)
2. Divine Comedy line checking (using divine_comedy_checker.py)
3. Repetition avoidance

Author: ruggsea
Date: 2024
"""

import os
import argparse
import torch
import wandb
from trl import GRPOConfig, GRPOTrainer

# Import from local modules
from reward_functions import (
    no_repetition_reward_func,
    # no_repetition_uniqueness_reward_func,  # Disabled: remove second repetition reward ([0,1])
    verse_reward_func,
    endecasillabo_reward_func,
    check_divine_comedy_plagiarism
)
from callbacks import (
    TokenizerWrapper,
    WandbCompletionCallback,
    BestCompletionsCallback
)
from utils import (
    create_dataset,
    generate_run_name,
    create_model_readme
)
from config import compute_output_dir
from divine_comedy_checker import DivineComedyChecker, is_from_divine_comedy
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

def main():
    # Suppress specific warnings
    import warnings
    import re
    
    # Define a filter function to catch the padding warning
    def filter_padding_warning(message, category, filename, lineno, file=None, line=None):
        if category == UserWarning and "padding was detected" in str(message):
            return None  # Suppress the warning
        return warnings.defaultaction  # Use default action for other warnings
    
    # Apply the filter
    warnings.filterwarnings("ignore", message=".*padding was detected.*")
    warnings.filterwarnings("ignore", message=".*padding_side='left'.*")
    
    # Monkey patch the generation module to suppress the warning at its source
    try:
        from transformers.generation.utils import GenerationMixin
        
        # Store the original _validate_model_kwargs method
        original_validate = GenerationMixin._validate_model_kwargs
        
        # Create a patched version that doesn't check padding side
        def patched_validate(self, *args, **kwargs):
            # Call the original method but catch and ignore the padding warning
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*padding was detected.*")
                warnings.filterwarnings("ignore", message=".*padding_side='left'.*")
                return original_validate(self, *args, **kwargs)
        
        # Apply the patch
        GenerationMixin._validate_model_kwargs = patched_validate
        print("Successfully patched generation module to suppress padding warnings")
    except Exception as e:
        print(f"Note: Could not patch generation module: {e}")
    
    # Args
    parser = argparse.ArgumentParser(description="Train a model with GRPO to generate Dante-like endecasillabi")
    parser.add_argument("--model_name", type=str, default="PleIAs/Pleias-350m-Preview", help="Model name to use")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: auto-generated based on date and model)")
    parser.add_argument("--run_name", type=str, default=None, help="Run name for wandb (default: auto-generated based on date and model)")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--max_prompt_length", type=int, default=256, help="Maximum prompt length")
    parser.add_argument("--max_completion_length", type=int, default=256, help="Maximum completion length")
    parser.add_argument("--num_generations", type=int, default=16, help="Number of generations per prompt")
    parser.add_argument("--sample_size", type=int, default=1000, help="Number of samples to use from dataset")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--endeca_mode", type=str, default="top_prob", choices=["top_prob","unique_only"], help="Endecasillabo reward mode")
    parser.add_argument("--max_endeca_candidates", type=int, default=20, help="Max syllabification candidates to enumerate")
    parser.add_argument("--rewards", type=str, default="all", 
                       help=(
                           "Comma-separated rewards to use, e.g. 'no_repetition,endecasillabo'. "
                           "Options: all, endecasillabo_only, no_repetition, no_repetition_uniqueness, verse_structure, endecasillabo, divine_comedy"
                       ))
    parser.add_argument("--kl_beta", type=float, default=0.05, help="KL-divergence coefficient (beta) for GRPO. Default 0.05")
    parser.add_argument("--endeca_weight", type=float, default=5.0, help="Multiplier applied to endecasillabo reward (default 5.0)")
    parser.add_argument("--use_repetition_reward", type=int, default=1, help="Whether to include repetition penalty reward (1) or not (0)")
    parser.add_argument("--gen_repetition_penalty", type=float, default=1.2, help="Repetition penalty used in generation/vLLM during GRPO")
    args = parser.parse_args()

    # Validate that batch_size is divisible by num_generations
    if args.batch_size % args.num_generations != 0:
        print(f"Error: batch_size ({args.batch_size}) must be divisible by num_generations ({args.num_generations})")
        print(f"Adjusting num_generations to {args.batch_size}")
        args.num_generations = args.batch_size
    print(f"Using batch size: {args.batch_size}")
    # Generate a unique run name
    run_name = generate_run_name(args.model_name, args.run_name)

    # Set output directory: prefer CHECKPOINT_LOCATION if provided
    args.output_dir = compute_output_dir(run_name, args.output_dir)
    print(f"Using output directory: {args.output_dir}")
    print(f"Using run name: {run_name}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Distributed env (MASTER_ADDR/MASTER_PORT) should be provided via environment

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Weights & Biases
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "dante-zero"),
        name=run_name,
        config=vars(args)
    )

    # Create tokenizer wrapper
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    
    # Match notebook behavior: ensure EOS and PAD are aligned and enforce left padding
    if tokenizer.eos_token is None:
        try:
            tokenizer.eos_token = "<|end_of_text|>"
        except Exception:
            pass
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    except Exception:
        pass
    tokenizer.padding_side = "left"
    
    # Wrap tokenizer if needed elsewhere (trainer requires a real tokenizer instance)
    wrapped_tokenizer = TokenizerWrapper(tokenizer)
    
    # Create model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        token=os.environ.get("HF_TOKEN"),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    ).to(device)

    # Create DivineComedyChecker
    print("Creating Divine Comedy Checker...")
    dc_checker = DivineComedyChecker()
    dc_checker.load_verses()

    # Create dataset
    print("Creating dataset...")
    dataset = create_dataset(args.sample_size, dc_checker)

    # Avoid duplicate wandb.init (already initialized above)

    # Create BestCompletionsCallback
    print("Creating BestCompletionsCallback...")
    
    # Build reward registry (with a couple of user-friendly aliases)
    reward_registry = {
        "verse_structure": (verse_reward_func, "Verse Structure"),
        "endecasillabo": (endecasillabo_reward_func, "Endecasillabo"),
        # aliases for convenience/typos
        "endecasyllab": (endecasillabo_reward_func, "Endecasillabo"),
        "endecasyllable": (endecasillabo_reward_func, "Endecasillabo"),
        "divine_comedy": (check_divine_comedy_plagiarism, "Divine Comedy Plagiarism"),
    }
    if int(args.use_repetition_reward) == 1:
        reward_registry["no_repetition"] = (no_repetition_reward_func, "No Repetition")

    # Parse --rewards into a list, supporting comma-separated values and legacy single-value options
    rewards_arg = (args.rewards or "").strip().lower()
    if rewards_arg == "all":
        base_keys = ["verse_structure", "endecasillabo", "divine_comedy"]
        selected_keys = (["no_repetition"] + base_keys) if int(args.use_repetition_reward) == 1 else base_keys
    elif rewards_arg == "endecasillabo_only":
        selected_keys = ["endecasillabo"]
    else:
        selected_keys = [part.strip() for part in rewards_arg.split(",") if part.strip()]
        if not selected_keys:
            raise ValueError("No rewards selected. Provide --rewards or use 'all'.")

    # Deduplicate while preserving order and validate
    seen = set()
    normalized_keys = []
    for key in selected_keys:
        key_norm = key
        if key_norm not in reward_registry:
            allowed = ["all", "endecasillabo_only"] + sorted(reward_registry.keys())
            raise ValueError(f"Unknown reward '{key}'. Allowed: {', '.join(allowed)}")
        if key_norm not in seen:
            seen.add(key_norm)
            normalized_keys.append(key_norm)

    reward_funcs = [reward_registry[k][0] for k in normalized_keys]
    reward_names = [reward_registry[k][1] for k in normalized_keys]

    print(f"Using rewards: {reward_names}")
    
    best_completions_callback = BestCompletionsCallback(
        tokenizer=wrapped_tokenizer,
        reward_funcs=reward_funcs,
        reward_names=reward_names,
        dc_checker=dc_checker,
        endeca_mode=args.endeca_mode,
        max_candidates=args.max_endeca_candidates,
        endeca_weight=args.endeca_weight
    )

    # Create training arguments
    print("Creating training arguments...")
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        run_name=run_name,
        learning_rate=args.learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        logging_steps=1,
        bf16=True if torch.cuda.is_available() else False,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.num_epochs,
        save_steps=500,
        max_grad_norm=0.1,
        beta=args.kl_beta,
        log_on_each_node=False,
        use_vllm=True,
        vllm_tensor_parallel_size=1,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=.5,
        generation_kwargs={
            "repetition_penalty": float(args.gen_repetition_penalty),
        },
        log_completions=True,
        report_to=["wandb"]
    )

    # Monkey patch the generation module to directly suppress the padding warning
    try:
        import transformers.generation.utils
        import logging
        
        # Get the original logger.warning function
        original_warning = logging.Logger.warning
        
        # Create a patched version that filters out the padding warning
        def patched_warning(self, msg, *args, **kwargs):
            if "padding_side='left'" not in str(msg) and "right-padding was detected" not in str(msg):
                original_warning(self, msg, *args, **kwargs)
        
        # Apply the patch
        logging.Logger.warning = patched_warning
        print("Successfully patched logger to suppress padding warnings")
    except Exception as e:
        print(f"Note: Could not patch logger: {e}")

    # Create trainer with callbacks
    print("Creating trainer...")
    trainer = GRPOTrainer(
        model=model,
        processing_class=wrapped_tokenizer,
        reward_funcs=best_completions_callback.get_wrapped_reward_funcs(),  # Use wrapped reward functions
        args=training_args,
        train_dataset=dataset,
        callbacks=[
            WandbCompletionCallback(wrapped_tokenizer),
            best_completions_callback
        ]
    )
    
    # Directly modify the model's generation config to set padding_side to 'left'
    if hasattr(model, 'generation_config'):
        model.generation_config._padding_side = 'left'
        print("Successfully set model.generation_config._padding_side = 'left'")

    # Train model
    print("Starting training...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*padding was detected.*")
        warnings.filterwarnings("ignore", message=".*padding_side='left'.*")
        trainer.train()

    # Save final model to {output_dir}/final
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    print(f"Saving final model to {final_dir}")
    try:
        trainer.save_model(final_dir)
    except Exception as _e:
        # fallback
        try:
            model.save_pretrained(final_dir)
        except Exception:
            pass

    # Upload model to Hugging Face Hub
    print("Preparing to upload model to Hugging Face Hub...")
    try:
        from huggingface_hub import HfApi
        import shutil
        
        # Get HF credentials from environment
        hf_token = os.environ.get("HF_TOKEN")
        hf_username = os.environ.get("HF_USERNAME")
        
        if not hf_token or not hf_username:
            print("Warning: HF_TOKEN or HF_USERNAME not found in environment. Skipping upload to Hugging Face Hub.")
        else:
            # Use the same run name for the HF repo
            repo_name = run_name
            repo_id = f"{hf_username}/{repo_name}"
            
            # Create README.md for the model
            model_readme = create_model_readme(
                model_name=args.model_name,
                num_epochs=args.num_epochs,
                username=hf_username,
                run_name=run_name,
                repo_id=repo_id
            )
            
            # Write README to output directory
            with open(os.path.join(args.output_dir, "README.md"), "w", encoding="utf-8") as f:
                f.write(model_readme)
            
            # Create a new repo on the Hub
            api = HfApi()
            print(f"Creating repository: {repo_id}")
            api.create_repo(
                repo_id=repo_id,
                token=hf_token,
                private=False,
                exist_ok=True
            )
            
            # Upload the model
            print(f"Uploading model to {repo_id}...")
            api.upload_folder(
                folder_path=args.output_dir,
                repo_id=repo_id,
                token=hf_token
            )
            
            print(f"Model uploaded successfully to: https://huggingface.co/{repo_id}")
    
    except Exception as e:
        print(f"Error uploading model to Hugging Face Hub: {e}")

    print("Training complete!")

if __name__ == "__main__":
    main() 