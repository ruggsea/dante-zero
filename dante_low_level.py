#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dante Low-Level GRPO Training Script

This script uses a low-level implementation of Generative Pre-trained Transformer Optimization (GRPO)
to fine-tune a model to write Dante-like endecasillabi (11-syllable lines used in Dante's Divine Comedy).

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
import torch.nn.functional as F
import wandb
import copy
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
import json
import pathlib
from tqdm import tqdm
import time
from datetime import timedelta

# Import from local modules
from reward_functions import (
    no_repetition_reward_func,
    verse_reward_func,
    endecasillabo_reward_func,
    check_divine_comedy_plagiarism
)
from utils import (
    create_dataset,
    create_eval_dataset,
    generate_run_name,
    create_model_readme
)
from divine_comedy_checker import DivineComedyChecker, is_from_divine_comedy

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

def set_random_seed(seed: int = 42):
    """
    Set the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    # Set the seed for Python's built-in random module
    random.seed(seed)

    # Set the seed for NumPy
    np.random.seed(seed)

    # Set the seed for PyTorch
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def selective_log_softmax(logits, input_ids):
    """
    Compute the log probabilities for the tokens specified in input_ids.
    
    Args:
        logits (torch.Tensor): Shape (batch_size, seq_len, vocab_size) with raw logits
        input_ids (torch.Tensor): Shape (batch_size, seq_len) with token indices
        
    Returns:
        torch.Tensor: Shape (batch_size, seq_len) with log probabilities
    """
    # Convert raw logits into log probabilities along the vocabulary axis
    log_probs = F.log_softmax(logits, dim=-1)  # Shape: (batch_size, seq_len, vocab_size)
    
    # Gather the log probability for each token in input_ids
    selected_log_probs = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1))
    
    # Remove the extra last dimension
    return selected_log_probs.squeeze(-1)

def compute_log_probabilities(model, input_ids, attention_mask, logits_to_keep):
    """
    Compute per-token log probabilities for the completion tokens.
    
    Args:
        model: The language model
        input_ids: Token ids for both prompt and completion
        attention_mask: Mask indicating which tokens are real (1) or padding (0)
        logits_to_keep: Number of tokens for which we need log probabilities
        
    Returns:
        torch.Tensor: Log probabilities for the last logits_to_keep tokens
    """
    # Forward pass through the model
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Get logits from model output
    logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)
    
    # Remove the last logit as it predicts the next token after the sequence
    logits = logits[:, :-1, :]
    
    # Slice the input_ids to keep only the completion tokens
    completion_ids = input_ids[:, -logits_to_keep:]
    
    # Also slice the logits to keep only those for the completion tokens
    completion_logits = logits[:, -logits_to_keep:, :]
    
    # Compute log probabilities for the selected tokens
    return selective_log_softmax(completion_logits, completion_ids)

def create_completion_mask(completion_ids, eos_token_id):
    """
    Create a binary mask for the generated completion tokens.
    Tokens after the first EOS are masked out.
    
    Args:
        completion_ids: Token ids for the completions
        eos_token_id: ID of the end-of-sequence token
        
    Returns:
        torch.Tensor: Binary mask where 1s are tokens to keep
    """
    # Find positions with EOS token
    is_eos = completion_ids == eos_token_id
    
    # Default to sequence length if no EOS is found
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), 
                         dtype=torch.long, device=completion_ids.device)
    
    # For sequences with an EOS, update to the index of first occurrence
    mask_exists = is_eos.any(dim=1)
    eos_idx[mask_exists] = is_eos.int().argmax(dim=1)[mask_exists]
    
    # Create indices for each position in sequence
    sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
    
    # Create mask: 1 for positions up to and including first EOS, 0 after
    completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
    
    return completion_mask

def generate_completions(model, tokenizer, prompts, num_generations=4, max_completion_length=32):
    """
    Generate multiple completions for each prompt.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts: List of input prompts
        num_generations: Number of completions per prompt
        max_completion_length: Maximum length of each completion
        
    Returns:
        tuple: (prompt_ids, prompt_mask, completion_ids, completion_mask)
    """
    device = next(model.parameters()).device
    
    # Tokenize prompts with left padding
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    prompt_ids = inputs["input_ids"].to(device)
    prompt_mask = inputs["attention_mask"].to(device)
    prompt_length = prompt_ids.size(1)
    
    # Repeat each prompt num_generations times
    prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
    prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)
    
    # Generate completions
    outputs = model.generate(
        prompt_ids,
        attention_mask=prompt_mask,
        max_new_tokens=max_completion_length,
        do_sample=True,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Extract completion tokens (removing prompt)
    completion_ids = outputs[:, prompt_length:]
    
    # Create mask for valid completion tokens (up to EOS)
    completion_mask = create_completion_mask(completion_ids, tokenizer.eos_token_id)
    
    return prompt_ids, prompt_mask, completion_ids, completion_mask

def generate_rollout_data(model, ref_model, tokenizer, batch_samples, num_generations, max_completion_length):
    """
    Generate rollouts and compute log probabilities for both policy and reference models.
    
    Args:
        model: Current policy model
        ref_model: Reference model
        tokenizer: Tokenizer
        batch_samples: List of training samples
        num_generations: Number of completions per prompt
        max_completion_length: Maximum completion length
        
    Returns:
        dict: Rollout data including log probabilities and rewards
    """
    tokenizer.padding_side = "left"
    device = next(model.parameters()).device
    
    # Extract prompts from batch samples
    prompts = [sample["prompt"] for sample in batch_samples]
    
    # Generate completions
    with torch.no_grad():
        prompt_ids, prompt_mask, completion_ids, completion_mask = generate_completions(
            model, tokenizer, prompts, num_generations, max_completion_length
        )
        
        # Combine prompt and completion
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        
        # Compute log probabilities from current policy
        old_log_probs = compute_log_probabilities(model, input_ids, attention_mask, logits_to_keep)
        
        # Compute log probabilities from reference model
        ref_log_probs = compute_log_probabilities(ref_model, input_ids, attention_mask, logits_to_keep)
    
    # Format completions for reward functions
    formatted_completions = []
    for ids in completion_ids:
        text = tokenizer.decode(ids, skip_special_tokens=True)
        formatted_completions.append([{"content": text}])
    
    # Repeat prompts for each completion
    repeated_prompts = []
    for prompt in prompts:
        repeated_prompts.extend([prompt] * num_generations)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_mask": completion_mask,
        "old_log_probs": old_log_probs,
        "ref_log_probs": ref_log_probs,
        "formatted_completions": formatted_completions,
        "repeated_prompts": repeated_prompts,
        "logits_to_keep": logits_to_keep,
        "batch_size": len(prompts),
        "num_generations": num_generations
    }

def compute_combined_reward(completions, prompts=None, dc_checker=None):
    """
    Compute combined reward from multiple reward functions.
    
    Args:
        completions: List of completions
        prompts: List of prompts
        dc_checker: Divine Comedy checker instance
        
    Returns:
        tuple: (Combined rewards tensor, Dictionary of individual rewards)
    """
    # Compute individual rewards
    repetition_rewards = no_repetition_reward_func(completions, prompts)
    verse_rewards = verse_reward_func(completions, prompts)
    endecasillabo_rewards = endecasillabo_reward_func(completions, prompts)
    plagiarism_rewards = check_divine_comedy_plagiarism(completions, prompts, dc_checker=dc_checker)
    
    # Combine rewards with equal weights
    combined_rewards = []
    for i in range(len(completions)):
        rep_reward = repetition_rewards[i] if i < len(repetition_rewards) else 0.0
        verse_reward = verse_rewards[i] if i < len(verse_rewards) else 0.0
        end_reward = endecasillabo_rewards[i] if i < len(endecasillabo_rewards) else 0.0
        plag_reward = plagiarism_rewards[i] if i < len(plagiarism_rewards) else 0.0
        
        # Equal weight combination (all weights = 1.0)
        combined = rep_reward + verse_reward + end_reward + plag_reward
        combined_rewards.append(combined)
    
    # Package individual rewards for logging
    reward_components = {
        "repetition_rewards": repetition_rewards,
        "verse_rewards": verse_rewards,
        "endecasillabo_rewards": endecasillabo_rewards,
        "plagiarism_rewards": plagiarism_rewards
    }
    
    return torch.tensor(combined_rewards, dtype=torch.float32), reward_components

def compute_advantages(rewards, num_generations):
    """
    Compute advantages for PPO, safely handling edge cases.
    
    Args:
        rewards: Tensor of rewards
        num_generations: Number of completions per prompt
        
    Returns:
        torch.Tensor: Advantages tensor
    """
    batch_size = rewards.size(0)
    
    # Handle case where batch_size is too small
    if batch_size <= 1:
        # Just return normalized rewards if not enough data
        mean = rewards.mean()
        std = rewards.std() if batch_size > 0 else torch.tensor(1.0, device=rewards.device)
        return ((rewards - mean) / (std + 1e-8)).unsqueeze(1)
    
    # Check if we can reshape properly
    if batch_size % num_generations != 0:
        # Just normalize across the whole batch if we can't reshape
        mean = rewards.mean()
        std = rewards.std()
        advantages = (rewards - mean) / (std + 1e-8)
        return advantages.unsqueeze(1)
    
    # Otherwise do the proper per-prompt normalization
    prompts_per_batch = batch_size // num_generations
    rewards_by_group = rewards.view(prompts_per_batch, num_generations)
    
    # Compute mean and standard deviation for each group
    group_means = rewards_by_group.mean(dim=1, keepdim=True)
    group_stds = rewards_by_group.std(dim=1, keepdim=True)
    
    # Normalize rewards to get advantages
    normalized_rewards = (rewards_by_group - group_means) / (group_stds + 1e-8)
    
    # Flatten back to original shape
    advantages = normalized_rewards.reshape(batch_size)
    
    return advantages.unsqueeze(1)  # Add dimension for token-wise operations

class BestCompletionsTracker:
    """Track the best completions during training for logging to wandb."""
    
    def __init__(self, top_n=10):
        """Initialize the tracker.
        
        Args:
            top_n: Number of top completions to track
        """
        self.top_n = top_n
        self.reset()
    
    def reset(self):
        """Reset the tracker for a new epoch."""
        self.best_completions = []
    
    def add_completions(self, prompts, completions, total_rewards, reward_components):
        """Add completions to the tracker.
        
        Args:
            prompts: List of prompts
            completions: List of completions
            total_rewards: Tensor of total rewards
            reward_components: Dict of individual reward components
        """
        # Convert tensors to lists for easier handling
        total_reward_values = total_rewards.tolist()
        
        # Get individual reward components
        repetition_values = reward_components["repetition_rewards"]
        verse_values = reward_components["verse_rewards"]
        endecasillabo_values = reward_components["endecasillabo_rewards"]
        plagiarism_values = reward_components["plagiarism_rewards"]
        
        # Add each completion with its rewards
        for i, (prompt, completion, total_reward) in enumerate(zip(prompts, completions, total_reward_values)):
            # Create an entry for this completion
            entry = {
                "prompt": prompt,
                "completion": completion,
                "full_text": f"{prompt} {completion}",
                "total_reward": total_reward,
                "repetition_reward": repetition_values[i] if i < len(repetition_values) else 0.0,
                "verse_reward": verse_values[i] if i < len(verse_values) else 0.0,
                "endecasillabo_reward": endecasillabo_values[i] if i < len(endecasillabo_values) else 0.0,
                "plagiarism_reward": plagiarism_values[i] if i < len(plagiarism_values) else 0.0
            }
            self.best_completions.append(entry)
        
        # Sort by total reward and keep only top_n
        self.best_completions.sort(key=lambda x: x["total_reward"], reverse=True)
        self.best_completions = self.best_completions[:self.top_n]
    
    def log_to_wandb(self, epoch):
        """Log the best completions to wandb.
        
        Args:
            epoch: Current epoch number
        """
        if not self.best_completions or wandb.run is None:
            return
        
        # Create a table for wandb
        table = wandb.Table(columns=[
            "Prompt", "Completion", "Full Text", "Total Reward", 
            "Repetition Reward", "Verse Reward", "Endecasillabo Reward", "Plagiarism Reward"
        ])
        
        # Add rows to the table
        for entry in self.best_completions:
            table.add_data(
                entry["prompt"],
                entry["completion"],
                entry["full_text"],
                entry["total_reward"],
                entry["repetition_reward"],
                entry["verse_reward"],
                entry["endecasillabo_reward"],
                entry["plagiarism_reward"]
            )
        
        # Log the table to wandb
        wandb.log({f"best_completions_epoch_{epoch}": table})

def maximize_grpo_objective(model, ref_model, rollout_data, tokenizer, dc_checker, 
                           optimizer, beta, epsilon):
    """
    Update policy model by maximizing the GRPO objective.
    
    Args:
        model: Current policy model
        ref_model: Reference model
        rollout_data: Dictionary with rollout data
        tokenizer: Tokenizer
        dc_checker: Divine Comedy checker
        optimizer: Optimizer
        beta: KL penalty coefficient
        epsilon: Clipping parameter
        
    Returns:
        tuple: (loss value, dict of reward metrics, dict of reward components)
    """
    # Extract data from rollout
    input_ids = rollout_data["input_ids"]
    attention_mask = rollout_data["attention_mask"]
    completion_mask = rollout_data["completion_mask"]
    old_log_probs = rollout_data["old_log_probs"]
    ref_log_probs = rollout_data["ref_log_probs"]
    logits_to_keep = rollout_data["logits_to_keep"]
    
    # Forward pass to get current log probabilities
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]  # Remove last position
    completion_ids = input_ids[:, -logits_to_keep:]
    completion_logits = logits[:, -logits_to_keep:, :]
    current_log_probs = selective_log_softmax(completion_logits, completion_ids)
    
    # Compute policy ratio
    ratio = torch.exp(current_log_probs - old_log_probs)
    
    # Get reward data
    formatted_completions = rollout_data["formatted_completions"]
    repeated_prompts = rollout_data["repeated_prompts"]
    batch_size = rollout_data["batch_size"]
    num_generations = rollout_data["num_generations"]
    
    # Compute rewards
    rewards, reward_components = compute_combined_reward(
        [comp[0]["content"] for comp in formatted_completions],
        repeated_prompts, 
        dc_checker
    )
    rewards = rewards.to(model.device)
    
    # Calculate per-prompt rewards safely
    prompts_per_batch = max(1, batch_size // num_generations)  # Ensure at least 1
    
    # Check if we have the expected number of rewards
    total_completions = len(rewards)
    
    # Log average reward (properly handling the case when batch is smaller than expected)
    avg_reward = rewards.mean().item()
    
    # Convert reward components to tensors for easier handling
    repetition_rewards = torch.tensor(reward_components["repetition_rewards"], device=model.device)
    verse_rewards = torch.tensor(reward_components["verse_rewards"], device=model.device)
    endecasillabo_rewards = torch.tensor(reward_components["endecasillabo_rewards"], device=model.device)
    plagiarism_rewards = torch.tensor(reward_components["plagiarism_rewards"], device=model.device)
    
    # Calculate component averages
    repetition_avg = repetition_rewards.mean().item() if len(repetition_rewards) > 0 else 0.0
    verse_avg = verse_rewards.mean().item() if len(verse_rewards) > 0 else 0.0
    endecasillabo_avg = endecasillabo_rewards.mean().item() if len(endecasillabo_rewards) > 0 else 0.0
    plagiarism_avg = plagiarism_rewards.mean().item() if len(plagiarism_rewards) > 0 else 0.0
    
    # Don't print here to avoid cluttering the progress bar
    # print(f"Average Reward: {avg_reward:.4f}")
    
    # Store reward metrics in a dictionary
    reward_metrics = {
        "final_reward_total": avg_reward,
        "final_reward_repetition": repetition_avg,
        "final_reward_verse": verse_avg,
        "final_reward_endecasillabo": endecasillabo_avg,
        "final_reward_plagiarism": plagiarism_avg
    }
    
    # Compute advantages
    advantages = compute_advantages(rewards, num_generations)
    
    # Compute surrogate loss with clipping
    surrogate1 = ratio * advantages
    surrogate2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    surrogate_loss = torch.min(surrogate1, surrogate2)
    
    # Compute KL divergence penalty
    kl_div = torch.exp(ref_log_probs - current_log_probs) - (ref_log_probs - current_log_probs) - 1
    kl_value = (kl_div * completion_mask).sum() / completion_mask.sum()
    
    # Combine losses
    per_token_loss = surrogate_loss - beta * kl_div
    loss = -((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    
    # Calculate the surrogate loss component
    surrogate_loss_value = -((surrogate_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean().item()
    
    # Optimization step
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
    optimizer.step()
    
    # Log metrics to wandb with proper grouping
    if wandb.run is not None:
        wandb.log({
            "rewards": {
                "total": avg_reward,
                "repetition": repetition_avg,
                "verse": verse_avg,
                "endecasillabo": endecasillabo_avg,
                "plagiarism": plagiarism_avg,
            },
            "losses": {
                "surrogate": surrogate_loss_value,
                "kl": kl_value.item(),
            }
        })
    
    return loss.item(), reward_metrics, reward_components

def evaluate_model(model, tokenizer, eval_data, dc_checker, num_generations=4, max_completion_length=128):
    """
    Evaluate the model on a separate evaluation dataset.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        eval_data: Evaluation dataset
        dc_checker: Divine Comedy checker
        num_generations: Number of generations per prompt
        max_completion_length: Maximum completion length
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    model.eval()
    device = next(model.parameters()).device
    
    all_prompts = []
    all_completions = []
    all_rewards = []
    reward_components = {
        "repetition_rewards": [],
        "verse_rewards": [],
        "endecasillabo_rewards": [],
        "plagiarism_rewards": []
    }
    
    # Process each example in the evaluation set
    print(f"Evaluating model on {len(eval_data)} examples with {num_generations} generations each...")
    
    eval_progress = tqdm(
        total=len(eval_data) * num_generations,
        desc="Evaluation",
        position=0,
        leave=True,
        ncols=100
    )
    
    with torch.no_grad():
        for example in eval_data:
            prompt = example["text"]
            prompt_tokens = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Generate completions for this prompt
            for _ in range(num_generations):
                outputs = model.generate(
                    prompt_tokens.input_ids,
                    attention_mask=prompt_tokens.attention_mask,
                    max_new_tokens=max_completion_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.2
                )
                
                completion = tokenizer.decode(outputs[0][prompt_tokens.input_ids.shape[1]:], skip_special_tokens=True)
                all_prompts.append(prompt)
                all_completions.append(completion)
                eval_progress.update(1)
    
    eval_progress.close()
    
    # Calculate rewards for all completions
    rewards, components = compute_combined_reward(all_completions, all_prompts, dc_checker)
    
    # Calculate average rewards
    avg_reward = rewards.mean().item()
    repetition_avg = sum(components["repetition_rewards"]) / len(components["repetition_rewards"]) if components["repetition_rewards"] else 0
    verse_avg = sum(components["verse_rewards"]) / len(components["verse_rewards"]) if components["verse_rewards"] else 0
    endecasillabo_avg = sum(components["endecasillabo_rewards"]) / len(components["endecasillabo_rewards"]) if components["endecasillabo_rewards"] else 0
    plagiarism_avg = sum(components["plagiarism_rewards"]) / len(components["plagiarism_rewards"]) if components["plagiarism_rewards"] else 0
    
    # Package evaluation metrics
    eval_metrics = {
        "eval_reward_total": avg_reward,
        "eval_reward_repetition": repetition_avg,
        "eval_reward_verse": verse_avg,
        "eval_reward_endecasillabo": endecasillabo_avg,
        "eval_reward_plagiarism": plagiarism_avg
    }
    
    print(f"Evaluation completed. Average reward: {avg_reward:.4f}")
    return eval_metrics

def train_with_grpo(model, tokenizer, train_data, eval_data, dc_checker, 
                   num_iterations=1, steps_per_iteration=500, 
                   batch_size=4, num_generations=4, max_completion_length=128, 
                   beta=0.1, learning_rate=5e-6, epsilon=0.2):
    """
    Train model with GRPO algorithm.
    
    Args:
        model: Initial model to fine-tune
        tokenizer: Tokenizer
        train_data: Training dataset
        eval_data: Evaluation dataset
        dc_checker: Divine Comedy checker
        num_iterations: Number of outer iterations
        steps_per_iteration: Steps per iteration
        batch_size: Batch size (number of prompts)
        num_generations: Completions per prompt
        max_completion_length: Maximum completion length
        beta: KL penalty coefficient
        learning_rate: Learning rate
        epsilon: Clipping parameter
        
    Returns:
        tuple: (model, dict of final metrics)
    """
    device = next(model.parameters()).device
    
    # Initialize optimizer with weight decay
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.1
    )
    
    # Setup cosine scheduler with 10% warmup
    warmup_steps = int(0.1 * num_iterations * steps_per_iteration)
    total_steps = num_iterations * steps_per_iteration
    
    def get_lr_multiplier(step):
        # Warmup phase
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # Cosine decay phase
        else:
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    # Global step counter for scheduling
    global_step = 0
    
    # Track final metrics
    final_metrics = {}
    
    # Create best completions tracker
    best_completions_tracker = BestCompletionsTracker(top_n=10)
    
    # For time estimation
    start_time = time.time()
    
    # Calculate the total number of steps across all iterations
    total_training_steps = 0
    for _ in range(1, num_iterations + 1):
        # Calculate number of batches per epoch for this iteration
        prompts_per_batch = batch_size // num_generations
        num_samples = len(train_data)
        batches_per_epoch = (num_samples + prompts_per_batch - 1) // prompts_per_batch
        iteration_steps = min(batches_per_epoch, steps_per_iteration) if steps_per_iteration > 0 else batches_per_epoch
        total_training_steps += iteration_steps
    
    # Initialize global progress bar
    global_progress_bar = tqdm(
        total=total_training_steps,
        desc=f"Overall Training Progress",
        position=0,
        leave=True,
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )
    
    # Main training loop
    for iteration in range(1, num_iterations + 1):
        print(f"\nStarting iteration {iteration}/{num_iterations}")
        
        # Reset the best completions tracker for this epoch
        best_completions_tracker.reset()
        
        # Create reference model
        reference_model = copy.deepcopy(model)
        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad = False
        reference_model = reference_model.to(device)
        
        # Set model to training mode
        model.train()
        
        # Shuffle dataset at the beginning of each epoch
        dataset_indices = list(range(len(train_data)))
        random.shuffle(dataset_indices)
        
        # Calculate number of batches per epoch
        # If batch_size is the number of total examples (prompts + generations),
        # we need to divide by num_generations to get number of unique prompts per batch
        prompts_per_batch = batch_size // num_generations
        num_samples = len(train_data)
        batches_per_epoch = (num_samples + prompts_per_batch - 1) // prompts_per_batch
        
        # Use either the calculated batches needed for full dataset, or the requested steps
        actual_steps = min(batches_per_epoch, steps_per_iteration) if steps_per_iteration > 0 else batches_per_epoch
        print(f"Dataset has {num_samples} samples, will process {actual_steps} batches with {prompts_per_batch} prompts per batch")
        
        # Track dataset position
        dataset_position = 0
        
        # Calculate time per step for better estimation
        if global_step > 0:
            elapsed_time = time.time() - start_time
            time_per_step = elapsed_time / global_step
        else:
            time_per_step = 0.1  # Initial guess if we don't have data yet
        
        # Update global progress bar description with current iteration
        global_progress_bar.set_description(f"Training [{iteration}/{num_iterations}]")
        
        # Steps loop
        for step in range(1, actual_steps + 1):
            # Set learning rate based on scheduler
            current_lr = learning_rate * get_lr_multiplier(global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            global_step += 1
            
            # Get batch indices for this step, handling wrap-around if needed
            batch_indices = []
            for i in range(prompts_per_batch):
                idx = (dataset_position + i) % num_samples
                batch_indices.append(dataset_indices[idx])
            dataset_position = (dataset_position + prompts_per_batch) % num_samples
            
            # Get prompts for this batch
            batch_samples = [train_data[i] for i in batch_indices]
            
            # Generate rollout data
            rollout_data = generate_rollout_data(
                model, reference_model, tokenizer,
                batch_samples, num_generations, max_completion_length
            )
            
            # Update model
            loss_value, reward_metrics, reward_components = maximize_grpo_objective(
                model, reference_model, rollout_data, tokenizer,
                dc_checker, optimizer, beta, epsilon
            )
            
            # Get raw completions and rewards for tracking best completions
            prompts = rollout_data["repeated_prompts"]
            completions = [comp[0]["content"] for comp in rollout_data["formatted_completions"]]
            
            # Calculate total rewards by combining components
            # Convert to CPU tensors first to avoid device issues
            repetition_rewards = torch.tensor(reward_components["repetition_rewards"])
            verse_rewards = torch.tensor(reward_components["verse_rewards"])
            endecasillabo_rewards = torch.tensor(reward_components["endecasillabo_rewards"])
            plagiarism_rewards = torch.tensor(reward_components["plagiarism_rewards"])
            
            total_rewards = repetition_rewards + verse_rewards + endecasillabo_rewards + plagiarism_rewards
            
            # Add to best completions tracker
            best_completions_tracker.add_completions(prompts, completions, total_rewards, reward_components)
            
            # Store the reward metrics for the last step of the last iteration
            if iteration == num_iterations and step == actual_steps:
                final_metrics.update(reward_metrics)
            
            # Update global progress bar with latest information
            avg_reward = reward_metrics["final_reward_total"]
            global_progress_bar.set_postfix({
                'loss': f"{loss_value:.4f}",
                'reward': f"{avg_reward:.4f}",
                'lr': f"{current_lr:.8f}",
                'iter': f"{iteration}/{num_iterations}"
            })
            global_progress_bar.update(1)
            
            # Print step-by-step progress (using tqdm.write to not interfere with progress bar)
            global_progress_bar.write(f"Iteration {iteration}/{num_iterations}, Step {step}/{actual_steps}, "
                  f"Loss: {loss_value:.4f}, Reward: {avg_reward:.4f}, LR: {current_lr:.8f}")
            
            # Recalculate time per step for better estimates
            elapsed_time = time.time() - start_time
            time_per_step = elapsed_time / global_step
            
            # Log basic metrics to wandb (detailed metrics are logged in maximize_grpo_objective)
            if wandb.run is not None:
                wandb.log({
                    "iteration": iteration,
                    "step": step,
                    "global_step": global_step,
                    "learning_rate": current_lr
                })
        
        # Run evaluation after each epoch
        print("\nRunning evaluation after epoch...")
        eval_metrics = evaluate_model(
            model, tokenizer, eval_data, dc_checker, 
            num_generations=1,  # Always use 1 generation for evaluation
            max_completion_length=max_completion_length
        )
        
        # Log evaluation metrics to wandb
        if wandb.run is not None:
            wandb.log({
                "eval": {
                    "reward_total": eval_metrics["eval_reward_total"],
                    "reward_repetition": eval_metrics["eval_reward_repetition"],
                    "reward_verse": eval_metrics["eval_reward_verse"],
                    "reward_endecasillabo": eval_metrics["eval_reward_endecasillabo"],
                    "reward_plagiarism": eval_metrics["eval_reward_plagiarism"]
                },
                "iteration": iteration
            })
        
        # Update final metrics with evaluation metrics
        final_metrics.update(eval_metrics)
        
        # Log best completions for this epoch
        best_completions_tracker.log_to_wandb(iteration)
        
        # Print ETA for remaining iterations
        remaining_iterations = num_iterations - iteration
        if remaining_iterations > 0:
            steps_remaining = remaining_iterations * actual_steps
            estimated_seconds = steps_remaining * time_per_step
            eta = timedelta(seconds=int(estimated_seconds))
            print(f"Estimated time remaining: {eta}")
    
    # Close global progress bar at the end
    global_progress_bar.close()
    
    return model, final_metrics

def optimize_model_memory(model):
    """
    Apply memory optimizations to the model.
    
    Args:
        model: The model to optimize
        
    Returns:
        The optimized model
    """
    # Set model to training mode
    model.train()
    
    # Disable caching for gradient checkpointing
    model.config.use_cache = False
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Enable input gradients
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    return model

def generate_sample(model, tokenizer, prompt, max_new_tokens=200):
    """
    Generate a sample completion for evaluation.
    
    Args:
        model: The model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum number of new tokens to generate
        
    Returns:
        str: Generated text
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def is_best_model(final_metrics, tracking_file="best_model_reward.json"):
    """
    Check if the current model has the best reward so far.
    
    Args:
        final_metrics: Dictionary of metrics including eval rewards
        tracking_file: Path to the JSON file tracking the best reward
        
    Returns:
        bool: True if this is the best model so far
    """
    # Get evaluation reward (prefer final eval reward over training reward)
    final_eval_reward = final_metrics.get("final_eval_reward_total", None)
    eval_reward = final_metrics.get("eval_reward_total", None)
    train_reward = final_metrics.get("final_reward_total", None)
    
    # Use the best available metric, preferring final evaluation
    if final_eval_reward is not None:
        final_reward = final_eval_reward
    elif eval_reward is not None:
        final_reward = eval_reward
    elif train_reward is not None:
        final_reward = train_reward
    else:
        final_reward = 10.0  # Default to high value if no metrics available
    
    # Set a very high initial threshold so most models don't get uploaded
    best_reward = {"value": 10.0}  # Initialize with high value (rewards are negative)
    
    # Create the file if it doesn't exist
    if os.path.exists(tracking_file):
        try:
            with open(tracking_file, "r") as f:
                best_reward = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    
    # Check if current reward is better (lower) than best so far
    is_best = final_reward < best_reward["value"]
    
    # Update the best reward if current is better
    if is_best:
        best_reward["value"] = final_reward
        with open(tracking_file, "w") as f:
            json.dump(best_reward, f)
        print(f"New best reward: {final_reward:.4f} (eval reward)")
    else:
        print(f"Current reward ({final_reward:.4f}) is not better than best reward ({best_reward['value']:.4f})")
    
    return is_best

def main():
    # Suppress specific warnings
    import warnings
    warnings.filterwarnings("ignore", message=".*padding was detected.*")
    warnings.filterwarnings("ignore", message=".*padding_side='left'.*")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a model with low-level GRPO to generate Dante-like endecasillabi")
    parser.add_argument("--model_name", type=str, default="PleIAs/Pleias-350m-Preview", help="Model name to use")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--run_name", type=str, default=None, help="Run name for wandb")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_generations", type=int, default=16, help="Number of generations per prompt")
    parser.add_argument("--max_completion_length", type=int, default=128, help="Maximum completion length")
    parser.add_argument("--sample_size", type=int, default=1000, help="Number of samples to use from dataset")
    parser.add_argument("--beta", type=float, default=0.1, help="KL penalty coefficient")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Clipping parameter")
    parser.add_argument("--steps_per_epoch", type=int, default=500, help="Steps per training epoch")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device_id", type=int, default=None, help="CUDA device ID to use (defaults to 0 if available)")
    args = parser.parse_args()
    
    # Set random seed
    set_random_seed(42)
    
    # Generate a unique run name
    run_name = generate_run_name(args.model_name, args.run_name)
    
    # Set output directory to use HF_HOME
    from huggingface_hub import constants
    hf_home = os.environ.get("HF_HOME", constants.HF_HOME)
    if args.output_dir is None:
        args.output_dir = os.path.join(hf_home, "dante_models", run_name)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Using output directory: {args.output_dir}")
    
    # Setup device
    if torch.cuda.is_available():
        if args.device_id is not None:
            if args.device_id >= torch.cuda.device_count():
                print(f"Warning: Requested device_id {args.device_id} exceeds available devices. Using device 0.")
                device = torch.device("cuda:0")
            else:
                device = torch.device(f"cuda:{args.device_id}")
        else:
            device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Initialize wandb with more detailed config
    print("Initializing wandb...")
    wandb_config = {
        # Model parameters
        "model_name": args.model_name,
        "output_dir": args.output_dir,
        
        # Training parameters
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "num_generations": args.num_generations,
        "max_completion_length": args.max_completion_length,
        "sample_size": args.sample_size,
        "steps_per_epoch": args.steps_per_epoch,
        
        # GRPO parameters
        "beta": args.beta,
        "epsilon": args.epsilon,
        
        # System info
        "device": str(device),
        "device_id": args.device_id if args.device_id is not None else 0 if torch.cuda.is_available() else None,
        "dtype": "bfloat16" if torch.cuda.is_available() else "float32",
        "resume_from_checkpoint": args.resume_from_checkpoint,
        
        # Dataset info
        "dante_tercets": True,
        "seed": 42
    }
    
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "dante-zero"),
        name=run_name,
        config=wandb_config
    )
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        token=os.environ.get("HF_TOKEN"),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    ).to(device)
    
    # Optimize model memory usage
    model = optimize_model_memory(model)
    
    # Create Divine Comedy checker
    print("Creating Divine Comedy Checker...")
    dc_checker = DivineComedyChecker()
    dc_checker.load_verses()
    
    # Create dataset
    print("Creating dataset...")
    train_data = create_dataset(args.sample_size, dc_checker)
    
    # Create separate evaluation dataset (50 examples)
    print("\nCreating evaluation dataset...")
    eval_data = create_eval_dataset(50, dc_checker)
    
    # Validate that batch_size is divisible by num_generations
    if args.batch_size % args.num_generations != 0:
        print(f"Error: batch_size ({args.batch_size}) must be divisible by num_generations ({args.num_generations})")
        print(f"Adjusting num_generations to {args.batch_size}")
        args.num_generations = args.batch_size
    
    # Generate a sample before training
    print("\nGenerating sample before training...")
    pre_sample = generate_sample(model, tokenizer, "Nel mezzo del cammin di nostra vita")
    print(f"Sample before training:\n{pre_sample}\n")
    
    # Train model with GRPO
    print("\nStarting GRPO training...")
    model, final_metrics = train_with_grpo(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        eval_data=eval_data,
        dc_checker=dc_checker,
        num_iterations=args.num_epochs,
        steps_per_iteration=args.steps_per_epoch,
        batch_size=args.batch_size,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        learning_rate=args.learning_rate,
        epsilon=args.epsilon
    )
    
    # Add final metrics to wandb summary
    if wandb.run is not None:
        for key, value in final_metrics.items():
            wandb.run.summary[key] = value
        
        # Also add general run info to summary
        wandb.run.summary['epochs_completed'] = args.num_epochs
        wandb.run.summary['final_learning_rate'] = args.learning_rate  # Use the input learning rate instead
        wandb.run.summary['model_name'] = args.model_name
        wandb.run.summary['dataset_size'] = args.sample_size
    
    # Generate a sample after training
    print("\nGenerating sample after training...")
    post_sample = generate_sample(model, tokenizer, "Nel mezzo del cammin di nostra vita")
    print(f"Sample after training:\n{post_sample}\n")
    
    # Final evaluation with 1 generation per prompt
    print("\nRunning final evaluation...")
    final_eval_metrics = evaluate_model(
        model, tokenizer, eval_data, dc_checker,
        num_generations=1,
        max_completion_length=args.max_completion_length
    )
    
    # Convert keys to indicate these are final evaluation metrics
    renamed_final_metrics = {}
    for key, value in final_eval_metrics.items():
        renamed_final_metrics[f"final_{key}"] = value
    
    # Update final metrics with evaluation results
    final_metrics.update(renamed_final_metrics)
    
    # Save model locally
    print(f"Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Check if this is the best model
    if is_best_model(final_metrics):
        # Upload to HuggingFace Hub if this is the best model and credentials are available
        try:
            from huggingface_hub import HfApi
            
            hf_token = os.environ.get("HF_TOKEN")
            hf_username = os.environ.get("HF_USERNAME")
            
            if hf_token and hf_username:
                repo_id = f"{hf_username}/{run_name}"
                api = HfApi()
                
                print(f"This is the best model so far! Creating repository: {repo_id}")
                api.create_repo(
                    repo_id=repo_id,
                    token=hf_token,
                    private=False,
                    exist_ok=True
                )
                
                print(f"Uploading best model to {repo_id}...")
                api.upload_folder(
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    token=hf_token
                )
                
                print(f"Best model uploaded successfully to: https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"Error uploading model to Hugging Face Hub: {e}")
    else:
        print("This model did not achieve the best reward. Skipping upload to Hugging Face Hub.")
    
    print("Training complete!")

if __name__ == "__main__":
    main()
