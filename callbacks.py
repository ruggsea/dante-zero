#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Callbacks for Dante GRPO Training

This module contains the callback classes used in the Dante GRPO training process.

Author: ruggsea
Date: 2024
"""

import wandb
import os
import functools
from transformers.trainer_callback import TrainerCallback
from transformers import ProcessorMixin
import torch
import html as html_lib
from simple_endecasillabo_checker import safe_get_all_syllabifications

class TokenizerWrapper(ProcessorMixin):
    """
    Wrapper for tokenizer to handle GRPO's specific requirements.
    """
    def __init__(self, base_tokenizer):
        self.base_tokenizer = base_tokenizer
        # Expose fields commonly expected on ProcessorMixin processors
        # Point tokenizer to self so downstream accesses (e.g., .tokenizer.decode)
        # resolve to this wrapper, which delegates to the base tokenizer.
        self.tokenizer = self
        self.feature_extractor = None
        self.image_processor = None
        
        # Copy attributes from base tokenizer
        self.pad_token = base_tokenizer.pad_token
        self.pad_token_id = base_tokenizer.pad_token_id
        self.eos_token = base_tokenizer.eos_token
        self.eos_token_id = base_tokenizer.eos_token_id
        
        # Set padding side to left for proper handling in GRPO
        self.padding_side = "left"
        # Also set it on the base tokenizer
        self.base_tokenizer.padding_side = "left"
        
        # Copy all other attributes from base tokenizer
        for attr_name in dir(base_tokenizer):
            if not attr_name.startswith('__') and not hasattr(self, attr_name):
                try:
                    setattr(self, attr_name, getattr(base_tokenizer, attr_name))
                except (AttributeError, TypeError):
                    pass
    
    def __call__(self, *args, **kwargs):
        # Forward to base tokenizer (padding side already set on base)
        result = self.base_tokenizer(*args, **kwargs)
        
        # Ensure input_ids are of type Long (int64)
        if isinstance(result, dict) and 'input_ids' in result:
            if torch.is_tensor(result['input_ids']):
                result['input_ids'] = result['input_ids'].long()
        
        return result
    
    def batch_decode(self, *args, **kwargs):
        # Forward to base tokenizer
        return self.base_tokenizer.batch_decode(*args, **kwargs)
    
    def decode(self, *args, **kwargs):
        # Forward to base tokenizer
        return self.base_tokenizer.decode(*args, **kwargs)

    def __getattr__(self, name):
        # Delegate missing attributes/methods to the base tokenizer
        return getattr(self.base_tokenizer, name)

    # Avoid saving None feature_extractor/image_processor via ProcessorMixin
    def save_pretrained(self, save_directory, **kwargs):
        try:
            if hasattr(self.base_tokenizer, 'save_pretrained'):
                return self.base_tokenizer.save_pretrained(save_directory, **kwargs)
        except Exception:
            pass
        return None

class WandbCompletionCallback(TrainerCallback):
    """
    Callback to log example completions to Weights & Biases.
    """
    def __init__(self, tokenizer, log_steps=500):
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.log_steps = log_steps
        self.last_log_step = 0
    
    def on_init_end(self, args, state, control, **kwargs):
        """
        Initialize callback at the start of training.
        """
        self.last_log_step = 0
    
    def on_step_end(self, args, state, control, **kwargs):
        """
        Log example completions at regular intervals.
        """
        # Dynamically set frequency: log every ~1% of total steps if available
        total_steps = getattr(state, 'max_steps', None) or getattr(state, 'num_train_epochs', None)
        try:
            # Prefer max_steps if present
            if hasattr(state, 'max_steps') and state.max_steps and state.max_steps > 0:
                dynamic_log_steps = max(1, state.max_steps // 100)
            else:
                dynamic_log_steps = self.log_steps
        except Exception:
            dynamic_log_steps = self.log_steps

        # Check if it's time to log
        if state.global_step - self.last_log_step >= dynamic_log_steps:
            try:
                # Get model and generation kwargs from trainer
                model = kwargs.get("model")
                generation_kwargs = kwargs.get("generation_kwargs", {})
                
                if model and hasattr(model, "generate") and hasattr(model, "prepare_inputs_for_generation") and (wandb.run is not None):
                    # Create a prompt with instruction + real-starter
                    prompt = "Scrivi delle terzine di endecasillabi in stile dantesco\n\nNel mezzo del cammin di nostra vita\n"
                    inputs = self.tokenizer(prompt, return_tensors="pt", padding_side="left").to(model.device)

                    # Generate text
                    outputs = model.generate(
                        inputs.input_ids,
                        max_new_tokens=200,
                        do_sample=True,
                        temperature=0.9,
                        top_p=0.9,
                        repetition_penalty=1.2,
                        **generation_kwargs
                    )

                    # Decode full text and prompt
                    full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    prompt_text = self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
                    completion_text = full_text[len(prompt_text):] if full_text.startswith(prompt_text) else full_text

                    # Build syllabification side panel for completion lines
                    completion_lines = [ln.strip() for ln in completion_text.split('\n') if ln.strip()]
                    side_rows = []
                    for ln in completion_lines:
                        ok, opts = safe_get_all_syllabifications(ln, timeout_seconds=1, max_candidates=5)
                        top_adm = None
                        if ok and opts:
                            for o in opts:
                                if o["checks"][0] or o["checks"][1]:
                                    top_adm = o
                                    break
                        if top_adm is not None:
                            side_rows.append((ln, top_adm["syllabification"], float(top_adm["probability"])))

                    esc_prompt = html_lib.escape(prompt_text)
                    esc_completion = html_lib.escape(completion_text)

                    # Prebuild syllabification rows to avoid backslashes in f-string expressions
                    rows_html_parts = []
                    for (l, s, p) in side_rows:
                        rows_html_parts.append(
                            "<div style='margin-bottom:6px;'>"
                            + f"<div style='color:#555;'>{html_lib.escape(l)}</div>"
                            + f"<pre style='margin:2px 0; background:#f5fff5; padding:4px;'>{html_lib.escape(s)}</pre>"
                            + f"<div style='font-size:12px;color:#666;'>p={p:.3f}</div>"
                            + "</div>"
                        )
                    rows_html = "".join(rows_html_parts)

                    # Construct two-column HTML with completion highlighted
                    html = f"""
<div style=\"display:flex; gap:16px; align-items:flex-start; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;\">
  <div style=\"flex:2; min-width:0;\">
    <pre style=\"margin:0; white-space:pre-wrap; word-wrap:break-word;\">{esc_prompt}<span style=\"background:#d7ffd9\">{esc_completion}</span></pre>
  </div>
  <div style=\"flex:1; min-width:220px; border-left:1px solid #eee; padding-left:12px;\">
    <div style=\"font-weight:bold; margin-bottom:6px;\">Sillabificazione (top ammessa)</div>
    {rows_html}
  </div>
</div>
"""

                    # Log to wandb
                    wandb.log({
                        "example_completion": wandb.Html(html),
                        "step": state.global_step
                    })
                    
                    # Update last log step
                    self.last_log_step = state.global_step
            
            except Exception as e:
                # Silent fail to avoid interrupting training
                pass
        
        return control

class BestCompletionsCallback(TrainerCallback):
    """
    Callback to track the best completions based on reward functions.
    """
    def __init__(self, tokenizer, reward_funcs, reward_names=None, top_k=5, log_steps=None, dc_checker=None, endeca_mode='top_prob', max_candidates=20, endeca_weight=1.0):
        self.tokenizer = tokenizer
        self.original_reward_funcs = reward_funcs
        self.reward_names = reward_names or [f"Reward {i}" for i in range(len(reward_funcs))]
        self.top_k = top_k
        self.best_completions = []
        self.max_stored_completions = 100  # Limit the number of stored completions
        self.log_steps = log_steps or 100  # Only process completions every 100 steps by default
        self.last_processed_step = 0
        self.dc_checker = dc_checker
        self.endeca_mode = endeca_mode
        self.max_candidates = max_candidates
        self.endeca_weight = float(endeca_weight)
        # Track per-epoch rewards to identify the best epoch by average total reward
        self.epoch_reward_sum = 0.0
        self.epoch_reward_count = 0
        self.best_epoch_avg_reward = float('-inf')
        self.best_saved_dir = None
        
        # Current batch data
        self.current_batch_data = {
            "prompts": None,
            "completions": None,
            "rewards": [[] for _ in range(len(reward_funcs))],
            "batch_collected": False
        }
        
        self.current_func_index = -1
        self.current_epoch = 0
    
    def _create_wrapped_func(self, func, index):
        """
        Create a wrapped reward function that captures completions and rewards.
        """
        @functools.wraps(func)
        def wrapped_func(completions, prompts=None, **kwargs):
            # Safety check - ensure we're not trying to process empty completions
            if not completions or len(completions) == 0:
                return [0.0]  # Return a default reward
            
            # Store completions and prompts if this is the first reward function
            if self.current_func_index < index:
                self.current_func_index = index
                self.current_batch_data["completions"] = completions
                self.current_batch_data["prompts"] = prompts
            
            # Add dc_checker to kwargs if available and not already present
            if self.dc_checker is not None and 'dc_checker' not in kwargs:
                kwargs['dc_checker'] = self.dc_checker
            # Inject endecasillabo config for the corresponding reward function
            if getattr(func, '__name__', '') == 'endecasillabo_reward_func':
                kwargs.setdefault('endeca_mode', self.endeca_mode)
                kwargs.setdefault('max_candidates', self.max_candidates)
            
            # Call the original reward function
            rewards = func(completions, prompts, **kwargs)
            # Apply endecasillabo weight if this is the endeca reward
            if getattr(func, '__name__', '') == 'endecasillabo_reward_func' and self.endeca_weight != 1.0:
                try:
                    rewards = [r * self.endeca_weight for r in rewards]
                except Exception:
                    pass
            
            # Store rewards
            self.current_batch_data["rewards"][index] = rewards
            
            # Mark batch as collected if this is the last reward function
            if index == len(self.original_reward_funcs) - 1:
                self.current_batch_data["batch_collected"] = True
            
            return rewards
        
        return wrapped_func
    
    def get_wrapped_reward_funcs(self):
        """
        Get wrapped versions of all reward functions.
        """
        return [self._create_wrapped_func(func, i) for i, func in enumerate(self.original_reward_funcs)]
    
    def _extract_completion_only(self, prompt, full_text):
        """
        Extract the completion part from the full text.
        """
        if prompt and full_text.startswith(prompt):
            return full_text[len(prompt):]
        return full_text
    
    def on_init_end(self, args, state, control, **kwargs):
        """
        Initialize callback at the start of training.
        """
        self.best_completions = []
    
    def on_step_begin(self, args, state, control, **kwargs):
        """
        Reset batch data at the start of each step.
        """
        self.current_batch_data["prompts"] = None
        self.current_batch_data["completions"] = None
        self.current_batch_data["rewards"] = [[] for _ in range(len(self.original_reward_funcs))]
        self.current_batch_data["batch_collected"] = False
        self.current_func_index = -1
        
        return control
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """
        Track the current epoch and reset best completions at the start of each epoch.
        """
        self.current_epoch = state.epoch
        self.best_completions = []
        # reset epoch aggregation
        self.epoch_reward_sum = 0.0
        self.epoch_reward_count = 0
        
        return control
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """
        Log the best completions at the end of each epoch.
        """
        try:
            # Save best-by-reward checkpoint if improved
            try:
                avg_reward = (self.epoch_reward_sum / self.epoch_reward_count) if self.epoch_reward_count > 0 else float('-inf')
                model = kwargs.get('model')
                if avg_reward > self.best_epoch_avg_reward and model is not None and hasattr(model, 'save_pretrained'):
                    best_dir = os.path.join(args.output_dir, 'best')
                    os.makedirs(best_dir, exist_ok=True)
                    try:
                        model.save_pretrained(best_dir)
                    except Exception:
                        pass
                    # Try to save tokenizer as well
                    try:
                        if hasattr(self.tokenizer, 'save_pretrained'):
                            self.tokenizer.save_pretrained(best_dir)
                    except Exception:
                        pass
                    self.best_epoch_avg_reward = avg_reward
                    self.best_saved_dir = best_dir
            except Exception as _e:
                # Never break training for checkpoint save issues
                pass

            # Sort best completions by total reward
            sorted_completions = sorted(self.best_completions, key=lambda x: x['total_reward'], reverse=True)
            
            # Take top_k
            top_completions = sorted_completions[:self.top_k]
            
            # Log to wandb using one HTML object per best completion (like example callback)
            if wandb.run is not None and top_completions:
                for idx, comp in enumerate(top_completions, start=1):
                    prompt_text = comp.get('prompt', '') or ''
                    completion_text = comp.get('completion', '') or ''
                    full_text = comp.get('full_text', prompt_text + completion_text) or (prompt_text + completion_text)

                    # Build syllabification side panel for completion lines
                    completion_lines = [ln.strip() for ln in completion_text.split('\n') if ln.strip()]
                    side_rows = []
                    for ln in completion_lines:
                        ok, opts = safe_get_all_syllabifications(ln, timeout_seconds=1, max_candidates=5)
                        top_adm = None
                        if ok and opts:
                            for o in opts:
                                if o["checks"][0] or o["checks"][1]:
                                    top_adm = o
                                    break
                        if top_adm is not None:
                            side_rows.append((ln, top_adm["syllabification"], float(top_adm["probability"])))

                    esc_prompt = html_lib.escape(prompt_text)
                    esc_completion = html_lib.escape(completion_text)

                    # Prebuild syllabification rows HTML
                    rows_html_parts = []
                    for (l, s, p) in side_rows:
                        rows_html_parts.append(
                            "<div style='margin-bottom:6px;'>"
                            + f"<div style='color:#555;'>{html_lib.escape(l)}</div>"
                            + f"<pre style='margin:2px 0; background:#f5fff5; padding:4px;'>{html_lib.escape(s)}</pre>"
                            + f"<div style='font-size:12px;color:#666;'>p={p:.3f}</div>"
                            + "</div>"
                        )
                    rows_html = "".join(rows_html_parts)

                    # Construct two-column HTML with completion highlighted
                    html = f"""
<div style=\"display:flex; gap:16px; align-items:flex-start; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;\">
  <div style=\"flex:2; min-width:0;\">
    <pre style=\"margin:0; white-space:pre-wrap; word-wrap:break-word;\">{esc_prompt}<span style=\"background:#d7ffd9\">{esc_completion}</span></pre>
    <div style=\"margin-top:6px; font-size:12px; color:#666;\">Total reward: {comp['total_reward']:.4f}</div>
  </div>
  <div style=\"flex:1; min-width:220px; border-left:1px solid #eee; padding-left:12px;\">
    <div style=\"font-weight:bold; margin-bottom:6px;\">Sillabificazione (top ammessa)</div>
    {rows_html}
  </div>
</div>
"""
                    wandb.log({
                        "best_completion": wandb.Html(html),
                        "epoch": self.current_epoch,
                        "rank": idx
                    })
        
        except Exception as e:
            # Silent fail to avoid interrupting training
            print(f"Error in BestCompletionsCallback.on_epoch_end: {e}")
        
        return control
    
    def on_step_end(self, args, state, control, **kwargs):
        """
        Process completions and update best completions list.
        """
        try:
            # Check if we have collected a batch
            if not self.current_batch_data["batch_collected"]:
                return control
                
            # Only process completions periodically to reduce overhead
            if state.global_step - self.last_processed_step < self.log_steps:
                return control
                
            self.last_processed_step = state.global_step
            
            # Get data from current batch
            completions = self.current_batch_data["completions"]
            prompts = self.current_batch_data["prompts"]
            rewards_list = self.current_batch_data["rewards"]
            
            # Process completions if available
            if completions and len(completions) > 0:
                # Calculate total reward for each completion
                total_rewards = []
                individual_rewards_list = []
                
                for i in range(len(completions)):
                    # Get individual rewards for this completion
                    individual_rewards = []
                    for reward_idx in range(len(rewards_list)):
                        if i < len(rewards_list[reward_idx]):
                            individual_rewards.append(rewards_list[reward_idx][i])
                        else:
                            individual_rewards.append(0.0)
                    
                    # Calculate total reward
                    total_reward = sum(individual_rewards)
                    
                    # Store rewards
                    total_rewards.append(total_reward)
                    individual_rewards_list.append(individual_rewards)
                
                # Add completions to best completions list
                for i in range(len(completions)):
                    prompt = prompts[i] if prompts is not None and i < len(prompts) else ""
                    completion = completions[i]
                    
                    # Extract just the completion part
                    completion_only = self._extract_completion_only(prompt, completion)
                    
                    # Add to best completions list
                    self.best_completions.append({
                        'prompt': prompt,
                        'completion': completion_only,
                        'full_text': completion,
                        'total_reward': total_rewards[i],
                        'individual_rewards': individual_rewards_list[i],
                        'reward_names': self.reward_names
                    })
                
                # Keep only the top completions to limit memory usage
                self.best_completions = sorted(
                    self.best_completions, 
                    key=lambda x: x['total_reward'], 
                    reverse=True
                )[:self.max_stored_completions]

                # Update epoch reward aggregates
                try:
                    self.epoch_reward_sum += float(sum(total_rewards))
                    self.epoch_reward_count += int(len(total_rewards))
                except Exception:
                    pass
            
        except Exception as e:
            # Silent fail to avoid interrupting training
            pass
        
        return control 