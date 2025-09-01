#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utilities for Dante GRPO Training

This module contains utility functions used in the Dante GRPO training process.

Author: ruggsea
Date: 2024
"""

import os
import random
import datetime
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

def create_dataset(sample_size, dc_checker):
    """
    Create a dataset of prompts from the Divine Comedy.
    
    Args:
        sample_size: Number of samples to include in the dataset
        dc_checker: Instance of DivineComedyChecker
        
    Returns:
        Dataset object with prompts
    """
    try:
        # Load original verses from the Divine Comedy files
        all_verses = []
        cantica_paths = [
            os.path.join('Dante', 'inferno.txt'),
            os.path.join('Dante', 'purgatorio.txt'),
            os.path.join('Dante', 'paradiso.txt')
        ]
        
        for path in cantica_paths:
            try:
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        current_verses = []
                        for line in f:
                            line = line.strip()
                            # Skip empty lines and canto headers
                            if line and not line.startswith('CANTO') and '•' not in line:
                                current_verses.append(line)
                        all_verses.extend(current_verses)
                    print(f"Loaded verses from {path}")
            except Exception as e:
                print(f"Error loading verses from {path}: {e}")
        
        if not all_verses:
            raise Exception("No verses loaded from Divine Comedy files")
        
        # Create prompts by sampling complete tercets
        prompt_list = []
        
        # Prepare tokenizer for token-length checks (enforce <= 256 tokens per prompt)
        try:
            tok_model = os.environ.get('PROMPT_TOKENIZER', 'PleIAs/Pleias-350m-Preview')
            count_tokenizer = AutoTokenizer.from_pretrained(tok_model, padding_side="left")
            if count_tokenizer.pad_token is None:
                count_tokenizer.pad_token = count_tokenizer.eos_token
        except Exception:
            count_tokenizer = None  # fallback to naive counting

        def token_len(text: str) -> int:
            if count_tokenizer is not None:
                try:
                    return len(count_tokenizer(text, add_special_tokens=False)["input_ids"])
                except Exception:
                    pass
            # naive fallback
            return len(text.split())

        # Create prompts for the entire dataset (no minimal/full split). Each prompt has 1–5 full tercets, mean≈2.
        instruction = "Scrivi delle terzine di endecasillabi in stile dantesco\n\n"
        target_count = sample_size
        generated = 0
        while generated < target_count:
            if not all_verses:
                break
            # Sample number of tercets from clipped normal ~N(2.0, 0.8) to [1,5]
            sampled = int(round(random.gauss(2.0, 0.8)))
            num_tercets = max(1, min(5, sampled))
            if len(all_verses) < num_tercets * 3:
                continue
            max_start = len(all_verses) - (num_tercets * 3)
            if max_start < 0:
                continue
            start_idx = random.randint(0, max_start)
            start_idx = (start_idx // 3) * 3
            verses = all_verses[start_idx:start_idx + (num_tercets * 3)]
            tercets = ["\n".join(verses[i:i+3]) for i in range(0, len(verses), 3)]
            prompt = instruction + "\n\n".join(tercets) + "\n"
            # Trim tercets to respect 256-token budget
            while token_len(prompt) > 256 and len(tercets) > 1:
                tercets = tercets[:-1]
                prompt = instruction + "\n\n".join(tercets) + "\n"
            if token_len(prompt) > 256:
                continue
            prompt_list.append(prompt)
            generated += 1
        
        # Verify no empty prompts
        for i, prompt in enumerate(prompt_list):
            if not prompt or prompt.strip() == "":
                # If somehow we get an empty prompt, replace it with a classic opening
                prompt_list[i] = "Nel mezzo\n"
        
        # Shuffle the prompts
        random.shuffle(prompt_list)
        
        dataset = Dataset.from_dict({'prompt': prompt_list})
        print(f"Created dataset with {len(dataset)} examples (1–5 terzine per prompt, mean≈2)")
        
        return dataset
    
    except Exception as e:
        print(f"Error creating dataset: {e}")
        # Fallback to a small dataset with properly structured examples
        prompt_list = [
            "Nel mezzo del cammin di nostra vita\nmi ritrovai per una selva oscura\nché la diritta via era smarrita\n",
            "La gloria di colui che tutto move\nper l'universo penetra, e risplende\nin una parte più e meno altrove\n",
            "O somma luce che tanto ti levi\ndai concetti mortali, a la mia mente\nripresta un poco di quel che parevi\n"
        ]
        dataset = Dataset.from_dict({'prompt': prompt_list})
        print(f"Using fallback dataset with {len(dataset)} examples")
        
        return dataset

def create_eval_dataset(sample_size, dc_checker):
    """
    Create a separate evaluation dataset with prompts similar to training but using different examples.
    
    Args:
        sample_size: Number of samples for evaluation
        dc_checker: Divine Comedy checker
        
    Returns:
        List of evaluation examples
    """
    # Get all available verses
    all_verses = dc_checker.get_all_verses()
    
    # Shuffle to get different examples than training
    random.shuffle(all_verses)
    
    # Create evaluation examples (format same as training data)
    eval_data = []
    
    # Add examples with minimal prompts
    minimal_prompts = ["Nel mezzo ", "Di quelle ", "La gloria ", "O somma ", "Vergine "]
    for prompt in minimal_prompts[:min(len(minimal_prompts), sample_size // 5)]:
        eval_data.append({
            "text": prompt,
            "type": "minimal"
        })
    
    # Add examples with full tercet prompts
    num_tercet_examples = sample_size - len(eval_data)
    verse_index = 0
    
    while len(eval_data) < sample_size and verse_index + 3 < len(all_verses):
        tercet = "\n".join(all_verses[verse_index:verse_index+3])
        eval_data.append({
            "text": tercet,
            "type": "tercet"
        })
        verse_index += 3
    
    print(f"Created evaluation dataset with {len(eval_data)} examples")
    print(f"- Minimal prompts: ~{len(eval_data) - num_tercet_examples} examples")
    print(f"- Full tercet prompts: ~{num_tercet_examples} examples")
    
    return eval_data

def generate_run_name(model_name, run_name=None):
    """
    Generate a unique run name based on date and model name.
    
    Args:
        model_name: Name of the model being trained
        run_name: Optional custom run name
        
    Returns:
        Unique run name string
    """
    if run_name:
        return run_name
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    model_short_name = model_name.split("/")[-1] if "/" in model_name else model_name
    return f"dante-zero-{timestamp}-{model_short_name}"

def create_model_readme(model_name, num_epochs, username, run_name, repo_id):
    """
    Create a README.md file for the model.
    
    Args:
        model_name: Name of the base model
        num_epochs: Number of training epochs
        username: Hugging Face username
        run_name: Run name for the training
        repo_id: Repository ID for the model
        
    Returns:
        README content as a string
    """
    return f"""# Dante-Zero Fine-tuned Model

This model was fine-tuned using Reinforcement Learning with Generative Pre-trained Transformer Optimization (GRPO) to generate Dante-style poetry in endecasillabi (11-syllable lines).

## Model Details

- **Base Model:** {model_name}
- **Training Method:** GRPO (Generative Pre-trained Transformer Optimization)
- **Training Data:** 1,000 chunks from Dante's Divine Comedy
- **Epochs:** {num_epochs}
- **Trained By:** {username}
- **Date:** {datetime.datetime.now().strftime("%Y-%m-%d")}
- **Run Name:** {run_name}

## Model Description

This model is specialized in generating Italian poetry in the style of Dante Alighieri's Divine Comedy. It has been trained to:

1. Generate proper endecasillabi (11-syllable lines)
2. Follow the structure of Dante's poetry
3. Avoid repetition
4. Create original content (not plagiarize the Divine Comedy)

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}", padding_side="left")

# Ensure proper tokenizer settings
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# Generate poetry
prompt = "Nel mezzo del cammin di nostra vita"
inputs = tokenizer(prompt, return_tensors="pt", padding_side="left")
outputs = model.generate(
    inputs.input_ids,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2
)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

## Reward Functions

The model was trained using several reward functions:

1. **Endecasillabo Checker:** Rewards proper 11-syllable lines
2. **Plagiarism Checker:** Penalizes copying from the Divine Comedy
3. **Verse Structure Checker:** Encourages verse-like structure
4. **Repetition Penalty:** Discourages repetitive text
""" 