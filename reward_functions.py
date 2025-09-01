#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reward Functions for Dante GRPO Training

This module contains the reward functions used to evaluate the quality of generated text
in the Dante GRPO training process.

Author: ruggsea
Date: 2024
"""

import re
import math
import time
import functools
from functools import wraps
import signal
import unicodedata
from statistics import mean
from collections import Counter
import random

from simple_endecasillabo_checker import is_endecasillabo, safe_get_all_syllabifications
from divine_comedy_checker import is_from_divine_comedy

def calculate_repetition_penalty(text, max_length=1024):
    """
    Calculate a penalty for repetitive text.
    Returns a value between -1.0 (highly repetitive) and 0.0 (no repetition).
    Optimized for performance by limiting text length.
    """
    # Handle empty or None input
    if not text:
        return 0.0
        
    # Limit text length for performance
    text = text[:max_length] if len(text) > max_length else text
    
    # Tokenize the text into words
    words = text.lower().split()
    
    if len(words) < 5:
        return 0.0  # Not enough words to calculate repetition
    
    # Calculate n-gram uniqueness for different n values
    uniq_1 = get_ngram_uniqueness(1)(words)
    uniq_2 = get_ngram_uniqueness(2)(words) if len(words) >= 2 else 1.0
    uniq_3 = get_ngram_uniqueness(3)(words) if len(words) >= 3 else 1.0
    
    # Calculate local repetition
    local_rep = local_repetition_penalty()(words)
    
    # Combine the metrics (weighted average)
    combined_score = (uniq_1 + 2*uniq_2 + 3*uniq_3 + 2*local_rep) / 8
    
    # Convert to a penalty between -1.0 and 0.0
    # Lower uniqueness = higher penalty
    penalty = -1.0 * (1.0 - combined_score)
    
    return max(-1.0, min(0.0, penalty))  # Clamp between -1.0 and 0.0

def get_ngram_uniqueness(n):
    """
    Returns a function that calculates the uniqueness of n-grams in a text.
    Higher values (closer to 1.0) indicate less repetition.
    """
    def calculate(words):
        if len(words) < n:
            return 1.0  # Not enough words for n-grams
        
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        unique_ngrams = len(set(ngrams))
        total_ngrams = len(ngrams)
        
        return unique_ngrams / total_ngrams if total_ngrams > 0 else 1.0
    
    return calculate

def local_repetition_penalty():
    """
    Returns a function that calculates repetition within sliding windows.
    Detects local repetitions that might not be caught by global measures.
    """
    def calculate(words):
        if len(words) < 10:
            return 1.0  # Not enough words for meaningful analysis
        
        # Use sliding windows of different sizes
        window_sizes = [5, 10, 15]
        penalties = []
        
        for size in window_sizes:
            if len(words) <= size:
                continue
                
            # Slide window through text
            window_penalties = []
            for i in range(len(words) - size + 1):
                window = words[i:i+size]
                unique_in_window = len(set(window))
                window_penalties.append(unique_in_window / size)
            
            penalties.append(mean(window_penalties))
        
        return mean(penalties) if penalties else 1.0
    
    return calculate

def no_repetition_reward_func(completions, prompts=None, **kwargs) -> list[float]:
    """
    Calculate repetition penalty for each completion.
    Returns a list of penalties between -1.0 (highly repetitive) and 0.0 (no repetition).
    """
    # Handle empty completions list
    if not completions:
        return [0.0]
    
    rewards = []
    
    for i, completion in enumerate(completions):
        try:
            # Handle empty completion
            if not completion or len(completion) == 0:
                rewards.append(0.0)
                continue
                
            # Extract only the completion part, excluding the prompt if provided
            if prompts is not None and i < len(prompts):
                # If the completion starts with the prompt, remove it
                prompt = prompts[i]
                if completion.startswith(prompt):
                    completion = completion[len(prompt):].strip()
            
            # Skip very short completions
            if len(completion) < 20:
                rewards.append(0.0)
                continue
                
            # Calculate repetition penalty
            penalty = calculate_repetition_penalty(completion)
            rewards.append(penalty)
        except Exception as e:
            # If there's an error, assign a neutral penalty
            rewards.append(-0.5)
    
    # Ensure one reward per completion
    while len(rewards) < len(completions):
        rewards.append(0.0)
        
    return rewards

# ============================
# Notebook-style repetition score (continuous [0,1])
# ============================
def _calculate_repetition_score_notebook(text: str) -> float:
    """Replicate the notebook's repetition scoring:
    - Combines n-gram uniqueness, local repetition, and vocab diversity
    - Returns a continuous score in [0,1] where higher is better (less repetition)
    """
    if not text:
        return 0.0

    words = text.lower().split()
    if len(words) < 8:
        return 1.0

    # 1) N-gram uniqueness with extra penalty for heavily repeated n-grams
    def get_ngram_uniqueness(n: int) -> float:
        if len(words) < n:
            return 1.0
        ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
        if not ngrams:
            return 1.0
        counts = Counter(ngrams)
        uniqueness = len(counts) / len(ngrams)
        repetition_penalty = sum(1 for c in counts.values() if c > 2) / len(counts) if counts else 0.0
        return uniqueness * (1.0 - repetition_penalty)

    unigram_score = get_ngram_uniqueness(1)
    bigram_score = get_ngram_uniqueness(2)
    trigram_score = get_ngram_uniqueness(3)
    fourgram_score = get_ngram_uniqueness(4)

    # 2) Local repetition (phrases repeating close to each other)
    def local_repetition_penalty_nb() -> float:
        window_size = 10
        local_repetitions = 0
        for i in range(len(words) - window_size):
            window = words[i:i+window_size]
            window_counts = Counter(window)
            local_repetitions += sum(1 for count in window_counts.values() if count > 2)
        return 1.0 / (1 + local_repetitions)

    local_score = local_repetition_penalty_nb()

    # 3) Vocabulary diversity
    vocab_diversity = len(set(words)) / len(words) if words else 0.0

    # Weighted combination
    weights = {
        'unigram': 0.1,
        'bigram': 0.2,
        'trigram': 0.3,
        'fourgram': 0.2,
        'local': 0.1,
        'vocab': 0.1,
    }

    final_score = (
        weights['unigram'] * unigram_score +
        weights['bigram'] * bigram_score +
        weights['trigram'] * trigram_score +
        weights['fourgram'] * fourgram_score +
        weights['local'] * local_score +
        weights['vocab'] * vocab_diversity
    )

    # Normalize: tanh(2x)/2 + 0.5
    normalized_score = math.tanh(2.0 * final_score) / 2.0 + 0.5
    return max(0.0, min(1.0, float(normalized_score)))

def no_repetition_uniqueness_reward_func(completions, prompts=None, **kwargs) -> list[float]:
    """Vectorized notebook-style repetition score.
    Returns a list of floats in [0,1] (higher is better).
    """
    if not completions:
        return [0.0]

    scores: list[float] = []
    for i, completion in enumerate(completions):
        try:
            text = completion or ""
            # Remove prompt prefix if provided
            if prompts is not None and i < len(prompts):
                prompt = prompts[i] or ""
                if text.startswith(prompt):
                    text = text[len(prompt):].strip()

            scores.append(_calculate_repetition_score_notebook(text))
        except Exception:
            scores.append(0.0)

    while len(scores) < len(completions):
        scores.append(0.0)

    return scores

def check_terzine_structure(text):
    """
    Check if the text has terzine (tercets) structure like in Dante's Divine Comedy.
    A terzina is a group of 3 lines, with empty lines between groups.
    
    Returns a reward based on the following criteria:
    - If fewer than 2 newlines: -0.75 penalty
    - For each line not in a tercet: -0.25 penalty
    - If fewer than 3 valid tercets: -0.5 penalty
    - For each line too short (<5 chars) or too long (>100 chars): -0.2 penalty
    
    Optimized for performance.
    """
    # Handle empty or None input
    if not text:
        return 0.0
        
    # Limit text length for performance
    text = text[:1000] if len(text) > 1000 else text
    
    # Count the number of newlines in the text
    newline_count = text.count('\n')
    
    # Calculate penalty for too few newlines
    newline_penalty = -0.75 if newline_count < 2 else 0.0
    
    # Split the text into non-empty lines
    non_empty_lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Two checks:
    # 1. Check if we have proper tercet structure (empty line after every 3 lines)
    # 2. Check if the groups we have are actually groups of 3 (terzine)
    
    # Check 1: Strict tercet structure check with original text
    lines_with_empties = [line for line in text.split('\n')]
    
    # Find sequences of non-empty lines
    non_empty_sequences = []
    current_seq = []
    
    for line in lines_with_empties:
        if line.strip():
            current_seq.append(line)
        elif current_seq:
            # We found an empty line, end the current sequence
            non_empty_sequences.append(current_seq)
            current_seq = []
    
    # Add final sequence if not empty
    if current_seq:
        non_empty_sequences.append(current_seq)
    
    # if nonempty sequences is empty, return -1
    if not non_empty_sequences:
        return -1
    
    # Count how many sequences have exactly 3 lines (proper tercets)
    proper_tercets = sum(1 for seq in non_empty_sequences if len(seq) == 3)
    
    # Penalty for fewer than 3 valid tercets
    tercet_count_penalty = -0.5 if proper_tercets < 3 else 0.0
    
    # Calculate penalty for non-tercet sequences (groups that don't have 3 lines)
    non_tercet_lines = sum(len(seq) for seq in non_empty_sequences if len(seq) != 3)
    non_tercet_penalty = non_tercet_lines * -0.25
    
    # Penalty for lines that are too short or too long
    length_penalty = 0.0
    for line in non_empty_lines:
        # Changed from 20 to 5 chars minimum - endecasillabi are often shorter than 20 chars
        if len(line.strip()) < 5 or len(line.strip()) > 100:
            length_penalty -= 0.2
    
    # Calculate total penalty
    total_penalty = newline_penalty + tercet_count_penalty + non_tercet_penalty + length_penalty
    
    # Cap the penalty to reasonable bounds
    return max(-2.0, min(1.0, total_penalty))

def verse_reward_func(completions, prompts=None, **kwargs) -> list[float]:
    """
    Check if completions have verse structure.
    Returns a list of rewards between -1.0 (no verse structure) and 1.0 (good verse structure).
    
    For minimal prompts (< 20 chars), includes the prompt as part of the first tercet.
    For longer prompts (complete tercets), excludes the prompt from evaluation.
    """
    # Handle empty completions list
    if not completions:
        return [-1.0]
    
    rewards = []
    
    for i, completion in enumerate(completions):
        try:
            # Handle empty completion
            if not completion or len(completion) == 0:
                rewards.append(-1.0)
                continue
            
            # Get the prompt if available
            prompt = prompts[i] if prompts is not None and i < len(prompts) else ""
            
            # For minimal prompts (< 20 chars), keep the prompt as part of the completion
            # For longer prompts (complete tercets), remove them from evaluation
            if len(prompt.strip()) >= 20 and completion.startswith(prompt):
                completion = completion[len(prompt):].strip()
            
            # Skip if the resulting completion is too short
            if len(completion.split()) < 20:
                rewards.append(-1.0)
                continue
            
            # Check terzine structure
            reward = check_terzine_structure(completion)
            rewards.append(reward)
            
        except Exception as e:
            # If there's an error, assign a neutral reward
            rewards.append(0.0)
    
    # Ensure one reward per completion
    while len(rewards) < len(completions):
        rewards.append(0.0)
    
    return rewards

def timeout(seconds=2):
    """
    Decorator to timeout a function after a specified number of seconds.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def handle_timeout(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            
            # Set the timeout handler
            signal.signal(signal.SIGALRM, handle_timeout)
            signal.alarm(seconds)
            
            try:
                result = func(*args, **kwargs)
            except TimeoutError as e:
                # Return False on timeout
                return False
            finally:
                # Cancel the alarm
                signal.alarm(0)
            
            return result
        return wrapper
    return decorator

@timeout(1)  # Reduce timeout from 2 seconds to 1 second
def safe_is_endecasillabo(line):
    """
    Safely check if a line is an endecasillabo with a timeout.
    """
    return is_endecasillabo(line)

def endecasillabo_reward_func(completions, prompts=None, **kwargs) -> list[float]:
    """
    Calculate reward based on the percentage of valid endecasillabi in the generated text.
    
    Returns a value between 0 and 1 based on:
    - Percentage of valid endecasillabi in the generated lines
    - Applies -0.5 penalty if fewer than 3 valid endecasillabi are found
    
    Args:
        completions: List of generated texts to evaluate
        prompts: Optional list of prompts that were used (to exclude from evaluation)
        
    Returns:
        list[float]: List of rewards between -0.5 and 1.0
    """
    # Handle empty completions list
    if not completions:
        return [0.0]
    
    rewards = []

    # Configuration
    mode = kwargs.get('endeca_mode', 'top_prob')  # 'top_prob' or 'unique_only'
    max_candidates = kwargs.get('max_candidates', 20)
    syllabification_log = kwargs.get('syllabification_log', None)  # optional list to collect details
    
    for i, completion in enumerate(completions):
        try:
            # Handle empty completion
            if not completion or len(completion) == 0:
                rewards.append(0.0)
                continue
            
            # Extract only the completion part, excluding the prompt if provided
            if prompts is not None and i < len(prompts):
                prompt = prompts[i]
                if completion.startswith(prompt):
                    completion = completion[len(prompt):].strip()
            
            
            # Split into lines and filter out empty ones
            lines = [line.strip() for line in completion.split('\n') if line.strip()]

            # Require at least 3 non-empty generated lines; otherwise, make reward unavailable
            if len(lines) < 3:
                rewards.append(-1.0)
                continue
            
            if not lines:
                rewards.append(0.0)
                continue
            
            # Scoring per requested mode
            line_scores = []
            per_line_logs = []
            for line in lines:
                info = {"line": line, "valid": False, "options": []}
                try:
                    ok, options = safe_get_all_syllabifications(line, timeout_seconds=1, max_candidates=max_candidates)
                    if not (ok and options):
                        line_scores.append(0.0)
                        per_line_logs.append(info)
                        continue

                    # Prepare compact options for logging (top 5)
                    info["options"] = [{
                        "syllabification": o["syllabification"],
                        "probability": o["probability"],
                        "syllable_count": o["syllable_count"],
                        "admissible": bool(o["checks"][0] or o["checks"][1])
                    } for o in options[:5]]

                    if mode == 'unique_only':
                        # Valid only if exactly one option and it's admissible
                        only = options[0]
                        is_adm = bool(only["checks"][0] or only["checks"][1])
                        score = 1.0 if (len(options) == 1 and is_adm) else 0.0
                        info["valid"] = bool(score == 1.0)
                    else:  # 'top_prob'
                        # Find best admissible option (already sorted desc)
                        top_adm = next((o for o in options if (o["checks"][0] or o["checks"][1])), None)
                        top_p = float(top_adm["probability"]) if top_adm is not None else 0.0
                        # Use the raw probability of the top admissible option (no normalization by sum)
                        score = top_p
                        info["valid"] = bool(top_adm is not None)

                    line_scores.append(score)
                    per_line_logs.append(info)
                except Exception:
                    line_scores.append(0.0)
                    per_line_logs.append(info)

            if syllabification_log is not None and isinstance(syllabification_log, list):
                syllabification_log.append({
                    "completion": completion,
                    "lines": per_line_logs
                })

            if not line_scores:
                rewards.append(0.0)
                continue

            # Final reward is mean of per-line scores in [0,1]
            reward = sum(line_scores) / len(line_scores)
            rewards.append(reward)
            
        except Exception as e:
            # If there's an error, assign a neutral reward
            rewards.append(0.0)
    
    # Ensure one reward per completion
    while len(rewards) < len(completions):
        rewards.append(0.0)
    
    return rewards

def check_divine_comedy_plagiarism(completions, prompts=None, **kwargs) -> list[float]:
    """
    Check if completions plagiarize the Divine Comedy.
    Returns a list of penalties (-1.0 for plagiarism, 0.0 otherwise).
    Optimized for performance using hash-based matching.
    """
    # Handle empty completions list
    if not completions:
        return [0.0]
    
    rewards = []
    
    # Get the dc_checker instance from kwargs
    dc_checker = kwargs.get('dc_checker', None)
    if dc_checker is None:
        # If no checker is provided, return neutral rewards
        return [0.0] * len(completions)
    
    for i, completion in enumerate(completions):
        try:
            # Handle empty completion
            if not completion or len(completion) == 0:
                rewards.append(0.0)
                continue
                
            # Extract only the completion part, excluding the prompt if provided
            if prompts is not None and i < len(prompts):
                # If the completion starts with the prompt, remove it
                prompt = prompts[i]
                if completion.startswith(prompt):
                    completion = completion[len(prompt):].strip()
            
            # Skip very short completions
            if len(completion) < 20:
                rewards.append(0.0)
                continue
                
            # Split into lines
            lines = [line.strip() for line in completion.split('\n') if line.strip()]
            
            if not lines:
                rewards.append(0.0)
                continue
            
            # Check only a sample of lines for plagiarism (at most 5)
            # Prioritize longer lines which are more likely to be distinctive
            lines_to_check = sorted(lines, key=len, reverse=True)[:5]
            
            # Check each line for plagiarism
            plagiarism_detected = False
            for line in lines_to_check:
                # Skip very short lines (less than 5 words)
                if len(line.split()) < 5:
                    continue
                
                # Check if line is from Divine Comedy using the provided checker instance
                # Use exact match only - much faster with the hash-based approach
                is_exact_match, _ = dc_checker.is_from_divine_comedy(line, exact_match=True, verbose=False)
                if is_exact_match:
                    plagiarism_detected = True
                    break
            
            # Apply penalty if plagiarism detected
            reward = -1.0 if plagiarism_detected else 0.0
            rewards.append(reward)
        except Exception as e:
            # If there's an error, assign a neutral penalty
            rewards.append(-0.5)
    
    # Ensure one reward per completion
    while len(rewards) < len(completions):
        rewards.append(0.0)
        
    return rewards

def test_endecasillabo_reward_func():
    # Test with a valid endecasillabo from Dante
    verse = "Nel mezzo del cammin di nostra vita"
    is_valid, syllabification = safe_is_endecasillabo(verse)
    print(f"Verse: '{verse}'")
    print(f"Is Endecasillabo: {is_valid}")
    print(f"Syllabification: {syllabification}")
    verse = "Nel mezzo del cammin di nostra dsds"
    is_valid, syllabification = safe_is_endecasillabo(verse)
    print(f"Verse: '{verse}'")
    print(f"Is Endecasillabo: {is_valid}")
    print(f"Syllabification: {syllabification}")
    verse = "Nel mezzo del cammin di nostra dsds"
    reward = endecasillabo_reward_func([verse])
    print(f"Reward: {reward}")
    verse = ""
    reward = endecasillabo_reward_func([verse, verse, verse, verse, verse, verse, verse, verse, verse, verse, verse, verse])
    print(f"Reward: {reward}")
if __name__ == "__main__":
    test_endecasillabo_reward_func()