## Dante-Zero

This project uses Reinforcement Learning with Generative Pre-trained Transformer Optimization (GRPO) to fine-tune a language model that can generate Dante-like poetry in endecasillabi (11-syllable lines).

### Overview
The model is trained to:
1. Follow the metric structure (hendecasyllabic meter and tercet grouping)
2. Produce proper endecasillabi (11-syllable lines)
3. Avoid repetition
4. Not plagiarize the original Divine Comedy

### Training Data
We sample 1,000 chunks from the Divine Comedy (1–5 tercets per prompt) for training prompts. Prompts are prefixed with a short Italian instruction (e.g., "Scrivi delle terzine di endecasillabi in stile dantesco") followed by the sampled tercets to encourage natural continuation.

### Requirements
- Python 3.10+ (CUDA recommended)
- Install deps with uv:
  - `uv venv .venv && . .venv/bin/activate && uv pip install -r requirements.txt`

### External dependency (Dante syllabifier)
We rely on the Dante syllabification algorithm to verify hendecasyllables. You need the code and data from the original project:
- Repo: [`asperti/Dante`](https://github.com/asperti/Dante)
- Paper: [`https://arxiv.org/abs/2010.13515`](https://arxiv.org/abs/2010.13515)

Place the following under `Dante/` in this repo:
- `dante.py` (syllabifier code)
- `dantes_dictionary.pkl` (dictionary required by the algorithm)
- Texts: `inferno.txt`, `purgatorio.txt`, `paradiso.txt` (used for prompts and plagiarism checks)

The syllabifier is invoked by `simple_endecasillabo_checker.py`.

### Setup (.env)
Create `.env` in project root:
- `HF_TOKEN`: Hugging Face token
- `HF_USERNAME`: Your HF username
- `WANDB_API_KEY`: Weights & Biases API key
- `WANDB_PROJECT`: e.g. `dante-zero`
- `CHECKPOINT_LOCATION`: Absolute path for checkpoints/outputs

### Running the Training
Default single run:
```bash
. .venv/bin/activate
python dante_grpo.py
```

#### Command-line Arguments (subset)
- `--model_name` (default: `PleIAs/Pleias-350m-Preview`)
- `--num_epochs` (default: 10)
- `--batch_size`, `--gradient_accumulation_steps`
- `--max_prompt_length`, `--max_completion_length`
- `--num_generations`, `--sample_size`
- `--endeca_mode` in `{top_prob, unique_only}`
- `--use_repetition_reward` in `{0,1}`
- `--gen_repetition_penalty` (e.g. 1.0–1.3)
- `--kl_beta` (e.g. 0.01–0.10)
- `--endeca_weight` (default 5.0)

### Reward Functions

The training uses several reward functions:

1. **Endecasillabo Checker**: Rewards the model based on the ratio of proper 11-syllable lines (endecasillabi) to total lines generated. This reward is multiplied by 5 (default `--endeca_weight`) to emphasize its importance, resulting in values from 0.0 to 5.0. A timeout mechanism (≈1 second per line) prevents the syllabification process from hanging on complex or malformed lines.

2. **Divine Comedy Plagiarism Checker**: Penalizes exact copying from the Divine Comedy with a −1.0 penalty. The checker uses a fast, hash-based exact-match on sampled longer lines; very short lines are excluded to avoid false positives with common phrases.

3. **Verse Structure Checker**: Encourages verse-like (tercet) structure by applying penalties for too-few lines, non-tercet groupings, and lines that are too short or too long.

4. **Repetition Penalty** (optional): Applies negative penalties (between −1.0 and 0.0) for repetitive text patterns. The penalty is calculated using a combination of:
   - N-gram uniqueness (unigrams, bigrams, trigrams, four-grams)
   - Local repetition within sliding windows
   - Overall vocabulary diversity
   
   The score is normalized to be more sensitive in the mid-range, ensuring that even subtle repetition patterns are penalized. The alternative repetition reward in [0,1] is disabled in this repo.

All reward functions are applied only to the generated completions, excluding the prompt text, to ensure fair evaluation of the model's outputs.

### Monitoring
Training logs to W&B; example completions and best-of-epoch are logged with syllabification side panels. Checkpoints are saved under `${CHECKPOINT_LOCATION}/{run_name}/{best|final}/`.

### Sweep Orchestration (64 runs)
```bash
. .venv/bin/activate
python orchestrator.py sweep --model-name PleIAs/Pleias-350m-Preview --gpus 0,3 --threshold-mb 1024 --poll-seconds 1
```
Grid: repetition reward on/off × endeca mode (top_prob/unique_only) × repetition_penalty [1.0,1.1,1.2,1.3] × KL beta [0.01,0.02,0.05,0.10].
Manifests in `manifests/`; use `repair` to mark completed/failed from checkpoints; resume by passing `--manifest`.

### Evaluation
1) Build a cached 100-prompt eval set (constructed like training prompts, deterministic).
2) For each checkpoint (best/final):
   - Generate one completion per prompt with fixed decoding settings (kept constant across models).
   - Compute endecasillabo accuracy: for each completion, count fraction of lines that are admissible per Dante syllabifier; aggregate mean across 100 prompts (0–100%).
3) LLM-as-judge pairwise matchups across models on a shared subset of prompts:
   - Requires a local OpenAI-compatible LLM server (e.g., vLLM). By default, the code targets `http://127.0.0.1:8008` and requests up to 16K tokens to avoid truncation.
   - The judge provides a chain-of-thought internally; on retry it returns a minimal JSON `{"choice":"A|B"}` for unambiguous scoring.
4) Rank models via Bradley–Terry on the pairwise wins/losses, and report both the judge-based ranking and the syllabification accuracy.

To verify a running judge:
```bash
curl -s http://127.0.0.1:8008/v1/models
```

### License
MIT. The syllabifier is from [`asperti/Dante`](https://github.com/asperti/Dante); see their paper at [`https://arxiv.org/abs/2010.13515`](https://arxiv.org/abs/2010.13515).