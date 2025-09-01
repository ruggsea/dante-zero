import os
import shutil
import types
from callbacks import BestCompletionsCallback


class DummyArgs:
    def __init__(self, output_dir):
        self.output_dir = output_dir


class DummyState:
    def __init__(self, epoch=1, global_step=100):
        self.epoch = epoch
        self.global_step = global_step


class DummyControl:
    pass


class DummyTokenizer:
    def __init__(self):
        self.padding_side = "left"
    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "tokenizer.json"), "w") as f:
            f.write("{}")


class DummyModel:
    def __init__(self):
        self.saved = []
    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        self.saved.append(path)
        with open(os.path.join(path, "pytorch_model.bin"), "wb") as f:
            f.write(b"dummy")


def const_reward(v):
    def rf(completions, prompts=None, **kwargs):
        return [v for _ in completions]
    return rf


def test_best_checkpoint_saved_on_improvement(tmp_path):
    outdir = tmp_path / "run"
    os.makedirs(outdir, exist_ok=True)
    tok = DummyTokenizer()
    # Two reward functions returning constants; total per completion will be their sum
    cb = BestCompletionsCallback(tok, [const_reward(1.0), const_reward(2.0)], reward_names=["r1","r2"], top_k=2, log_steps=1)
    # Begin epoch
    cb.on_epoch_begin(DummyArgs(str(outdir)), DummyState(epoch=1), DummyControl())
    # Simulate a step with two completions
    completions = ["a", "b"]
    prompts = ["p1", "p2"]
    # Call wrapped funcs to populate rewards and aggregates
    rfuncs = cb.get_wrapped_reward_funcs()
    # First reward func
    rfuncs[0](completions, prompts)
    # Second reward func triggers batch_collected and on_step_end behavior
    cb.on_step_begin(DummyArgs(str(outdir)), DummyState(epoch=1, global_step=0), DummyControl())
    rfuncs = cb.get_wrapped_reward_funcs()
    rfuncs[0](completions, prompts)
    rfuncs[1](completions, prompts)
    cb.on_step_end(DummyArgs(str(outdir)), DummyState(epoch=1, global_step=1), DummyControl())
    # End epoch should save best
    model = DummyModel()
    cb.on_epoch_end(DummyArgs(str(outdir)), DummyState(epoch=1), DummyControl(), model=model)
    best_dir = os.path.join(str(outdir), "best")
    assert os.path.isdir(best_dir)
    assert os.path.exists(os.path.join(best_dir, "pytorch_model.bin"))



