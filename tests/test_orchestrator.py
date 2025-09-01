from orchestrator import pick_free_gpu, parse_nvidia_smi_memory_used, build_sweep_configs, format_run_name, TrainConfig


def test_parse_nvidia_smi_memory_used_basic():
    out = """\n0\n102\n2048\n\n"""
    vals = parse_nvidia_smi_memory_used(out)
    assert vals == [0, 102, 2048]


def test_pick_free_gpu_1gb_threshold_override():
    # Simulate GPUs [500MB, 1500MB, 0MB, 2048MB]
    used = [500, 1500, 0, 2048]
    gpu = pick_free_gpu(threshold_mb=1024, candidates=(0, 1, 2, 3), memory_used_override=used)
    assert gpu == 0  # 500 <= 1024

    gpu = pick_free_gpu(threshold_mb=1024, candidates=(1, 2, 3), memory_used_override=used)
    assert gpu == 2  # 1500>1GB, 2 is 0MB


def test_build_sweep_configs_grid_sizes():
    cfgs = build_sweep_configs("PleIAs/Pleias-350m-Preview")
    # 2 (rr) * 2 (endeca) * 4 (rp) * 4 (kl) = 64
    assert len(cfgs) == 64
    assert any(isinstance(c, TrainConfig) for c in cfgs)


def test_format_run_name_contains_flags():
    cfg = TrainConfig(
        model_name="m",
        use_repetition_reward=True,
        endeca_mode="top_prob",
        gen_repetition_penalty=1.20,
        kl_beta=0.05,
    )
    rn = format_run_name("dante-20250101-pleias", cfg)
    assert "rr_on" in rn
    assert "emode_top" in rn
    assert "rp_1.20" in rn
    assert "kl_0.05" in rn



