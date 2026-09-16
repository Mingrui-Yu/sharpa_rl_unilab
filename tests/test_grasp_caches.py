import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from unilab.base.run_control import RunComplete

from sharpa_rl_unilab import assets
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.cache import (
    grasp_cache_output_file,
    resolve_grasp_cache_file,
    sample_scale_grasp_caches,
    validate_grasp_caches,
)
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.grasp import SharpaGraspRecorder


@pytest.fixture
def bundled_cache(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    relative = "caches/sharpa_grasp_linspace_1.npy"
    source = bundle / relative
    source.parent.mkdir(parents=True)
    np.save(source, np.zeros((1, 29), dtype=np.float32))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    (bundle / "manifest.json").write_text(json.dumps({"sha256": {relative: digest}}))
    monkeypatch.setattr(assets, "ASSETS_ROOT_PATH", bundle)
    monkeypatch.setenv(assets.CACHE_ENV_VAR, str(tmp_path / "managed"))
    assets.ensure_assets()
    return relative, source


def test_generated_grasps_survive_asset_repair_and_override_bundled_data(bundled_cache):
    relative, source = bundled_cache
    prefix = "caches/sharpa_grasp_linspace"
    managed = assets.cache_root() / relative
    assert resolve_grasp_cache_file(prefix, 1) == managed
    output = grasp_cache_output_file(prefix, 1)
    output.parent.mkdir(parents=True)
    generated = np.ones((3, 29), dtype=np.float32)
    np.save(output, generated)
    managed.write_bytes(b"damaged asset")
    assets.ensure_assets()
    assert managed.read_bytes() == source.read_bytes()
    assert resolve_grasp_cache_file(prefix, 1) == output
    np.testing.assert_array_equal(np.load(output), generated)


def test_grasp_target_counts_rows_and_stops_at_exact_limit(bundled_cache):
    recorder = SharpaGraspRecorder.__new__(SharpaGraspRecorder)
    recorder._env = SimpleNamespace(
        reset_time_outs=np.ones(5, bool), reset_terminated=np.zeros(5, bool), extras={}
    )
    recorder._observation = SimpleNamespace(
        dof_pos=np.ones((5, 22)), object_pos=np.ones((5, 3)), object_quat=np.ones((5, 4))
    )
    recorder.task_state = SimpleNamespace(scale_values=np.array([1.0]))
    recorder._prefix = "caches/sharpa_grasp_linspace"
    recorder._target = 6
    recorder._num_rows = 0
    recorder._rows = []
    recorder._saved = recorder._target_notified = False
    recorder._auto_save = True
    output = grasp_cache_output_file(recorder._prefix, 1)

    recorder.record_pre_reset(np.arange(5))
    assert recorder.total_saved == 5 and not output.exists()
    with pytest.raises(RunComplete):
        recorder.record_pre_reset(np.arange(5))
    assert recorder.total_saved == 6
    assert np.load(output).shape == (6, 29)
    assert recorder._env.extras["log"]["grasp/cache_size"] == 6
    recorder.close()
    assets.ensure_assets()
    assert np.load(resolve_grasp_cache_file(recorder._prefix, 1)).shape == (6, 29)


def test_cache_validation_and_sampling_indices():
    for invalid in ((), (np.zeros((0, 29)),), (np.zeros((1, 28)),), (np.full((1, 29), np.nan),)):
        with pytest.raises(ValueError):
            validate_grasp_caches(invalid)
    caches = (np.zeros((2, 29)), np.ones((3, 29)))
    validate_grasp_caches(caches)
    for ids in (np.array([-1]), np.array([2]), np.array([0.5])):
        with pytest.raises(ValueError, match="indices"):
            sample_scale_grasp_caches(caches, ids)
    result = sample_scale_grasp_caches(caches, np.array([1, 0]), rng=np.random.default_rng(1))
    np.testing.assert_array_equal(result[:, 0], [1, 0])


@pytest.mark.slow
def test_grasp_recorder_runs_at_real_mujoco_reset_boundary(tmp_path, monkeypatch):
    from unilab.base.config_adapter import BackendAdapter, create_env

    from sharpa_rl_unilab.cli import compose_config
    from sharpa_rl_unilab.tasks.sharpa_inhand.terms.grasp import SharpaGraspQualityTermination

    monkeypatch.setenv(assets.CACHE_ENV_VAR, str(tmp_path / "assets"))
    monkeypatch.setenv("SHARPA_GRASP_SCALE", "0.8")
    monkeypatch.delenv("SHARPA_GRASP_VARIANT_FILE", raising=False)
    cfg = compose_config(
        "ppo",
        "mujoco",
        ["env.max_episode_seconds=0.1", "env.recorders.grasp_cache.params.target=1"],
        task="sharpa_inhand_grasp",
    )

    def allow_timeout(self, env, **params):
        # Exercise recorder/reset wiring independently of stochastic grasp success.
        self.observation.snapshot(env)
        self.invalid.fill(False)
        return self.invalid

    monkeypatch.setattr(SharpaGraspQualityTermination, "__call__", allow_timeout)
    overrides = BackendAdapter(cfg, root_dir=".", algo_name="ppo").build_task_env_cfg_override()
    env = create_env(cfg, num_envs=2, env_cfg_override=overrides)
    try:
        env.reset(seed=7)
        with pytest.raises(RunComplete):
            for _ in range(3):
                env.step(np.zeros((2, 22)))
    finally:
        env.close()
    assets.ensure_assets()
    output = resolve_grasp_cache_file("caches/sharpa_grasp_linspace", 0.8)
    assert output == grasp_cache_output_file("caches/sharpa_grasp_linspace", 0.8)
    rows = np.load(output)
    assert rows.shape == (1, 29) and np.isfinite(rows).all()
