"""Regression tests for MDN training and inference routing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from generator.mdn import MotiveDecompositionNetwork
from generator.mdn_trainer import MDNTrainer


def _write_synthetic_episodes(data_dir: Path, seed: int = 7, count: int = 80) -> None:
    rng = np.random.default_rng(seed)
    data_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        context = rng.normal(size=14).astype(np.float32)
        payoff = float(0.7 * context[0] + 0.4 * context[1] + rng.normal(scale=0.1))
        motives = np.array([
            0.6 * context[2] - 0.2 * context[3] + rng.normal(scale=0.05),
            -0.4 * context[4] + 0.3 * context[5] + rng.normal(scale=0.05),
        ], dtype=np.float32)
        np.savez(
            data_dir / f"episode_{index:03d}.npz",
            obs=context,
            payoff=payoff,
            motives=motives,
            skill_id=f"skill_{index:03d}",
            terminated=bool(index % 3 == 0),
        )


def test_mdn_training_and_inference_modes_return_expected_contract():
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork(input_dim=14, num_motives=2, num_directions=3)
    context = torch.randn(4, 14)

    p_hat, q_hat = model(context, training=True)
    assert p_hat.shape == (4,)
    assert q_hat.shape == (4, 2)
    assert torch.all(p_hat >= 0)
    assert torch.all(p_hat <= 1)
    assert torch.isfinite(q_hat).all()

    alpha, support = model(context, training=False)
    assert alpha.shape == (4, 2)
    assert support.shape == (4, 3)
    assert torch.all(alpha > 0)
    assert torch.all(support >= 0)
    assert torch.all(support <= 1)


def test_accept_labels_are_binary_and_correctly_shaped(tmp_path: Path):
    data_dir = tmp_path / "episodes"
    _write_synthetic_episodes(data_dir, count=12)

    trainer = MDNTrainer(data_dir=data_dir, model_dir=tmp_path / "models", epochs=1, verbose=False)
    records = trainer.load_records()
    labels, baseline_stats = trainer.build_dataset(records)

    accept_labels = labels.accept_labels
    assert accept_labels.shape == (12,)
    assert set(np.unique(accept_labels)).issubset({0.0, 1.0})
    assert "baseline_payoff" in baseline_stats
    assert "baseline_motives" in baseline_stats


def test_composite_loss_decreases_over_training(tmp_path: Path):
    data_dir = tmp_path / "episodes"
    _write_synthetic_episodes(data_dir, count=90)

    trainer = MDNTrainer(
        data_dir=data_dir,
        model_dir=tmp_path / "models",
        seed=21,
        batch_size=16,
        epochs=60,
        learning_rate=5e-3,
        lambda_mse=0.1,
        patience=20,
        verbose=False,
    )
    result = trainer.fit()

    history = result["history"]
    assert len(history["train_loss"]) >= 2
    assert history["train_loss"][-1] < history["train_loss"][0]
    assert history["val_loss"][-1] <= max(history["val_loss"])


def test_checkpoint_reload_reproduces_outputs(tmp_path: Path):
    data_dir = tmp_path / "episodes"
    _write_synthetic_episodes(data_dir, count=40)

    trainer = MDNTrainer(data_dir=data_dir, model_dir=tmp_path / "models", seed=11, epochs=20, verbose=False)
    trainer.fit()

    checkpoint = torch.load(trainer.checkpoint_path, map_location="cpu")
    restored = MotiveDecompositionNetwork(input_dim=14, num_motives=2, num_directions=2)
    restored.load_state_dict(checkpoint["model_state_dict"])

    sample = torch.tensor(np.load(sorted(data_dir.glob("*.npz"))[0], allow_pickle=True)["obs"], dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        original_train = trainer.model.cpu()(sample, training=True)
        restored_train = restored(sample, training=True)
        original_infer = trainer.model.cpu()(sample, training=False)
        restored_infer = restored(sample, training=False)

    assert torch.allclose(original_train[0], restored_train[0])
    assert torch.allclose(original_train[1], restored_train[1])
    assert torch.allclose(original_infer[0], restored_infer[0])
    assert torch.allclose(original_infer[1], restored_infer[1])


def test_seed_produces_deterministic_training_curves(tmp_path: Path):
    data_dir = tmp_path / "episodes"
    _write_synthetic_episodes(data_dir, count=50)

    common_kwargs = dict(
        data_dir=data_dir,
        model_dir=tmp_path / "models",
        seed=123,
        batch_size=10,
        epochs=25,
        learning_rate=1e-3,
        lambda_mse=0.1,
        patience=10,
        verbose=False,
    )

    first = MDNTrainer(**common_kwargs).fit()["history"]
    second = MDNTrainer(**common_kwargs).fit()["history"]

    assert first["train_loss"] == second["train_loss"]
    assert first["val_loss"] == second["val_loss"]
