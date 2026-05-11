"""MDN training pipeline for SubRep."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import random
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from baseline.improvement_calculator import ImprovementCalculator
from certification.cds_test import CDSGate
from generator.mdn import MotiveDecompositionNetwork
from utils.ips_estimator import estimate_q_ips


@dataclass(frozen=True)
class EpisodeRecord:
    context: np.ndarray
    payoff: float
    motives: np.ndarray
    skill_id: str
    terminated: bool


class EpisodeSummaryDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    def __init__(self, records: list[EpisodeRecord], accept_labels: np.ndarray, q_ips: np.ndarray) -> None:
        self.records = records
        self.accept_labels = accept_labels.astype(np.float32).reshape(-1)
        self.q_ips = q_ips.astype(np.float32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        record = self.records[index]
        context = torch.tensor(record.context, dtype=torch.float32)
        accept = torch.tensor(self.accept_labels[index], dtype=torch.float32)
        q_target = torch.tensor(self.q_ips[index], dtype=torch.float32)
        motives = torch.tensor(record.motives, dtype=torch.float32)
        return context, accept, q_target, motives


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def _load_npz_episode(path: Path) -> EpisodeRecord:
    with np.load(path, allow_pickle=True) as data:
        for key in ("obs", "payoff", "motives", "skill_id", "terminated"):
            if key not in data:
                raise KeyError(f"{path} is missing required key {key!r}")

        context = np.asarray(data["obs"], dtype=np.float32).reshape(-1)
        payoff = float(np.asarray(data["payoff"]).reshape(()))
        motives = np.asarray(data["motives"], dtype=np.float32).reshape(-1)
        skill_id = str(np.asarray(data["skill_id"]).reshape(()))
        terminated = bool(np.asarray(data["terminated"]).reshape(()))

    if context.shape[0] != 14:
        raise ValueError(f"Expected context dimension 14, got {context.shape[0]} in {path}")
    if motives.ndim != 1:
        raise ValueError(f"Expected motive vector to be 1D, got {motives.shape} in {path}")

    return EpisodeRecord(context=context, payoff=payoff, motives=motives, skill_id=skill_id, terminated=terminated)


def _build_accept_labels(records: list[EpisodeRecord]) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    payoffs = np.asarray([record.payoff for record in records], dtype=np.float32)
    motives = np.asarray([record.motives for record in records], dtype=np.float32)
    baseline_stats = {
        "baseline_payoff": float(np.mean(payoffs)),
        "baseline_motives": np.mean(motives, axis=0).astype(np.float32),
    }
    calculator = ImprovementCalculator(baseline_stats)
    gate = CDSGate()

    labels = []
    for record in records:
        delta_r, delta_n = calculator.compute_improvements(record.payoff, record.motives)
        labels.append(int(gate.admit(delta_r, delta_n)))

    return np.asarray(labels, dtype=np.int64), baseline_stats


class MDNTrainer:
    def __init__(
        self,
        data_dir: str | Path = "data",
        model_dir: str | Path = "models",
        seed: int = 42,
        batch_size: int = 16,
        epochs: int = 80,
        learning_rate: float = 1e-3,
        lambda_mse: float = 0.1,
        val_split: float = 0.2,
        hidden_dim: int = 64,
        num_hidden_layers: int = 2,
        patience: int = 10,
        verbose: bool = True,
        device: str | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.lambda_mse = float(lambda_mse)
        self.val_split = float(val_split)
        self.hidden_dim = int(hidden_dim)
        self.num_hidden_layers = int(num_hidden_layers)
        self.patience = int(patience)
        self.verbose = bool(verbose)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        _seed_everything(self.seed)

        self.model = MotiveDecompositionNetwork(
            input_dim=14,
            num_motives=2,
            num_directions=2,
            hidden_dim=self.hidden_dim,
            num_hidden_layers=self.num_hidden_layers,
        ).to(self.device)

        self.checkpoint_path = self.model_dir / "mdn_best.pth"
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_bce": [],
            "train_mse": [],
            "val_bce": [],
            "val_mse": [],
            "lr": [],
        }

    def load_records(self) -> list[EpisodeRecord]:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {self.data_dir}")

        files = sorted(self.data_dir.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No .npz episode files found in {self.data_dir}")

        return [_load_npz_episode(path) for path in files]

    def build_dataset(self, records: list[EpisodeRecord]) -> tuple[EpisodeSummaryDataset, dict[str, np.ndarray | float]]:
        accept_labels, baseline_stats = _build_accept_labels(records)
        q_ips = estimate_q_ips([{"motives": record.motives} for record in records], gamma=0.99)
        dataset = EpisodeSummaryDataset(records, accept_labels=accept_labels, q_ips=q_ips)
        return dataset, baseline_stats

    def _split_indices(self, num_items: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        indices = np.arange(num_items)
        rng.shuffle(indices)
        split_point = max(1, int(round(num_items * (1.0 - self.val_split))))
        split_point = min(split_point, num_items - 1) if num_items > 1 else num_items
        train_indices = indices[:split_point]
        val_indices = indices[split_point:]
        if len(val_indices) == 0:
            val_indices = train_indices[-1:]
            train_indices = train_indices[:-1]
        return train_indices, val_indices

    def _make_loader(self, dataset: Dataset, indices: np.ndarray, shuffle: bool) -> DataLoader:
        generator = torch.Generator().manual_seed(self.seed)
        subset = Subset(dataset, indices.tolist())
        return DataLoader(subset, batch_size=self.batch_size, shuffle=shuffle, generator=generator)

    def _run_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer | None = None) -> tuple[float, float, float]:
        total_loss = 0.0
        total_bce = 0.0
        total_mse = 0.0
        num_samples = 0

        if optimizer is None:
            self.model.eval()
            grad_context = torch.no_grad()
        else:
            self.model.train()
            grad_context = torch.enable_grad()

        bce_loss = nn.BCELoss()
        mse_loss = nn.MSELoss()

        with grad_context:
            for context, accept, q_target, _motives in loader:
                context = context.to(self.device)
                accept = accept.to(self.device)
                q_target = q_target.to(self.device)

                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)

                p_hat, q_hat = self.model(context, training=True)
                loss_bce = bce_loss(p_hat.view_as(accept), accept)
                loss_mse = mse_loss(q_hat, q_target)
                loss = loss_bce + self.lambda_mse * loss_mse

                if optimizer is not None:
                    loss.backward()
                    optimizer.step()

                batch_size = context.shape[0]
                num_samples += batch_size
                total_loss += float(loss.item()) * batch_size
                total_bce += float(loss_bce.item()) * batch_size
                total_mse += float(loss_mse.item()) * batch_size

        denominator = max(num_samples, 1)
        return total_loss / denominator, total_bce / denominator, total_mse / denominator

    def fit(self) -> dict[str, Any]:
        _seed_everything(self.seed)
        records = self.load_records()
        dataset, baseline_stats = self.build_dataset(records)
        train_indices, val_indices = self._split_indices(len(dataset))

        train_loader = self._make_loader(dataset, train_indices, shuffle=True)
        val_loader = self._make_loader(dataset, val_indices, shuffle=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

        best_state = None
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        self.model.to(self.device)
        self.model.train()

        for epoch in range(self.epochs):
            train_loss, train_bce, train_mse = self._run_epoch(train_loader, optimizer=optimizer)
            val_loss, val_bce, val_mse = self._run_epoch(val_loader, optimizer=None)
            scheduler.step(val_loss)

            current_lr = float(optimizer.param_groups[0]["lr"])
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_bce"].append(train_bce)
            self.history["train_mse"].append(train_mse)
            self.history["val_bce"].append(val_bce)
            self.history["val_mse"].append(val_mse)
            self.history["lr"].append(current_lr)

            if self.verbose:
                print(
                    f"Epoch {epoch + 1:03d}/{self.epochs} | train={train_loss:.6f} val={val_loss:.6f} "
                    f"(bce={val_bce:.6f}, mse={val_mse:.6f}, lr={current_lr:.2e})"
                )

            if val_loss < best_val_loss - 1e-8:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_state = {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lambda_mse": self.lambda_mse,
                    "seed": self.seed,
                    "metrics": {key: list(values) for key, values in self.history.items()},
                    "split": {
                        "train_indices": train_indices.tolist(),
                        "val_indices": val_indices.tolist(),
                        "val_split": self.val_split,
                    },
                    "baseline_stats": {
                        "baseline_payoff": float(baseline_stats["baseline_payoff"]),
                        "baseline_motives": np.asarray(baseline_stats["baseline_motives"], dtype=np.float32).tolist(),
                    },
                    "best_epoch": epoch + 1,
                    "best_val_loss": best_val_loss,
                }
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.patience:
                if self.verbose:
                    print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

        if best_state is None:
            best_state = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lambda_mse": self.lambda_mse,
                "seed": self.seed,
                "metrics": {key: list(values) for key, values in self.history.items()},
                "split": {
                    "train_indices": train_indices.tolist(),
                    "val_indices": val_indices.tolist(),
                    "val_split": self.val_split,
                },
                "baseline_stats": {
                    "baseline_payoff": float(baseline_stats["baseline_payoff"]),
                    "baseline_motives": np.asarray(baseline_stats["baseline_motives"], dtype=np.float32).tolist(),
                },
                "best_epoch": len(self.history["val_loss"]),
                "best_val_loss": float(self.history["val_loss"][-1]) if self.history["val_loss"] else float("inf"),
            }

        self.model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, self.checkpoint_path)
        if self.verbose:
            print(f"Best checkpoint saved to {self.checkpoint_path}")

        return {
            "history": {key: list(values) for key, values in self.history.items()},
            "best_val_loss": best_val_loss,
            "checkpoint_path": str(self.checkpoint_path),
            "baseline_stats": best_state["baseline_stats"],
            "split": best_state["split"],
        }

    def train(self) -> dict[str, Any]:
        """Alias for fit() to support a simple trainer API."""
        return self.fit()
