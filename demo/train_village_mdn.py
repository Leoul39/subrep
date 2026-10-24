"""
Train the village MDN that gives contextual weights and weight regions.

Trains on contexts from every partition, evaluates on held-out contexts drawn
with a different seed, and saves the checkpoint with a manifest. The held-out
thresholds below are fixed before training.

Usage:
    python -m demo.train_village_mdn
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from village_sim.weights import (
    CONCENTRATION,
    MOTIVE_SHIFT,
    evaluate_village_mdn,
    sample_training_contexts,
    train_village_mdn,
)

TRAIN_PER_PARTITION = 200
TRAIN_SEED = 1
HELD_OUT_PER_PARTITION = 100
HELD_OUT_SEED = 99
EPOCHS = 2000
MAX_MEAN_WEIGHT_ERROR = 0.02
MAX_WEIGHT_ERROR = 0.1
DEFAULT_CHECKPOINT = Path("models/village_mdn.pth")


def train(checkpoint: Path | None = None, *, epochs: int = EPOCHS, verbose: bool = True) -> dict:
    """Train, evaluate on held-out contexts, and save the checkpoint and manifest."""
    checkpoint = Path(checkpoint or DEFAULT_CHECKPOINT)
    weights, training = train_village_mdn(
        sample_training_contexts(TRAIN_PER_PARTITION, seed=TRAIN_SEED), epochs=epochs, seed=0
    )
    held_out = evaluate_village_mdn(weights, sample_training_contexts(HELD_OUT_PER_PARTITION, seed=HELD_OUT_SEED))
    passed = (
        held_out["mean_weight_error"] <= MAX_MEAN_WEIGHT_ERROR
        and held_out["max_weight_error"] <= MAX_WEIGHT_ERROR
        and held_out["feasible_region_rate"] == 1.0
    )
    weights.save(checkpoint)
    manifest = {
        "artifact_id": f"village-mdn-{hashlib.sha256(checkpoint.read_bytes()).hexdigest()[:12]}",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "producer": "demo.train_village_mdn",
        "config": {
            "train_per_partition": TRAIN_PER_PARTITION,
            "train_seed": TRAIN_SEED,
            "held_out_per_partition": HELD_OUT_PER_PARTITION,
            "held_out_seed": HELD_OUT_SEED,
            "epochs": epochs,
            "concentration": CONCENTRATION,
            "motive_shift": MOTIVE_SHIFT,
        },
        "thresholds": {"max_mean_weight_error": MAX_MEAN_WEIGHT_ERROR, "max_weight_error": MAX_WEIGHT_ERROR},
        "training": training,
        "held_out": held_out,
        "passed": passed,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    manifest_path = checkpoint.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if verbose:
        print(f"held-out: {held_out}")
        print(f"{'PASSED' if passed else 'FAILED'} thresholds; checkpoint {checkpoint}")
    return {"checkpoint": checkpoint, "manifest": manifest, "manifest_path": manifest_path}


if __name__ == "__main__":
    train()
