"""Motive Decomposition Network (MDN)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn


class MotiveDecompositionNetwork(nn.Module):
    """Predict context-conditioned motive geometry with explicit routing."""

    def __init__(
        self,
        input_dim: int = 14,
        num_motives: int = 2,
        num_directions: int = 2,
        hidden_dim: int = 64,
        num_hidden_layers: int = 2,
        alpha_epsilon: float = 1e-6,
        num_objectives: int | None = None,
    ) -> None:
        super().__init__()

        if num_objectives is not None:
            num_motives = num_objectives
        if num_hidden_layers < 2 or num_hidden_layers > 3:
            raise ValueError(f"Expected num_hidden_layers in [2, 3], got {num_hidden_layers}")

        self.input_dim = int(input_dim)
        self.num_motives = int(num_motives)
        self.num_objectives = self.num_motives
        self.num_directions = int(num_directions)
        self.hidden_dim = int(hidden_dim)
        self.num_hidden_layers = int(num_hidden_layers)
        self.alpha_epsilon = float(alpha_epsilon)

        trunk_layers: list[nn.Module] = [nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU()]
        for _ in range(self.num_hidden_layers - 1):
            trunk_layers.extend([nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()])

        self.backbone = nn.Sequential(*trunk_layers)
        self.gate_head = nn.Linear(self.hidden_dim, 1)
        self.motive_head = nn.Linear(self.hidden_dim, self.num_motives)
        self.dirichlet_head = nn.Linear(self.hidden_dim, self.num_motives)
        self.support_head = nn.Linear(self.hidden_dim, self.num_directions)

        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Apply stable initialization across all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _validate_context(self, context: Tensor) -> tuple[Tensor, bool]:
        if context.ndim not in (1, 2):
            raise ValueError(
                f"Expected context with shape ({self.input_dim},) or (N, {self.input_dim}), got {tuple(context.shape)}"
            )

        is_single_input = context.ndim == 1
        if is_single_input:
            if context.shape[0] != self.input_dim:
                raise ValueError(
                    f"Expected single context shape ({self.input_dim},), got {tuple(context.shape)}"
                )
            context = context.unsqueeze(0)
        elif context.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected batched context shape (N, {self.input_dim}), got {tuple(context.shape)}"
            )

        return context, is_single_input

    def forward(self, context: Tensor, training: bool = True) -> tuple[Tensor, Tensor]:
        context, is_single_input = self._validate_context(context)

        features = self.backbone(context)
        if training:
            p_hat = self.sigmoid(self.gate_head(features)).squeeze(-1)
            q_hat = self.motive_head(features)
            if is_single_input:
                p_hat = p_hat.squeeze(0)
                q_hat = q_hat.squeeze(0)
            return p_hat, q_hat

        alpha = self.softplus(self.dirichlet_head(features)) + self.alpha_epsilon
        support_values = self.sigmoid(self.support_head(features))
        if is_single_input:
            alpha = alpha.squeeze(0)
            support_values = support_values.squeeze(0)
        return alpha, support_values
