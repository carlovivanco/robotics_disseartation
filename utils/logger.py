"""
Unified logger: W&B + TensorBoard + CSV.
Wraps both backends so training code is backend-agnostic.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(
        self,
        cfg,
        output_dir: str,
        run_name: str | None = None,
    ) -> None:
        self._step = 0
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # ── TensorBoard ───────────────────────────────────────────────────────
        self._tb = SummaryWriter(log_dir=str(self._output_dir / "tensorboard"))

        # ── W&B ──────────────────────────────────────────────────────────────
        self._use_wandb = getattr(cfg.logging, "use_wandb", False)
        if self._use_wandb:
            try:
                import wandb

                self._wandb = wandb
                wandb.init(
                    project=cfg.logging.project,
                    entity=cfg.logging.entity or None,
                    name=run_name or cfg.logging.run_name,
                    tags=list(cfg.logging.tags),
                    config=self._flatten_cfg(cfg),
                    dir=str(self._output_dir),
                    resume="allow",
                )
            except ImportError:
                print("[Logger] wandb not installed — falling back to TensorBoard only.")
                self._use_wandb = False

        # ── CSV fallback ──────────────────────────────────────────────────────
        csv_path = self._output_dir / "metrics.csv"
        self._csv_file = open(csv_path, "a", newline="")
        self._csv_writer = None  # initialised on first log call

    # ── public API ────────────────────────────────────────────────────────────

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        step = step if step is not None else self._step
        self._step = step

        # TensorBoard
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self._tb.add_scalar(k, v, global_step=step)

        # W&B
        if self._use_wandb:
            self._wandb.log(metrics, step=step)

        # CSV (writes header on first call)
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=["step"] + sorted(metrics.keys())
            )
            self._csv_writer.writeheader()
        row = {"step": step, **metrics}
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def log_image(self, tag: str, image, step: int | None = None) -> None:
        step = step if step is not None else self._step
        self._tb.add_image(tag, image, global_step=step)
        if self._use_wandb:
            import wandb
            self._wandb.log({tag: wandb.Image(image)}, step=step)

    def close(self) -> None:
        self._tb.close()
        self._csv_file.close()
        if self._use_wandb:
            self._wandb.finish()

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _flatten_cfg(cfg, prefix: str = "") -> dict:
        """Recursively flatten an OmegaConf / dataclass config into a flat dict."""
        try:
            from omegaconf import OmegaConf
            return OmegaConf.to_container(cfg, resolve=True)
        except Exception:
            return {}
