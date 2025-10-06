"""
Simple, stable loss for emergency debugging.

This module provides a minimal XTTS loss implementation that avoids
complex smoothing/auxiliary terms to help validate the training loop
when diagnosing NaN issues.
"""

from typing import Dict, Optional
import tensorflow as tf


def _safe_mean(x: tf.Tensor) -> tf.Tensor:
    x = tf.cast(x, tf.float32)
    x = tf.where(tf.math.is_finite(x), x, tf.zeros_like(x))
    denom = tf.size(x)
    denom = tf.cast(tf.maximum(denom, 1), tf.float32)
    return tf.reduce_sum(x) / denom


class SimpleXTTSEmergencyLoss(tf.keras.losses.Loss):
    """Ultra-simple, numerically stable loss for XTTS training.

    Computes:
      total = mel_weight * L1(mel) + stop_weight * BCE(stop)

    Exposes helper accessors used by the trainer for logging.
    """

    def __init__(
        self,
        mel_loss_weight: float = 1.0,
        stop_loss_weight: float = 0.1,
        name: str = "simple_xtts_loss",
    ) -> None:
        super().__init__(name=name)
        self.mel_loss_weight = float(mel_loss_weight)
        self.stop_loss_weight = float(stop_loss_weight)

        # Cached values for logging
        self._last_losses: Dict[str, tf.Tensor] = {}
        self._last_weighted: Dict[str, tf.Tensor] = {}
        self._raw_total: Optional[tf.Tensor] = None

    def call(self, y_true: Dict[str, tf.Tensor], y_pred: Dict[str, tf.Tensor]) -> tf.Tensor:
        mel_target = tf.cast(y_true.get("mel_target"), tf.float32)
        mel_out = tf.cast(y_pred.get("mel_output"), tf.float32)

        # Frame mask derived from stop targets if available
        stop_t_raw = y_true.get("stop_target")
        valid_mask = tf.ones_like(mel_target[:, :, :1], dtype=tf.float32)
        if stop_t_raw is not None:
            stop_t = tf.cast(stop_t_raw, tf.float32)
            valid_mask = valid_mask * (1.0 - tf.clip_by_value(stop_t, 0.0, 1.0))
        else:
            stop_t = None

        # Broadcast mask across mel bins automatically
        mel_l1 = _safe_mean(tf.abs(mel_out - mel_target) * valid_mask)

        # Stop BCE
        stop_pred_raw = y_pred.get("stop_tokens")
        stop_p = tf.cast(stop_pred_raw, tf.float32) if stop_pred_raw is not None else None
        if stop_t is None or stop_p is None:
            stop_bce = tf.zeros((), dtype=tf.float32)
        else:
            stop_bce = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=stop_t, logits=stop_p)
            )

        total = self.mel_loss_weight * mel_l1 + self.stop_loss_weight * stop_bce

        # Cache for logging
        self._last_losses = {
            "mel_loss": mel_l1,
            "stop_loss": stop_bce,
            "total_loss": total,
        }
        self._last_weighted = {
            "w_mel_loss": self.mel_loss_weight * mel_l1,
            "w_stop_loss": self.stop_loss_weight * stop_bce,
        }
        self._raw_total = total

        return total

    # Helper API expected by the trainer
    def get_losses(self) -> Dict[str, tf.Tensor]:
        return {k: tf.identity(v) for k, v in self._last_losses.items()}

    def get_weighted_losses(self) -> Dict[str, tf.Tensor]:
        return {k: tf.identity(v) for k, v in self._last_weighted.items()}

    def get_raw_total_loss(self) -> Optional[tf.Tensor]:
        return self._raw_total

    def get_stability_metrics(self) -> Dict[str, tf.Tensor]:
        # Minimal metrics to satisfy trainer logging
        return {
            "running_mel_loss": tf.cast(self._last_losses.get("mel_loss", 0.0), tf.float32),
            "running_total_loss": tf.cast(self._raw_total if self._raw_total is not None else 0.0, tf.float32),
            "loss_variance": tf.zeros([], tf.float32),
            "loss_stability_score": tf.ones([], tf.float32),
            "step_count": tf.ones([], tf.float32),
        }

    def reset_stability_state(self) -> None:
        self._last_losses = {}
        self._last_weighted = {}
        self._raw_total = None
