"""
Loss functions for MyXTTS training.

This module implements various loss functions used in XTTS training,
including mel spectrogram loss, stop token loss, and combined losses.
"""

import tensorflow as tf
from typing import Dict, Optional, Tuple


def mel_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    lengths: Optional[tf.Tensor] = None
) -> tf.Tensor:
    """
    Mel spectrogram L1 loss.
    
    Args:
        y_true: Target mel spectrograms [batch, time, n_mels]
        y_pred: Predicted mel spectrograms [batch, time, n_mels]
        lengths: Sequence lengths [batch] for masking
        
    Returns:
        Mel loss tensor
    """
    # Compute L1 loss
    loss = tf.abs(y_true - y_pred)
    
    # Apply sequence masking if lengths provided
    if lengths is not None:
        mask = tf.sequence_mask(
            lengths,
            maxlen=tf.shape(y_true)[1],
            dtype=tf.float32
        )
        mask = tf.expand_dims(mask, -1)  # [batch, time, 1]
        loss = loss * mask
        
        # Normalize by actual sequence lengths
        loss = tf.reduce_sum(loss, axis=[1, 2])  # [batch]
        normalizer = tf.cast(lengths, tf.float32) * tf.cast(tf.shape(y_true)[2], tf.float32)
        loss = loss / tf.maximum(normalizer, 1.0)
        loss = tf.reduce_mean(loss)
    else:
        loss = tf.reduce_mean(loss)
    
    return loss


def stop_token_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    lengths: Optional[tf.Tensor] = None
) -> tf.Tensor:
    """
    Stop token binary cross-entropy loss.
    
    Args:
        y_true: Target stop tokens [batch, time, 1]
        y_pred: Predicted stop tokens [batch, time, 1]  
        lengths: Sequence lengths [batch] for masking
        
    Returns:
        Stop token loss tensor
    """
    # Binary cross-entropy loss
    loss = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False)
    
    # Apply sequence masking if lengths provided
    if lengths is not None:
        mask = tf.sequence_mask(
            lengths,
            maxlen=tf.shape(y_true)[1],
            dtype=tf.float32
        )
        loss = loss * mask
        
        # Normalize by actual sequence lengths
        loss = tf.reduce_sum(loss, axis=1)  # [batch]
        loss = loss / tf.maximum(tf.cast(lengths, tf.float32), 1.0)
        loss = tf.reduce_mean(loss)
    else:
        loss = tf.reduce_mean(loss)
    
    return loss


def attention_loss(
    attention_weights: tf.Tensor,
    text_lengths: tf.Tensor,
    mel_lengths: tf.Tensor
) -> tf.Tensor:
    """
    Attention alignment loss to encourage monotonic attention.
    
    Args:
        attention_weights: Attention weights [batch, heads, mel_time, text_time]
        text_lengths: Text sequence lengths [batch]
        mel_lengths: Mel sequence lengths [batch]
        
    Returns:
        Attention loss tensor
    """
    batch_size = tf.shape(attention_weights)[0]
    mel_time = tf.shape(attention_weights)[2]
    text_time = tf.shape(attention_weights)[3]
    
    # Average over heads
    attention_weights = tf.reduce_mean(attention_weights, axis=1)  # [batch, mel_time, text_time]
    
    # Create expected monotonic alignment
    mel_positions = tf.range(mel_time, dtype=tf.float32)
    text_positions = tf.range(text_time, dtype=tf.float32)
    
    expected_alignment = []
    for i in range(batch_size):
        mel_len = mel_lengths[i]
        text_len = text_lengths[i]
        
        # Linear alignment
        alignment_ratio = tf.cast(text_len - 1, tf.float32) / tf.cast(mel_len - 1, tf.float32)
        expected_text_pos = mel_positions * alignment_ratio
        
        # Create Gaussian around expected positions
        text_pos_expanded = tf.expand_dims(text_positions, 0)  # [1, text_time]
        expected_pos_expanded = tf.expand_dims(expected_text_pos, 1)  # [mel_time, 1]
        
        gaussian = tf.exp(-0.5 * tf.square(text_pos_expanded - expected_pos_expanded) / 2.0)
        gaussian = gaussian / tf.reduce_sum(gaussian, axis=1, keepdims=True)
        
        expected_alignment.append(gaussian)
    
    expected_alignment = tf.stack(expected_alignment, axis=0)
    
    # Compute KL divergence
    attention_weights = tf.clip_by_value(attention_weights, 1e-8, 1.0 - 1e-8)
    expected_alignment = tf.clip_by_value(expected_alignment, 1e-8, 1.0 - 1e-8)
    
    kl_loss = tf.reduce_sum(
        expected_alignment * tf.math.log(expected_alignment / attention_weights),
        axis=[1, 2]
    )
    
    return tf.reduce_mean(kl_loss)


def duration_loss(
    predicted_durations: tf.Tensor,
    target_durations: tf.Tensor,
    text_lengths: tf.Tensor
) -> tf.Tensor:
    """
    Duration prediction loss.
    
    Args:
        predicted_durations: Predicted durations [batch, text_len]
        target_durations: Target durations [batch, text_len] 
        text_lengths: Text sequence lengths [batch]
        
    Returns:
        Duration loss tensor
    """
    # MSE loss
    loss = tf.square(predicted_durations - target_durations)
    
    # Apply masking
    mask = tf.sequence_mask(
        text_lengths,
        maxlen=tf.shape(predicted_durations)[1],
        dtype=tf.float32
    )
    loss = loss * mask
    
    # Normalize by sequence lengths
    loss = tf.reduce_sum(loss, axis=1)
    loss = loss / tf.maximum(tf.cast(text_lengths, tf.float32), 1.0)
    
    return tf.reduce_mean(loss)


class XTTSLoss(tf.keras.losses.Loss):
    """
    Combined loss function for XTTS training.
    
    Combines mel spectrogram loss, stop token loss, and optional
    regularization losses with configurable weights.
    """
    
    def __init__(
        self,
        mel_loss_weight: float = 45.0,
        stop_loss_weight: float = 1.0,
        attention_loss_weight: float = 1.0,
        duration_loss_weight: float = 1.0,
        name: str = "xtts_loss"
    ):
        """
        Initialize XTTS loss.
        
        Args:
            mel_loss_weight: Weight for mel spectrogram loss
            stop_loss_weight: Weight for stop token loss
            attention_loss_weight: Weight for attention loss
            duration_loss_weight: Weight for duration loss
            name: Loss name
        """
        super().__init__(name=name)
        
        self.mel_loss_weight = mel_loss_weight
        self.stop_loss_weight = stop_loss_weight
        self.attention_loss_weight = attention_loss_weight
        self.duration_loss_weight = duration_loss_weight
    
    def call(
        self,
        y_true: Dict[str, tf.Tensor],
        y_pred: Dict[str, tf.Tensor]
    ) -> tf.Tensor:
        """
        Compute combined loss.
        
        Args:
            y_true: Dictionary of target tensors
            y_pred: Dictionary of predicted tensors
            
        Returns:
            Combined loss tensor
        """
        losses = {}
        total_loss = 0.0
        
        # Mel spectrogram loss
        if "mel_target" in y_true and "mel_output" in y_pred:
            mel_l = mel_loss(
                y_true["mel_target"],
                y_pred["mel_output"],
                y_true.get("mel_lengths")
            )
            losses["mel_loss"] = mel_l
            total_loss += self.mel_loss_weight * mel_l
        
        # Stop token loss
        if "stop_target" in y_true and "stop_tokens" in y_pred:
            stop_l = stop_token_loss(
                y_true["stop_target"],
                y_pred["stop_tokens"],
                y_true.get("mel_lengths")
            )
            losses["stop_loss"] = stop_l
            total_loss += self.stop_loss_weight * stop_l
        
        # Attention loss (if attention weights available)
        if ("attention_weights" in y_pred and 
            "text_lengths" in y_true and 
            "mel_lengths" in y_true):
            attn_l = attention_loss(
                y_pred["attention_weights"],
                y_true["text_lengths"],
                y_true["mel_lengths"]
            )
            losses["attention_loss"] = attn_l
            total_loss += self.attention_loss_weight * attn_l
        
        # Duration loss (if duration predictions available)
        if ("duration_pred" in y_pred and 
            "duration_target" in y_true and
            "text_lengths" in y_true):
            dur_l = duration_loss(
                y_pred["duration_pred"],
                y_true["duration_target"],
                y_true["text_lengths"]
            )
            losses["duration_loss"] = dur_l
            total_loss += self.duration_loss_weight * dur_l
        
        # Store individual losses for monitoring
        self.losses = losses
        
        return total_loss
    
    def get_losses(self) -> Dict[str, tf.Tensor]:
        """Get individual loss components."""
        return getattr(self, 'losses', {})


def create_stop_targets(mel_lengths: tf.Tensor, max_length: int) -> tf.Tensor:
    """
    Create stop token targets from mel lengths.
    
    Args:
        mel_lengths: Mel sequence lengths [batch]
        max_length: Maximum sequence length
        
    Returns:
        Stop token targets [batch, max_length, 1]
    """
    batch_size = tf.shape(mel_lengths)[0]
    
    # Create position indices
    positions = tf.range(max_length, dtype=tf.int32)
    positions = tf.expand_dims(positions, 0)  # [1, max_length]
    positions = tf.tile(positions, [batch_size, 1])  # [batch, max_length]
    
    # Create stop targets (1 at end positions, 0 elsewhere)
    mel_lengths_expanded = tf.expand_dims(mel_lengths - 1, 1)  # [batch, 1]
    stop_targets = tf.cast(
        tf.equal(positions, mel_lengths_expanded),
        tf.float32
    )
    
    # Add channel dimension
    stop_targets = tf.expand_dims(stop_targets, -1)  # [batch, max_length, 1]
    
    return stop_targets


def compute_mel_statistics(mel_spectrograms: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Compute mel spectrogram statistics for normalization.
    
    Args:
        mel_spectrograms: Batch of mel spectrograms [batch, time, n_mels]
        
    Returns:
        Tuple of (mean, std) tensors
    """
    # Compute statistics across batch and time dimensions
    mean = tf.reduce_mean(mel_spectrograms, axis=[0, 1], keepdims=True)
    std = tf.math.reduce_std(mel_spectrograms, axis=[0, 1], keepdims=True)
    
    return mean, std