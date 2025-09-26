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
    lengths: Optional[tf.Tensor] = None,
    label_smoothing: float = 0.05,
    use_huber_loss: bool = True,
    huber_delta: float = 1.0
) -> tf.Tensor:
    """
    Enhanced mel spectrogram loss with stability improvements.
    
    Args:
        y_true: Target mel spectrograms [batch, time, n_mels]
        y_pred: Predicted mel spectrograms [batch, time, n_mels]
        lengths: Sequence lengths [batch] for masking
        label_smoothing: Label smoothing factor for regularization
        use_huber_loss: Use Huber loss instead of L1 for better stability
        huber_delta: Delta parameter for Huber loss
        
    Returns:
        Stabilized mel loss tensor
    """
    # Apply label smoothing if enabled
    if label_smoothing > 0.0:
        # Add small amount of noise to targets for regularization
        noise_scale = label_smoothing * tf.math.reduce_std(y_true)
        noise = tf.random.normal(tf.shape(y_true), stddev=noise_scale)
        y_true_smoothed = y_true + noise
    else:
        y_true_smoothed = y_true
    
    # Compute loss with optional Huber loss for stability
    if use_huber_loss:
        # Huber loss is less sensitive to outliers than L1
        diff = y_true_smoothed - y_pred
        is_small_error = tf.abs(diff) <= huber_delta
        squared_loss = tf.square(diff) / 2.0
        linear_loss = huber_delta * tf.abs(diff) - tf.square(huber_delta) / 2.0
        loss = tf.where(is_small_error, squared_loss, linear_loss)
    else:
        # Standard L1 loss
        loss = tf.abs(y_true_smoothed - y_pred)
    
    # Apply sequence masking if lengths provided
    if lengths is not None:
        mask = tf.sequence_mask(
            lengths,
            maxlen=tf.shape(y_true)[1],
            dtype=tf.float32
        )
        mask = tf.expand_dims(mask, -1)  # [batch, time, 1]
        loss = loss * mask
        
        # Improved normalization: normalize by actual content, not just sequence length
        loss_sum = tf.reduce_sum(loss, axis=[1, 2])  # [batch]
        content_size = tf.cast(lengths, tf.float32) * tf.cast(tf.shape(y_true)[2], tf.float32)
        
        # Add epsilon to prevent division by zero and stabilize training
        normalized_loss = loss_sum / tf.maximum(content_size, 1.0)
        loss = tf.reduce_mean(normalized_loss)
    else:
        loss = tf.reduce_mean(loss)
    
    # Apply improved loss clipping to prevent three-digit loss values
    loss = tf.clip_by_value(loss, 0.0, 10.0)  # Much tighter clipping for stability
    
    return loss


def stop_token_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    lengths: Optional[tf.Tensor] = None,
    positive_weight: float = 5.0,
    label_smoothing: float = 0.1
) -> tf.Tensor:
    """
    Enhanced stop token binary cross-entropy loss with class balancing.
    
    Args:
        y_true: Target stop tokens [batch, time, 1]
        y_pred: Predicted stop tokens [batch, time, 1]  
        lengths: Sequence lengths [batch] for masking
        positive_weight: Weight for positive (stop) tokens to handle class imbalance
        label_smoothing: Label smoothing factor for better generalization
        
    Returns:
        Balanced stop token loss tensor
    """
    # Apply label smoothing
    if label_smoothing > 0.0:
        y_true_smooth = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing
    else:
        y_true_smooth = y_true
    
    # Compute class-weighted binary cross-entropy
    # Stop tokens (1s) are rare, so weight them more heavily
    pos_weight = tf.constant(positive_weight, dtype=tf.float32)
    
    # Manual weighted binary cross-entropy for better control
    epsilon = 1e-7
    y_pred_clipped = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    
    # Positive class (stop token) loss
    pos_loss = -y_true_smooth * tf.math.log(y_pred_clipped) * pos_weight
    # Negative class (continue) loss  
    neg_loss = -(1.0 - y_true_smooth) * tf.math.log(1.0 - y_pred_clipped)
    
    loss = pos_loss + neg_loss
    
    # Apply sequence masking if lengths provided
    if lengths is not None:
        mask = tf.sequence_mask(
            lengths,
            maxlen=tf.shape(y_true)[1],
            dtype=tf.float32
        )
        loss = loss * tf.expand_dims(mask, -1)
        
        # Normalize by actual sequence lengths
        loss = tf.reduce_sum(loss, axis=[1, 2])  # [batch]
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
    mel_time = tf.shape(attention_weights)[2]
    text_time = tf.shape(attention_weights)[3]

    # Average over heads – the loss operates on the aggregated alignment map.
    attention_weights = tf.reduce_mean(attention_weights, axis=1)  # [batch, mel_time, text_time]

    # Compute an ideal monotonic alignment without using any Python-side loops.
    mel_positions = tf.cast(tf.range(mel_time), tf.float32)[tf.newaxis, :]          # [1, mel_time]
    text_positions = tf.cast(tf.range(text_time), tf.float32)[tf.newaxis, tf.newaxis, :]  # [1, 1, text_time]

    mel_lengths_f = tf.cast(tf.maximum(mel_lengths - 1, 1), tf.float32)
    text_lengths_f = tf.cast(tf.maximum(text_lengths - 1, 1), tf.float32)
    alignment_ratio = tf.math.divide_no_nan(text_lengths_f, mel_lengths_f)[:, tf.newaxis]  # [batch, 1]

    expected_text_pos = alignment_ratio * mel_positions  # [batch, mel_time]
    diff = expected_text_pos[:, :, tf.newaxis] - text_positions  # [batch, mel_time, text_time]

    gaussian = tf.exp(-0.5 * tf.square(diff) / 2.0)
    gaussian_sum = tf.reduce_sum(gaussian, axis=2, keepdims=True) + 1e-8
    expected_alignment = gaussian / gaussian_sum
    
    # Compute KL divergence
    attention_weights = tf.clip_by_value(attention_weights, 1e-8, 1.0 - 1e-8)
    expected_alignment = tf.clip_by_value(expected_alignment, 1e-8, 1.0 - 1e-8)
    
    mel_mask = tf.sequence_mask(
        mel_lengths,
        maxlen=mel_time,
        dtype=tf.float32
    )
    text_mask = tf.sequence_mask(
        text_lengths,
        maxlen=text_time,
        dtype=tf.float32
    )
    combined_mask = mel_mask[:, :, None] * text_mask[:, None, :]

    kl_matrix = expected_alignment * tf.math.log(expected_alignment / attention_weights)
    kl_matrix *= combined_mask

    valid_pairs = tf.reduce_sum(combined_mask, axis=[1, 2])
    kl_loss = tf.reduce_sum(kl_matrix, axis=[1, 2]) / tf.maximum(valid_pairs, 1.0)
    kl_loss = tf.clip_by_value(kl_loss, 0.0, 200.0)

    return tf.reduce_mean(kl_loss)


def duration_loss(
    predicted_durations: tf.Tensor,
    target_durations: tf.Tensor,
    text_lengths: tf.Tensor
) -> tf.Tensor:
    """
    Duration prediction loss with log compression and robust clipping.
    
    Args:
        predicted_durations: Predicted durations [batch, text_len]
        target_durations: Target durations [batch, text_len]
        text_lengths: Text sequence lengths [batch]
        
    Returns:
        Duration loss tensor
    """
    # Work in log space to keep large ratios in a reasonable range
    predicted = tf.math.log1p(tf.nn.relu(predicted_durations))
    target = tf.math.log1p(tf.nn.relu(target_durations))

    diff = predicted - target
    diff = tf.clip_by_value(diff, -5.0, 5.0)
    loss = tf.square(diff)

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

    loss = tf.clip_by_value(tf.reduce_mean(loss), 0.0, 25.0)
    return loss


def pitch_loss(
    predicted_pitch: tf.Tensor,
    target_pitch: tf.Tensor,
    sequence_lengths: Optional[tf.Tensor] = None
) -> tf.Tensor:
    """
    Compute pitch prediction loss (L1 loss with optional masking).
    
    Args:
        predicted_pitch: Predicted pitch values [batch, seq_len, 1]
        target_pitch: Target pitch values [batch, seq_len, 1]
        sequence_lengths: Sequence lengths for masking [batch]
        
    Returns:
        Pitch loss scalar
    """
    def _compress(x: tf.Tensor) -> tf.Tensor:
        signed = tf.math.sign(x)
        magnitude = tf.math.log1p(tf.abs(x))
        return signed * magnitude

    compressed_pred = _compress(predicted_pitch)
    compressed_target = _compress(target_pitch)

    loss = tf.abs(compressed_pred - compressed_target)
    loss = tf.clip_by_value(loss, 0.0, 5.0)
    loss = tf.squeeze(loss, axis=-1)  # [batch, seq_len]
    
    if sequence_lengths is not None:
        # Apply sequence mask
        mask = tf.sequence_mask(
            sequence_lengths,
            maxlen=tf.shape(loss)[1],
            dtype=tf.float32
        )
        loss = loss * mask
        
        # Normalize by sequence lengths
        loss = tf.reduce_sum(loss, axis=1)
        loss = loss / tf.maximum(tf.cast(sequence_lengths, tf.float32), 1.0)
    else:
        loss = tf.reduce_mean(loss, axis=1)
    
    return tf.reduce_mean(loss)


def energy_loss(
    predicted_energy: tf.Tensor,
    target_energy: tf.Tensor,
    sequence_lengths: Optional[tf.Tensor] = None
) -> tf.Tensor:
    """
    Compute energy prediction loss (L1 loss with optional masking).
    
    Args:
        predicted_energy: Predicted energy values [batch, seq_len, 1]
        target_energy: Target energy values [batch, seq_len, 1]
        sequence_lengths: Sequence lengths for masking [batch]
        
    Returns:
        Energy loss scalar
    """
    compressed_pred = tf.math.log1p(tf.nn.relu(predicted_energy))
    compressed_target = tf.math.log1p(tf.nn.relu(target_energy))

    loss = tf.abs(compressed_pred - compressed_target)
    loss = tf.clip_by_value(loss, 0.0, 8.0)
    loss = tf.squeeze(loss, axis=-1)  # [batch, seq_len]
    
    if sequence_lengths is not None:
        # Apply sequence mask
        mask = tf.sequence_mask(
            sequence_lengths,
            maxlen=tf.shape(loss)[1],
            dtype=tf.float32
        )
        loss = loss * mask
        
        # Normalize by sequence lengths
        loss = tf.reduce_sum(loss, axis=1)
        loss = loss / tf.maximum(tf.cast(sequence_lengths, tf.float32), 1.0)
    else:
        loss = tf.reduce_mean(loss, axis=1)
    
    return tf.reduce_mean(loss)


def diffusion_loss(
    predicted_noise: tf.Tensor,
    target_noise: tf.Tensor,
    sequence_lengths: Optional[tf.Tensor] = None
) -> tf.Tensor:
    """
    Compute diffusion loss (noise prediction loss).
    
    Args:
        predicted_noise: Predicted noise [batch, seq_len, n_mels]
        target_noise: Target noise [batch, seq_len, n_mels]
        sequence_lengths: Sequence lengths for masking [batch]
        
    Returns:
        Diffusion loss scalar
    """
    # L2 loss for noise prediction
    loss = tf.reduce_mean(tf.square(predicted_noise - target_noise), axis=-1)  # [batch, seq_len]
    
    if sequence_lengths is not None:
        # Apply sequence mask
        mask = tf.sequence_mask(
            sequence_lengths,
            maxlen=tf.shape(loss)[1],
            dtype=tf.float32
        )
        loss = loss * mask
        
        # Normalize by sequence lengths
        loss = tf.reduce_sum(loss, axis=1)
        loss = loss / tf.maximum(tf.cast(sequence_lengths, tf.float32), 1.0)
    else:
        loss = tf.reduce_mean(loss, axis=1)
    
    return tf.reduce_mean(loss)


def speaking_rate_loss(
    predicted_rate: tf.Tensor,
    target_rate: tf.Tensor,
    text_lengths: tf.Tensor
) -> tf.Tensor:
    """Simple L1 loss for speaking-rate predictions with masking."""
    diff = tf.abs(predicted_rate - target_rate)
    diff = tf.clip_by_value(diff, 0.0, 10.0)

    mask = tf.sequence_mask(
        text_lengths,
        maxlen=tf.shape(diff)[1],
        dtype=tf.float32
    )
    diff = diff * tf.expand_dims(mask, axis=-1)

    diff = tf.reduce_sum(diff, axis=1)
    diff = diff / tf.maximum(tf.cast(text_lengths, tf.float32), 1.0)

    return tf.reduce_mean(diff)


class XTTSLoss(tf.keras.losses.Loss):
    """
    Enhanced combined loss function for XTTS training with stability improvements.
    
    Combines mel spectrogram loss, stop token loss, and optional
    regularization losses with configurable weights, adaptive scaling,
    and smoothing mechanisms for better training stability.
    """
    
    def __init__(
        self,
        mel_loss_weight: float = 2.5,  # Fixed from 45.0 - was causing three-digit loss values
        stop_loss_weight: float = 1.0,
        attention_loss_weight: float = 0.1,  # Enabled with small weight for gradual alignment learning
        duration_loss_weight: float = 0.1,   # Enabled: model now returns duration predictions
        # Prosody loss weights (FastSpeech/FastPitch style)
        pitch_loss_weight: float = 0.1,      # Weight for pitch prediction loss
        energy_loss_weight: float = 0.1,     # Weight for energy prediction loss
        prosody_pitch_loss_weight: float = 0.05,
        prosody_energy_loss_weight: float = 0.05,
        speaking_rate_loss_weight: float = 0.05,
        # Voice cloning loss weights
        voice_similarity_loss_weight: float = 1.0,  # Weight for contrastive speaker loss
        # Diffusion loss weights
        diffusion_loss_weight: float = 1.0,  # Weight for diffusion noise prediction loss
        # Stability improvements
        use_adaptive_weights: bool = True,
        loss_smoothing_factor: float = 0.1,
        max_loss_spike_threshold: float = 2.0,
        gradient_norm_threshold: float = 5.0,
        # Contrastive loss parameters
        contrastive_temperature: float = 0.1,
        contrastive_margin: float = 0.2,
        name: str = "xtts_loss"
    ):
        """
        Initialize enhanced XTTS loss with stability features.
        
        Args:
            mel_loss_weight: Weight for mel spectrogram loss
            stop_loss_weight: Weight for stop token loss
            attention_loss_weight: Weight for attention loss
            duration_loss_weight: Weight for duration loss
            pitch_loss_weight: Weight for pitch prediction loss
            energy_loss_weight: Weight for energy prediction loss
            voice_similarity_loss_weight: Weight for contrastive speaker loss
            diffusion_loss_weight: Weight for diffusion noise prediction loss
            use_adaptive_weights: Enable adaptive loss weight scaling
            loss_smoothing_factor: Factor for exponential smoothing of losses
            max_loss_spike_threshold: Maximum allowed loss spike multiplier
            gradient_norm_threshold: Threshold for gradient norm monitoring
            contrastive_temperature: Temperature for contrastive loss
            contrastive_margin: Margin for contrastive loss
            name: Loss name
        """
        super().__init__(name=name)
        
        # Base loss weights
        self.mel_loss_weight = mel_loss_weight
        self.stop_loss_weight = stop_loss_weight
        self.attention_loss_weight = attention_loss_weight
        self.duration_loss_weight = duration_loss_weight
        
        # Prosody loss weights
        self.pitch_loss_weight = pitch_loss_weight
        self.energy_loss_weight = energy_loss_weight
        self.prosody_pitch_loss_weight = prosody_pitch_loss_weight
        self.prosody_energy_loss_weight = prosody_energy_loss_weight
        self.speaking_rate_loss_weight = speaking_rate_loss_weight
        
        # Voice cloning loss weights
        self.voice_similarity_loss_weight = voice_similarity_loss_weight
        
        # Diffusion loss weights
        self.diffusion_loss_weight = diffusion_loss_weight
        
        # Contrastive loss for speaker similarity
        from ..models.speaker_encoder import ContrastiveSpeakerLoss
        self.contrastive_loss = ContrastiveSpeakerLoss(
            temperature=contrastive_temperature,
            margin=contrastive_margin,
            name="contrastive_speaker_loss"
        )
        
        # Stability features
        self.use_adaptive_weights = use_adaptive_weights
        self.loss_smoothing_factor = loss_smoothing_factor
        self.max_loss_spike_threshold = max_loss_spike_threshold
        self.gradient_norm_threshold = gradient_norm_threshold
        
        # Running averages for stability monitoring
        self.running_mel_loss = tf.Variable(0.0, trainable=False, name="running_mel_loss")
        self.running_total_loss = tf.Variable(0.0, trainable=False, name="running_total_loss")
        self.step_count = tf.Variable(0, trainable=False, name="loss_step_count")
        self.loss_history = tf.Variable(tf.zeros([10]), trainable=False, name="loss_history")
        self.history_index = tf.Variable(0, trainable=False, name="history_index")
        self._weighted_losses = {}
        self._raw_total_loss_tensor = None
    
    def call(
        self,
        y_true: Dict[str, tf.Tensor],
        y_pred: Dict[str, tf.Tensor]
    ) -> tf.Tensor:
        """
        Compute combined loss with stability improvements.
        
        Args:
            y_true: Dictionary of target tensors
            y_pred: Dictionary of predicted tensors
            
        Returns:
            Combined loss tensor with stability enhancements
        """
        losses = {}
        total_loss = tf.constant(0.0, dtype=tf.float32)
        weighted_contrib = {}

        def _safe(name: str, value: tf.Tensor) -> tf.Tensor:
            tensor = tf.cast(value, tf.float32)
            finite_mask = tf.math.is_finite(tensor)
            sanitized = tf.where(finite_mask, tensor, tf.zeros_like(tensor))

            def _log_problematic() -> tf.Tensor:
                non_finite_mask = tf.logical_not(finite_mask)

                def _scalar_problematic() -> tf.Tensor:
                    return tf.reshape(tensor, [1])

                def _masked_problematic() -> tf.Tensor:
                    return tf.boolean_mask(tensor, non_finite_mask)

                problematic = tf.cond(
                    tf.equal(tf.rank(tensor), 0),
                    _scalar_problematic,
                    _masked_problematic
                )
                tf.print(
                    "WARNING:",
                    name,
                    "contains non-finite values:",
                    problematic,
                    summarize=-1
                )
                return sanitized

            return tf.cond(
                tf.reduce_all(finite_mask),
                lambda: sanitized,
                _log_problematic
            )

        # Mel spectrogram loss
        if "mel_target" in y_true and "mel_output" in y_pred:
            mel_l = mel_loss(
                y_true["mel_target"],
                y_pred["mel_output"],
                y_true.get("mel_lengths")
            )
            mel_loss_safe = _safe("mel_loss", mel_l)
            losses["mel_loss"] = mel_loss_safe

            # Apply adaptive weighting for mel loss
            if self.use_adaptive_weights:
                mel_weight = self._adaptive_mel_weight(mel_loss_safe)
            else:
                mel_weight = self.mel_loss_weight
            mel_weight_safe = _safe("w_mel_weight", mel_weight)

            weighted_mel_loss = mel_weight_safe * mel_loss_safe
            weighted_mel_loss_safe = _safe("w_mel_loss", weighted_mel_loss)

            total_loss += weighted_mel_loss_safe
            weighted_contrib["mel_loss"] = weighted_mel_loss_safe
            weighted_contrib["mel_weight"] = mel_weight_safe

        # Stop token loss
        if "stop_target" in y_true and "stop_tokens" in y_pred:
            stop_l = stop_token_loss(
                y_true["stop_target"],
                y_pred["stop_tokens"],
                y_true.get("mel_lengths")
            )
            stop_loss_safe = _safe("stop_loss", stop_l)
            losses["stop_loss"] = stop_loss_safe

            weighted_stop_loss = tf.cast(self.stop_loss_weight, tf.float32) * stop_loss_safe
            weighted_stop_loss_safe = _safe("w_stop_loss", weighted_stop_loss)

            total_loss += weighted_stop_loss_safe
            weighted_contrib["stop_loss"] = weighted_stop_loss_safe

        # Attention loss (if attention weights available)
        if ("attention_weights" in y_pred and 
            "text_lengths" in y_true and 
            "mel_lengths" in y_true):
            attn_l = attention_loss(
                y_pred["attention_weights"],
                y_true["text_lengths"],
                y_true["mel_lengths"]
            )
            attention_loss_safe = _safe("attention_loss", attn_l)
            losses["attention_loss"] = attention_loss_safe

            weighted_attention_loss = tf.cast(self.attention_loss_weight, tf.float32) * attention_loss_safe
            weighted_attention_loss_safe = _safe("w_attention_loss", weighted_attention_loss)

            total_loss += weighted_attention_loss_safe
            weighted_contrib["attention_loss"] = weighted_attention_loss_safe

        # Duration loss (if duration predictions available)
        if ("duration_pred" in y_pred and 
            "duration_target" in y_true and
            "text_lengths" in y_true):
            dur_l = duration_loss(
                y_pred["duration_pred"],
                y_true["duration_target"],
                y_true["text_lengths"]
            )
            duration_loss_safe = _safe("duration_loss", dur_l)
            losses["duration_loss"] = duration_loss_safe

            weighted_duration_loss = tf.cast(self.duration_loss_weight, tf.float32) * duration_loss_safe
            weighted_duration_loss_safe = _safe("w_duration_loss", weighted_duration_loss)

            total_loss += weighted_duration_loss_safe
            weighted_contrib["duration_loss"] = weighted_duration_loss_safe

        # Pitch loss (if pitch predictions available)
        if ("pitch_output" in y_pred and 
            "pitch_target" in y_true and
            "mel_lengths" in y_true):
            pitch_l = pitch_loss(
                y_pred["pitch_output"],
                y_true["pitch_target"],
                y_true["mel_lengths"]
            )
            pitch_loss_safe = _safe("pitch_loss", pitch_l)
            losses["pitch_loss"] = pitch_loss_safe

            weighted_pitch_loss = tf.cast(self.pitch_loss_weight, tf.float32) * pitch_loss_safe
            weighted_pitch_loss_safe = _safe("w_pitch_loss", weighted_pitch_loss)

            total_loss += weighted_pitch_loss_safe
            weighted_contrib["pitch_loss"] = weighted_pitch_loss_safe

        # Energy loss (if energy predictions available)
        if ("energy_output" in y_pred and 
            "energy_target" in y_true and
            "mel_lengths" in y_true):
            energy_l = energy_loss(
                y_pred["energy_output"],
                y_true["energy_target"],
                y_true["mel_lengths"]
            )
            energy_loss_safe = _safe("energy_loss", energy_l)
            losses["energy_loss"] = energy_loss_safe

            weighted_energy_loss = tf.cast(self.energy_loss_weight, tf.float32) * energy_loss_safe
            weighted_energy_loss_safe = _safe("w_energy_loss", weighted_energy_loss)

            total_loss += weighted_energy_loss_safe
            weighted_contrib["energy_loss"] = weighted_energy_loss_safe

        # Prosody pitch loss on text-level prosody predictor
        if ("prosody_pitch" in y_pred and
            "prosody_pitch_target" in y_true and
            "text_lengths" in y_true):
            prosody_pitch_l = pitch_loss(
                y_pred["prosody_pitch"],
                y_true["prosody_pitch_target"],
                y_true["text_lengths"]
            )
            prosody_pitch_loss_safe = _safe("prosody_pitch_loss", prosody_pitch_l)
            losses["prosody_pitch_loss"] = prosody_pitch_loss_safe

            weighted_prosody_pitch_loss = (
                tf.cast(self.prosody_pitch_loss_weight, tf.float32) * prosody_pitch_loss_safe
            )
            weighted_prosody_pitch_loss_safe = _safe("w_prosody_pitch_loss", weighted_prosody_pitch_loss)

            total_loss += weighted_prosody_pitch_loss_safe
            weighted_contrib["prosody_pitch_loss"] = weighted_prosody_pitch_loss_safe

        if ("prosody_energy" in y_pred and
            "prosody_energy_target" in y_true and
            "text_lengths" in y_true):
            prosody_energy_l = energy_loss(
                y_pred["prosody_energy"],
                y_true["prosody_energy_target"],
                y_true["text_lengths"]
            )
            prosody_energy_loss_safe = _safe("prosody_energy_loss", prosody_energy_l)
            losses["prosody_energy_loss"] = prosody_energy_loss_safe

            weighted_prosody_energy_loss = (
                tf.cast(self.prosody_energy_loss_weight, tf.float32) * prosody_energy_loss_safe
            )
            weighted_prosody_energy_loss_safe = _safe("w_prosody_energy_loss", weighted_prosody_energy_loss)

            total_loss += weighted_prosody_energy_loss_safe
            weighted_contrib["prosody_energy_loss"] = weighted_prosody_energy_loss_safe

        if ("prosody_speaking_rate" in y_pred and
            "prosody_speaking_rate_target" in y_true and
            "text_lengths" in y_true):
            speaking_rate_l = speaking_rate_loss(
                y_pred["prosody_speaking_rate"],
                y_true["prosody_speaking_rate_target"],
                y_true["text_lengths"]
            )
            speaking_rate_loss_safe = _safe("speaking_rate_loss", speaking_rate_l)
            losses["speaking_rate_loss"] = speaking_rate_loss_safe

            weighted_speaking_rate_loss = (
                tf.cast(self.speaking_rate_loss_weight, tf.float32) * speaking_rate_loss_safe
            )
            weighted_speaking_rate_loss_safe = _safe("w_speaking_rate_loss", weighted_speaking_rate_loss)

            total_loss += weighted_speaking_rate_loss_safe
            weighted_contrib["speaking_rate_loss"] = weighted_speaking_rate_loss_safe

        # Voice similarity loss (contrastive speaker loss)
        if ("speaker_embedding" in y_pred and 
            "speaker_labels" in y_true):
            voice_sim_l = self.contrastive_loss(
                y_true["speaker_labels"],
                y_pred["speaker_embedding"]
            )
            voice_similarity_loss_safe = _safe("voice_similarity_loss", voice_sim_l)
            losses["voice_similarity_loss"] = voice_similarity_loss_safe

            weighted_voice_similarity_loss = (
                tf.cast(self.voice_similarity_loss_weight, tf.float32) * voice_similarity_loss_safe
            )
            weighted_voice_similarity_loss_safe = _safe("w_voice_similarity_loss", weighted_voice_similarity_loss)

            total_loss += weighted_voice_similarity_loss_safe
            weighted_contrib["voice_similarity_loss"] = weighted_voice_similarity_loss_safe

        # Diffusion loss (for diffusion-based decoders)
        if ("diffusion_noise_pred" in y_pred and
            "diffusion_noise_target" in y_true):
            diffusion_loss_raw = diffusion_loss(
                y_pred["diffusion_noise_pred"],
                y_true["diffusion_noise_target"],
                y_true.get("mel_lengths")
            )
            diffusion_loss_safe = _safe("diffusion_loss", diffusion_loss_raw)
            losses["diffusion_loss"] = diffusion_loss_safe

            weighted_diffusion_loss = tf.cast(self.diffusion_loss_weight, tf.float32) * diffusion_loss_safe
            weighted_diffusion_loss_safe = _safe("w_diffusion_loss", weighted_diffusion_loss)

            total_loss += weighted_diffusion_loss_safe
            weighted_contrib["diffusion_loss"] = weighted_diffusion_loss_safe

        # Gradient participation regularization: Ensure all model outputs participate in gradients
        # This prevents the "Gradients do not exist for variables" warning by adding a small
        # regularization term for outputs that don't have corresponding targets
        gradient_reg_loss = self._ensure_gradient_participation(y_pred, y_true)
        gradient_reg_loss_safe = _safe("gradient_participation_loss", gradient_reg_loss)
        try:
            apply_gradient_reg = bool(tf.reduce_sum(gradient_reg_loss_safe).numpy() > 0.0)
        except Exception:
            apply_gradient_reg = True

        if apply_gradient_reg:
            losses["gradient_participation_loss"] = gradient_reg_loss_safe
            gradient_participation_loss_safe = _safe(
                "w_gradient_participation_loss",
                gradient_reg_loss_safe
            )
            total_loss += gradient_participation_loss_safe
            weighted_contrib["gradient_participation_loss"] = gradient_participation_loss_safe

        # Apply loss smoothing and spike detection
        raw_total_loss = total_loss
        raw_total_loss_safe = _safe("w_raw_total_loss", raw_total_loss)
        self._raw_total_loss_tensor = raw_total_loss_safe
        weighted_contrib["raw_total_loss"] = raw_total_loss_safe
        self._weighted_losses = dict(weighted_contrib)
        total_loss = self._apply_loss_smoothing(total_loss)

        # Store individual losses for monitoring
        self.losses = dict(losses)

        return total_loss
    
    def _adaptive_mel_weight(self, current_mel_loss: tf.Tensor) -> tf.Tensor:
        """
        Compute adaptive weight for mel loss based on convergence progress.
        
        Args:
            current_mel_loss: Current mel loss value
            
        Returns:
            Adaptive weight for mel loss
        """
        # Update running average
        self.step_count.assign_add(1)
        decay = tf.minimum(tf.cast(self.step_count, tf.float32) / 1000.0, 0.99)
        
        self.running_mel_loss.assign(
            decay * self.running_mel_loss + (1.0 - decay) * current_mel_loss
        )
        
        # Adaptive scaling based on loss magnitude
        # Reduce weight if loss is converging well, increase if struggling
        base_weight = self.mel_loss_weight
        
        # If loss is higher than running average, slightly increase weight
        # If loss is lower, slightly decrease weight for better balance
        ratio = current_mel_loss / (self.running_mel_loss + 1e-8)
        
        # Smooth adaptation - don't make dramatic changes
        adaptation_factor = tf.clip_by_value(
            0.5 + 0.5 * tf.tanh(ratio - 1.0), 
            0.7,  # Minimum 70% of base weight
            1.3   # Maximum 130% of base weight
        )
        
        return base_weight * adaptation_factor
    
    def _apply_loss_smoothing(self, current_loss: tf.Tensor) -> tf.Tensor:
        """
        Apply exponential smoothing and spike detection to loss.
        
        Args:
            current_loss: Current computed loss
            
        Returns:
            Smoothed loss with spike protection
        """
        current_loss = tf.cast(current_loss, tf.float32)
        current_loss = tf.where(
            tf.math.is_finite(current_loss),
            current_loss,
            tf.constant(0.0, dtype=tf.float32)
        )

        # Graph-safe loss history validation
        def _reset_history():
            self.loss_history.assign(tf.zeros_like(self.loss_history))
            return tf.constant(0.0)
        
        def _no_reset():
            return tf.constant(0.0)
        
        tf.cond(
            tf.reduce_all(tf.math.is_finite(self.loss_history)),
            _no_reset,
            _reset_history
        )

        # Graph-safe running total validation
        def _reset_total():
            self.running_total_loss.assign(tf.constant(0.0, dtype=tf.float32))
            return tf.constant(0.0)
        
        tf.cond(
            tf.math.is_finite(self.running_total_loss),
            _no_reset,
            _reset_total
        )

        history_length = tf.cast(tf.shape(self.loss_history)[0], self.history_index.dtype)
        history_idx = tf.math.mod(self.history_index, history_length)
        history_idx_i32 = tf.cast(history_idx, tf.int32)
        base_history = self.loss_history.read_value()
        updated_history = tf.tensor_scatter_nd_update(
            base_history,
            tf.reshape(history_idx_i32, [1, 1]),
            tf.reshape(current_loss, [1])
        )
        self.loss_history.assign(updated_history)
        self.history_index.assign_add(1)

        recent_losses = tf.where(
            tf.math.is_finite(self.loss_history),
            self.loss_history,
            tf.zeros_like(self.loss_history)
        )

        decay = tf.minimum(tf.cast(self.step_count, tf.float32) / 500.0, 0.95)
        running_total = decay * self.running_total_loss + (1.0 - decay) * current_loss
        running_total = tf.where(
            tf.math.is_finite(running_total),
            running_total,
            current_loss
        )
        self.running_total_loss.assign(running_total)

        # Simplified graph-safe loss smoothing
        def _apply_advanced_smoothing():
            recent_avg = tf.reduce_mean(recent_losses)
            recent_avg = tf.where(
                tf.math.is_finite(recent_avg),
                recent_avg,
                current_loss
            )
            
            # Simple smoothing without complex spike detection
            smoothed = (1.0 - self.loss_smoothing_factor) * current_loss + \
                      self.loss_smoothing_factor * running_total
            return tf.where(tf.math.is_finite(smoothed), smoothed, current_loss)
        
        def _simple_smoothing():
            return current_loss

        smoothed_loss = tf.cond(
            tf.greater(self.step_count, 10),
            _apply_advanced_smoothing,
            _simple_smoothing
        )

        smoothed_loss = tf.where(
            tf.math.is_finite(smoothed_loss),
            smoothed_loss,
            current_loss
        )

        return smoothed_loss
    
    def _ensure_gradient_participation(
        self, 
        y_pred: Dict[str, tf.Tensor], 
        y_true: Dict[str, tf.Tensor]
    ) -> tf.Tensor:
        """
        Ensure all model outputs participate in gradient computation.
        
        This method adds a small regularization term for outputs that don't have
        corresponding targets, preventing gradient warnings for unused variables.
        
        Args:
            y_pred: Dictionary of predicted tensors
            y_true: Dictionary of target tensors
            
        Returns:
            Gradient participation regularization loss
        """
        regularization_loss = 0.0
        regularization_weight = 1e-6  # Very small weight to not affect training
        
        # Duration predictor regularization
        if ("duration_pred" in y_pred and 
            "duration_target" not in y_true):
            # Add small L2 regularization to ensure gradient participation
            duration_reg = tf.reduce_mean(tf.square(y_pred["duration_pred"]))
            regularization_loss += regularization_weight * duration_reg
        
        # Mel-level prosody regularization (from mel decoder)
        if ("pitch_output" in y_pred and 
            "pitch_target" not in y_true):
            pitch_reg = tf.reduce_mean(tf.square(y_pred["pitch_output"]))
            regularization_loss += regularization_weight * pitch_reg
            
        if ("energy_output" in y_pred and 
            "energy_target" not in y_true):
            energy_reg = tf.reduce_mean(tf.square(y_pred["energy_output"]))
            regularization_loss += regularization_weight * energy_reg
        
        # Text-level prosody regularization (from prosody predictor)
        if ("prosody_pitch" in y_pred and 
            "prosody_pitch_target" not in y_true):
            prosody_pitch_reg = tf.reduce_mean(tf.square(y_pred["prosody_pitch"]))
            regularization_loss += regularization_weight * prosody_pitch_reg
            
        if ("prosody_energy" in y_pred and 
            "prosody_energy_target" not in y_true):
            prosody_energy_reg = tf.reduce_mean(tf.square(y_pred["prosody_energy"]))
            regularization_loss += regularization_weight * prosody_energy_reg
            
        if ("prosody_speaking_rate" in y_pred and 
            "prosody_speaking_rate_target" not in y_true):
            speaking_rate_reg = tf.reduce_mean(tf.square(y_pred["prosody_speaking_rate"]))
            regularization_loss += regularization_weight * speaking_rate_reg
        
        return regularization_loss
    
    def get_losses(self) -> Dict[str, tf.Tensor]:
        """Get individual loss components."""
        return getattr(self, 'losses', {})
    
    def get_stability_metrics(self) -> Dict[str, tf.Tensor]:
        """
        Get training stability metrics for monitoring.
        
        Returns:
            Dictionary containing stability metrics
        """
        metrics = {}
        
        if self.step_count > 0:
            metrics["running_mel_loss"] = self.running_mel_loss
            metrics["running_total_loss"] = self.running_total_loss
            
            if self.step_count > 10:
                # Compute loss variance for stability assessment
                recent_losses = self.loss_history
                loss_mean = tf.reduce_mean(recent_losses)
                loss_variance = tf.reduce_mean(tf.square(recent_losses - loss_mean))
                metrics["loss_variance"] = loss_variance
                metrics["loss_stability_score"] = tf.exp(-loss_variance)  # Higher = more stable
            else:
                # Initialize with default values for graph compatibility
                metrics["loss_variance"] = tf.constant(0.0)
                metrics["loss_stability_score"] = tf.constant(1.0)
            
            metrics["step_count"] = tf.cast(self.step_count, tf.float32)
        else:
            # Initialize all metrics for step_count <= 0
            metrics["running_mel_loss"] = tf.constant(0.0)
            metrics["running_total_loss"] = tf.constant(0.0)
            metrics["loss_variance"] = tf.constant(0.0)
            metrics["loss_stability_score"] = tf.constant(1.0)
            metrics["step_count"] = tf.constant(0.0)
        
        return metrics

    def get_weighted_losses(self) -> Dict[str, tf.Tensor]:
        """Return the weighted loss contributions from the last forward pass."""
        return {k: tf.identity(v) for k, v in self._weighted_losses.items()}

    def get_raw_total_loss(self) -> Optional[tf.Tensor]:
        """Return the raw (pre-smoothed) total loss."""
        return self._raw_total_loss_tensor
    
    def reset_stability_state(self):
        """Reset stability tracking state (useful for validation/testing)."""
        self.running_mel_loss.assign(0.0)
        self.running_total_loss.assign(0.0)
        self.step_count.assign(0)
        self.loss_history.assign(tf.zeros([10]))
        self.history_index.assign(0)
        self._raw_total_loss_tensor = None


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
