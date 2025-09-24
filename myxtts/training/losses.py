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
    
    # Apply gradient clipping at the loss level for additional stability
    loss = tf.clip_by_value(loss, 0.0, 100.0)  # Prevent extreme loss values
    
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
    # L1 loss for pitch (more robust than L2)
    loss = tf.abs(predicted_pitch - target_pitch)
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
    # L1 loss for energy (more robust than L2)
    loss = tf.abs(predicted_energy - target_energy)
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
        mel_loss_weight: float = 45.0,
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
        total_loss = 0.0
        
        # Mel spectrogram loss
        if "mel_target" in y_true and "mel_output" in y_pred:
            mel_l = mel_loss(
                y_true["mel_target"],
                y_pred["mel_output"],
                y_true.get("mel_lengths")
            )
            losses["mel_loss"] = mel_l
            
            # Apply adaptive weighting for mel loss
            if self.use_adaptive_weights:
                mel_weight = self._adaptive_mel_weight(mel_l)
            else:
                mel_weight = self.mel_loss_weight
                
            total_loss += mel_weight * mel_l
        
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
        
        # Pitch loss (if pitch predictions available)
        if ("pitch_output" in y_pred and 
            "pitch_target" in y_true and
            "mel_lengths" in y_true):
            pitch_l = pitch_loss(
                y_pred["pitch_output"],
                y_true["pitch_target"],
                y_true["mel_lengths"]
            )
            losses["pitch_loss"] = pitch_l
            total_loss += self.pitch_loss_weight * pitch_l

        # Energy loss (if energy predictions available)
        if ("energy_output" in y_pred and 
            "energy_target" in y_true and
            "mel_lengths" in y_true):
            energy_l = energy_loss(
                y_pred["energy_output"],
                y_true["energy_target"],
                y_true["mel_lengths"]
            )
            losses["energy_loss"] = energy_l
            total_loss += self.energy_loss_weight * energy_l

        # Prosody pitch loss on text-level prosody predictor
        if ("prosody_pitch" in y_pred and
            "prosody_pitch_target" in y_true and
            "text_lengths" in y_true):
            prosody_pitch_l = pitch_loss(
                y_pred["prosody_pitch"],
                y_true["prosody_pitch_target"],
                y_true["text_lengths"]
            )
            losses["prosody_pitch_loss"] = prosody_pitch_l
            total_loss += self.prosody_pitch_loss_weight * prosody_pitch_l

        if ("prosody_energy" in y_pred and
            "prosody_energy_target" in y_true and
            "text_lengths" in y_true):
            prosody_energy_l = energy_loss(
                y_pred["prosody_energy"],
                y_true["prosody_energy_target"],
                y_true["text_lengths"]
            )
            losses["prosody_energy_loss"] = prosody_energy_l
            total_loss += self.prosody_energy_loss_weight * prosody_energy_l

        if ("prosody_speaking_rate" in y_pred and
            "prosody_speaking_rate_target" in y_true and
            "text_lengths" in y_true):
            speaking_rate_l = speaking_rate_loss(
                y_pred["prosody_speaking_rate"],
                y_true["prosody_speaking_rate_target"],
                y_true["text_lengths"]
            )
            losses["speaking_rate_loss"] = speaking_rate_l
            total_loss += self.speaking_rate_loss_weight * speaking_rate_l
        
        # Voice similarity loss (contrastive speaker loss)
        if ("speaker_embedding" in y_pred and 
            "speaker_labels" in y_true):
            voice_sim_l = self.contrastive_loss(
                y_true["speaker_labels"],
                y_pred["speaker_embedding"]
            )
            losses["voice_similarity_loss"] = voice_sim_l
            total_loss += self.voice_similarity_loss_weight * voice_sim_l
        
        # Diffusion loss (for diffusion-based decoders)
        if ("diffusion_noise_pred" in y_pred and 
            "diffusion_noise_target" in y_true):
            diff_l = diffusion_loss(
                y_pred["diffusion_noise_pred"],
                y_true["diffusion_noise_target"],
                y_true.get("mel_lengths")
            )
            losses["diffusion_loss"] = diff_l
            total_loss += self.diffusion_loss_weight * diff_l
        
        # Gradient participation regularization: Ensure all model outputs participate in gradients
        # This prevents the "Gradients do not exist for variables" warning by adding a small
        # regularization term for outputs that don't have corresponding targets
        gradient_reg_loss = self._ensure_gradient_participation(y_pred, y_true)
        if gradient_reg_loss > 0:
            losses["gradient_participation_loss"] = gradient_reg_loss
            total_loss += gradient_reg_loss
        
        # Apply loss smoothing and spike detection
        total_loss = self._apply_loss_smoothing(total_loss)
        
        # Store individual losses for monitoring
        self.losses = losses
        
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
        # Update loss history for spike detection
        history_idx = self.history_index % 10
        self.loss_history = tf.tensor_scatter_nd_update(
            self.loss_history,
            [[history_idx]],
            [current_loss]
        )
        self.history_index.assign_add(1)
        
        # Compute running total loss average
        decay = tf.minimum(tf.cast(self.step_count, tf.float32) / 500.0, 0.95)
        self.running_total_loss.assign(
            decay * self.running_total_loss + (1.0 - decay) * current_loss
        )
        
        # Spike detection: if current loss is much higher than recent average
        if self.step_count > 10:
            recent_avg = tf.reduce_mean(self.loss_history)
            spike_ratio = current_loss / (recent_avg + 1e-8)
            
            # If we detect a spike, apply dampening
            if spike_ratio > self.max_loss_spike_threshold:
                dampening_factor = tf.minimum(
                    self.max_loss_spike_threshold / spike_ratio,
                    1.0
                )
                smoothed_loss = current_loss * dampening_factor
            else:
                # Apply light exponential smoothing for stability
                smoothed_loss = (
                    (1.0 - self.loss_smoothing_factor) * current_loss +
                    self.loss_smoothing_factor * self.running_total_loss
                )
        else:
            smoothed_loss = current_loss
            
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
            
            metrics["step_count"] = tf.cast(self.step_count, tf.float32)
        
        return metrics
    
    def reset_stability_state(self):
        """Reset stability tracking state (useful for validation/testing)."""
        self.running_mel_loss.assign(0.0)
        self.running_total_loss.assign(0.0)
        self.step_count.assign(0)
        self.loss_history.assign(tf.zeros([10]))
        self.history_index.assign(0)


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
