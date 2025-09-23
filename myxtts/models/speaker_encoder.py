"""
Pre-trained speaker encoder implementations for enhanced voice conditioning.

This module provides speaker encoder implementations including ECAPA-TDNN,
Resemblyzer-style encoders, and Coqui speaker models for improved voice cloning.
"""

import tensorflow as tf
from typing import Optional, Tuple, Dict
import numpy as np


class ContrastiveSpeakerLoss(tf.keras.losses.Loss):
    """
    Contrastive loss for speaker similarity with GE2E-style implementation.
    
    This implements a contrastive loss function that encourages similar speakers
    to have close embeddings while pushing different speakers apart.
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        margin: float = 0.2,
        name: str = "contrastive_speaker_loss"
    ):
        """
        Initialize contrastive speaker loss.
        
        Args:
            temperature: Temperature for scaling similarities
            margin: Margin for contrastive learning
            name: Loss name
        """
        super().__init__(name=name)
        self.temperature = temperature
        self.margin = margin
    
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute contrastive speaker loss.
        
        Args:
            y_true: Speaker labels [batch_size]
            y_pred: Speaker embeddings [batch_size, embedding_dim]
            
        Returns:
            Contrastive loss value
        """
        # Normalize embeddings
        embeddings = tf.nn.l2_normalize(y_pred, axis=1)
        
        # Compute pairwise similarities
        similarities = tf.matmul(embeddings, embeddings, transpose_b=True)
        
        # Scale by temperature
        similarities = similarities / self.temperature
        
        # Create positive and negative masks
        labels = tf.cast(y_true, tf.int32)
        labels_eq = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
        
        # Mask out diagonal (self-similarity)
        batch_size = tf.shape(embeddings)[0]
        mask = tf.logical_not(tf.eye(batch_size, dtype=tf.bool))
        labels_eq = tf.logical_and(labels_eq, mask)
        
        # Positive pairs (same speaker)
        positive_mask = tf.cast(labels_eq, tf.float32)
        
        # Negative pairs (different speakers)
        negative_mask = tf.cast(tf.logical_and(mask, tf.logical_not(labels_eq)), tf.float32)
        
        # Compute positive loss (minimize distance between same speakers)
        positive_similarities = similarities * positive_mask
        positive_loss = -tf.reduce_sum(positive_similarities * positive_mask) / tf.maximum(tf.reduce_sum(positive_mask), 1.0)
        
        # Compute negative loss (maximize distance between different speakers)
        negative_similarities = similarities * negative_mask
        negative_loss = tf.reduce_sum(tf.nn.relu(negative_similarities - self.margin) * negative_mask) / tf.maximum(tf.reduce_sum(negative_mask), 1.0)
        
        # Combined loss
        total_loss = positive_loss + negative_loss
        
        return total_loss


class PretrainedSpeakerEncoder(tf.keras.layers.Layer):
    """
    Pre-trained speaker encoder wrapper supporting multiple architectures.
    
    Supports ECAPA-TDNN, Resemblyzer-style encoders, and other pre-trained
    speaker recognition models for enhanced voice conditioning.
    """
    
    def __init__(
        self,
        config,
        pretrained_path: Optional[str] = None,
        freeze_weights: bool = True,
        embedding_dim: int = 256,
        encoder_type: str = "ecapa_tdnn",
        name: str = "pretrained_speaker_encoder"
    ):
        """
        Initialize pre-trained speaker encoder.
        
        Args:
            config: Model configuration
            pretrained_path: Path to pre-trained weights
            freeze_weights: Whether to freeze encoder weights
            embedding_dim: Output embedding dimension
            encoder_type: Type of encoder ("ecapa_tdnn", "resemblyzer", "coqui")
            name: Layer name
        """
        super().__init__(name=name)
        
        self.config = config
        self.pretrained_path = pretrained_path
        self.freeze_weights = freeze_weights
        self.embedding_dim = embedding_dim
        self.encoder_type = encoder_type.lower()
        
        # Build encoder based on type
        if self.encoder_type == "ecapa_tdnn":
            self._build_ecapa_tdnn()
        elif self.encoder_type == "resemblyzer":
            self._build_resemblyzer()
        elif self.encoder_type == "coqui":
            self._build_coqui_encoder()
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
    
    def _build_ecapa_tdnn(self):
        """Build ECAPA-TDNN style speaker encoder."""
        # Frame-level feature extraction
        self.frame_conv = tf.keras.Sequential([
            tf.keras.layers.Conv1D(512, 5, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(512, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
        ])
        
        # SE-Res2Block layers (simplified version)
        self.se_res_blocks = [
            self._build_se_res2block(512, 512, dilation=1),
            self._build_se_res2block(512, 512, dilation=2),
            self._build_se_res2block(512, 512, dilation=3),
        ]
        
        # Aggregation layer
        self.aggregation = tf.keras.layers.Conv1D(1536, 1, activation='relu')
        
        # Statistics pooling
        self.stats_pooling = tf.keras.layers.Lambda(self._statistical_pooling)
        
        # Final embedding layers
        self.embedding_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(self.embedding_dim, activation=None),
        ])
    
    def _build_se_res2block(self, in_channels: int, out_channels: int, dilation: int = 1):
        """Build SE-Res2Block (simplified version)."""
        return tf.keras.Sequential([
            tf.keras.layers.Conv1D(
                out_channels, 3, 
                padding='same', 
                dilation_rate=dilation,
                activation='relu'
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(out_channels, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
        ])
    
    def _statistical_pooling(self, x):
        """Statistical pooling: compute mean and std."""
        mean = tf.reduce_mean(x, axis=1)
        variance = tf.reduce_mean(tf.square(x - tf.expand_dims(mean, 1)), axis=1)
        std = tf.sqrt(variance + 1e-8)
        return tf.concat([mean, std], axis=-1)
    
    def _build_resemblyzer(self):
        """Build Resemblyzer-style speaker encoder."""
        # LSTM-based encoder similar to Resemblyzer
        self.lstm_layers = tf.keras.Sequential([
            tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.1),
            tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.1),
            tf.keras.layers.LSTM(256, return_sequences=False, dropout=0.1),
        ])
        
        # Projection to embedding dimension
        self.projection = tf.keras.layers.Dense(self.embedding_dim, activation=None)
    
    def _build_coqui_encoder(self):
        """Build Coqui-style speaker encoder."""
        # Convolutional front-end
        self.conv_frontend = tf.keras.Sequential([
            tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(512, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
        ])
        
        # Global average pooling and projection
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.projection = tf.keras.layers.Dense(self.embedding_dim, activation='tanh')
    
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass through speaker encoder.
        
        Args:
            inputs: Input mel spectrogram [batch, time, n_mels]
            training: Training mode flag
            
        Returns:
            Speaker embedding [batch, embedding_dim]
        """
        x = inputs
        
        if self.encoder_type == "ecapa_tdnn":
            # Transpose for Conv1D (batch, time, features) -> (batch, features, time)
            x = tf.transpose(x, [0, 2, 1])
            
            # Frame-level processing
            x = self.frame_conv(x, training=training)
            
            # SE-Res2Blocks
            for block in self.se_res_blocks:
                residual = x
                x = block(x, training=training)
                x = x + residual  # Residual connection
            
            # Aggregation
            x = self.aggregation(x, training=training)
            
            # Statistical pooling
            x = self.stats_pooling(x)
            
            # Final embedding
            embedding = self.embedding_layers(x, training=training)
            
        elif self.encoder_type == "resemblyzer":
            # LSTM processing
            x = self.lstm_layers(x, training=training)
            embedding = self.projection(x, training=training)
            
        elif self.encoder_type == "coqui":
            # Transpose for Conv1D
            x = tf.transpose(x, [0, 2, 1])
            
            # Convolutional processing
            x = self.conv_frontend(x, training=training)
            
            # Global pooling and projection
            x = self.global_pool(x)
            embedding = self.projection(x, training=training)
        
        # L2 normalize embedding
        embedding = tf.nn.l2_normalize(embedding, axis=-1)
        
        return embedding
    
    def build(self, input_shape):
        """Build the layer with given input shape."""
        super().build(input_shape)
        
        # Load pre-trained weights if available
        if self.pretrained_path and tf.io.gfile.exists(self.pretrained_path):
            try:
                self.load_weights(self.pretrained_path)
                print(f"Loaded pre-trained weights from {self.pretrained_path}")
            except Exception as e:
                print(f"Warning: Could not load pre-trained weights: {e}")
        
        # Freeze weights if requested
        if self.freeze_weights:
            self.trainable = False
            print(f"Frozen speaker encoder weights (type: {self.encoder_type})")