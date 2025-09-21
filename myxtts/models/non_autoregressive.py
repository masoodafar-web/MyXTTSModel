"""
Non-autoregressive decoder implementations for MyXTTS.

This module provides modern decoding strategies including FastSpeech-style
non-autoregressive decoding for faster and smoother inference.
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Optional, Tuple
from .layers import TransformerBlock, PositionalEncoding
from ..config.config import ModelConfig


class DurationPredictor(tf.keras.layers.Layer):
    """
    Duration predictor for non-autoregressive decoding.
    
    Predicts the duration (number of mel frames) for each text token
    to enable parallel mel generation.
    """
    
    def __init__(
        self,
        config: ModelConfig,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout_rate: float = 0.1,
        name: str = "duration_predictor"
    ):
        """
        Initialize duration predictor.
        
        Args:
            config: Model configuration
            hidden_dim: Hidden dimension for predictor layers
            num_layers: Number of predictor layers
            dropout_rate: Dropout rate
            name: Layer name
        """
        super().__init__(name=name)
        
        self.config = config
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Convolutional layers for feature extraction
        self.conv_layers = []
        for i in range(num_layers):
            conv_layer = tf.keras.layers.Conv1D(
                hidden_dim,
                kernel_size=3,
                padding='same',
                activation='relu',
                name=f'conv_{i}'
            )
            self.conv_layers.append(conv_layer)
        
        # Dropout
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
        # Output projection
        self.duration_projection = tf.keras.layers.Dense(
            1,
            activation='relu',  # Duration should be positive
            name='duration_projection'
        )
    
    def call(
        self,
        encoder_output: tf.Tensor,
        training: bool = False
    ) -> tf.Tensor:
        """
        Predict durations for each text token.
        
        Args:
            encoder_output: Text encoder output [batch, text_len, d_model]
            training: Training mode flag
            
        Returns:
            Duration predictions [batch, text_len, 1]
        """
        x = encoder_output
        
        # Apply convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            x = self.dropout(x, training=training)
        
        # Predict durations
        durations = self.duration_projection(x)
        
        return durations


class LengthRegulator(tf.keras.layers.Layer):
    """
    Length regulator for expanding text representations based on durations.
    
    Converts text-level representations to mel-level representations
    by expanding each text token according to its predicted duration.
    """
    
    def __init__(self, name: str = "length_regulator"):
        """Initialize length regulator."""
        super().__init__(name=name)
    
    def call(
        self,
        encoder_output: tf.Tensor,
        durations: tf.Tensor,
        max_length: Optional[int] = None
    ) -> tf.Tensor:
        """
        Expand encoder output based on durations.
        
        Args:
            encoder_output: Text encoder output [batch, text_len, d_model]
            durations: Duration predictions [batch, text_len, 1]
            max_length: Maximum output length (for padding)
            
        Returns:
            Expanded representations [batch, mel_len, d_model]
        """
        batch_size = tf.shape(encoder_output)[0]
        text_len = tf.shape(encoder_output)[1]
        d_model = tf.shape(encoder_output)[2]
        
        # Round durations to integers
        durations = tf.maximum(tf.round(durations), 1.0)  # Minimum duration of 1
        durations = tf.cast(durations, tf.int32)
        durations = tf.squeeze(durations, axis=-1)  # [batch, text_len]
        
        # Calculate output lengths for each batch
        output_lengths = tf.reduce_sum(durations, axis=1)  # [batch]
        
        if max_length is None:
            max_length = tf.reduce_max(output_lengths)
        
        # Use tf.RaggedTensor for efficient dynamic length handling
        expanded_list = []
        
        for i in range(batch_size):
            # Get durations for this batch item
            batch_durations = durations[i]  # [text_len]
            batch_encoder = encoder_output[i]  # [text_len, d_model]
            
            # Create indices for expansion
            indices = tf.repeat(tf.range(text_len), batch_durations)
            
            # Expand based on indices
            expanded = tf.gather(batch_encoder, indices)  # [expanded_len, d_model]
            
            # Pad or truncate to max_length
            current_len = tf.shape(expanded)[0]
            if current_len < max_length:
                padding = tf.zeros([max_length - current_len, d_model], dtype=encoder_output.dtype)
                expanded = tf.concat([expanded, padding], axis=0)
            else:
                expanded = expanded[:max_length, :]
            
            expanded_list.append(expanded)
        
        # Stack all batches
        expanded_output = tf.stack(expanded_list, axis=0)  # [batch, max_length, d_model]
        
        return expanded_output


class NonAutoregressiveDecoder(tf.keras.layers.Layer):
    """
    Non-autoregressive mel decoder for parallel mel generation.
    
    Based on FastSpeech architecture for faster inference compared to
    the autoregressive decoder.
    """
    
    def __init__(
        self,
        config: ModelConfig,
        name: str = "non_autoregressive_decoder"
    ):
        """
        Initialize non-autoregressive decoder.
        
        Args:
            config: Model configuration
            name: Layer name
        """
        super().__init__(name=name)
        
        self.config = config
        self.d_model = config.decoder_dim
        
        # Duration predictor
        self.duration_predictor = DurationPredictor(config)
        
        # Length regulator
        self.length_regulator = LengthRegulator()
        
        # Encoder to decoder projection (to handle dimension mismatch)
        self.encoder_to_decoder_projection = tf.keras.layers.Dense(
            self.d_model,
            name="encoder_to_decoder_projection"
        )
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            self.d_model,
            max_length=2000,
            name="positional_encoding"
        )
        
        # Transformer decoder blocks (without causal masking)
        self.transformer_blocks = [
            TransformerBlock(
                self.d_model,
                config.decoder_heads,
                self.d_model * 4,
                dropout=0.1,
                is_decoder=False,  # Non-causal attention
                use_gradient_checkpointing=config.enable_gradient_checkpointing,
                name=f"transformer_block_{i}"
            )
            for i in range(config.decoder_layers)
        ]
        
        # Output projections
        self.mel_projection = tf.keras.layers.Dense(
            config.n_mels,
            name="mel_projection"
        )
        
        # Speaker conditioning projection
        self.speaker_projection = tf.keras.layers.Dense(
            self.d_model,
            name="speaker_projection"
        )
        
        self.dropout = tf.keras.layers.Dropout(0.1)
    
    def call(
        self,
        encoder_output: tf.Tensor,
        durations: Optional[tf.Tensor] = None,
        speaker_embedding: Optional[tf.Tensor] = None,
        max_length: Optional[int] = None,
        training: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Forward pass for non-autoregressive decoding.
        
        Args:
            encoder_output: Text encoder output [batch, text_len, d_model]
            durations: Ground truth durations for training [batch, text_len, 1]
            speaker_embedding: Speaker embedding [batch, speaker_dim]
            max_length: Maximum output length
            training: Training mode flag
            
        Returns:
            Tuple of (mel_output, duration_predictions)
        """
        # Predict durations
        duration_pred = self.duration_predictor(encoder_output, training=training)
        
        # Use ground truth durations during training, predictions during inference
        if training and durations is not None:
            durations_to_use = durations
        else:
            durations_to_use = duration_pred
        
        # Expand encoder output based on durations
        expanded_output = self.length_regulator(
            encoder_output,
            durations_to_use,
            max_length=max_length
        )
        
        # Project to decoder dimension
        expanded_output = self.encoder_to_decoder_projection(expanded_output)
        
        # Add speaker conditioning if provided
        if speaker_embedding is not None:
            # Broadcast speaker embedding to all time steps
            batch_size = tf.shape(expanded_output)[0]
            mel_len = tf.shape(expanded_output)[1]
            
            speaker_expanded = tf.expand_dims(speaker_embedding, 1)
            speaker_expanded = tf.tile(speaker_expanded, [1, mel_len, 1])
            
            # Concatenate and project
            expanded_output = tf.concat([expanded_output, speaker_expanded], axis=-1)
            expanded_output = self.speaker_projection(expanded_output)
        
        # Add positional encoding (after speaker conditioning to ensure correct dimensions)
        x = self.positional_encoding(expanded_output)
        x = self.dropout(x, training=training)
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)
        
        # Generate mel output
        mel_output = self.mel_projection(x)
        
        return mel_output, duration_pred


class DecoderStrategy(tf.keras.layers.Layer):
    """
    Decoder strategy interface for switching between autoregressive and non-autoregressive decoding.
    
    Provides a unified interface for different decoding strategies.
    """
    
    def __init__(
        self,
        config: ModelConfig,
        strategy: str = "autoregressive",
        name: str = "decoder_strategy"
    ):
        """
        Initialize decoder strategy.
        
        Args:
            config: Model configuration
            strategy: Decoding strategy ("autoregressive", "non_autoregressive")
            name: Layer name
        """
        super().__init__(name=name)
        
        self.config = config
        self.strategy = strategy
        
        if strategy == "autoregressive":
            # Import here to avoid circular imports
            from .xtts import MelDecoder
            self.decoder = MelDecoder(config, name="autoregressive_decoder")
        elif strategy == "non_autoregressive":
            self.decoder = NonAutoregressiveDecoder(config, name="non_autoregressive_decoder")
        else:
            raise ValueError(f"Unsupported decoder strategy: {strategy}")
    
    def call(
        self,
        decoder_inputs: Optional[tf.Tensor],
        encoder_output: tf.Tensor,
        durations: Optional[tf.Tensor] = None,
        speaker_embedding: Optional[tf.Tensor] = None,
        encoder_mask: Optional[tf.Tensor] = None,
        decoder_mask: Optional[tf.Tensor] = None,
        max_length: Optional[int] = None,
        training: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor, Optional[tf.Tensor]]:
        """
        Forward pass using selected strategy.
        
        Args:
            decoder_inputs: Previous mel frames (for autoregressive) [batch, mel_len, n_mels]
            encoder_output: Text encoder output [batch, text_len, d_model]
            durations: Duration targets/predictions [batch, text_len, 1]
            speaker_embedding: Speaker embedding [batch, speaker_dim]
            encoder_mask: Encoder attention mask
            decoder_mask: Decoder attention mask
            max_length: Maximum output length
            training: Training mode flag
            
        Returns:
            Tuple of (mel_output, stop_tokens_or_durations, optional_attention_weights)
        """
        if self.strategy == "autoregressive":
            if decoder_inputs is None:
                raise ValueError("decoder_inputs required for autoregressive strategy")
            
            mel_output, stop_tokens = self.decoder(
                decoder_inputs,
                encoder_output,
                speaker_embedding=speaker_embedding,
                encoder_mask=encoder_mask,
                decoder_mask=decoder_mask,
                training=training
            )
            return mel_output, stop_tokens, None
            
        elif self.strategy == "non_autoregressive":
            mel_output, duration_pred = self.decoder(
                encoder_output,
                durations=durations,
                speaker_embedding=speaker_embedding,
                max_length=max_length,
                training=training
            )
            return mel_output, duration_pred, None
            
        else:
            raise ValueError(f"Unsupported decoder strategy: {self.strategy}")