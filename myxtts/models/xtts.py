"""
MyXTTS Model Implementation.

This module implements a TensorFlow-based XTTS (eXtreme Text-To-Speech) model
with multilingual support and voice cloning capabilities.
"""

import tensorflow as tf
import numpy as np
from typing import Dict, Tuple, Optional, List
import math

from .layers import (
    MultiHeadAttention,
    PositionalEncoding,
    TransformerBlock,
    FeedForward,
    ConvolutionalLayer
)
from ..config.config import ModelConfig


class TextEncoder(tf.keras.layers.Layer):
    """
    Text encoder for XTTS model.
    
    Encodes input text sequences into contextualized representations
    using transformer architecture.
    """
    
    def __init__(self, config: ModelConfig, name: str = "text_encoder"):
        """
        Initialize text encoder.
        
        Args:
            config: Model configuration
            name: Layer name
        """
        super().__init__(name=name)
        
        self.config = config
        self.d_model = config.text_encoder_dim
        
        # Embedding layers
        self.token_embedding = tf.keras.layers.Embedding(
            config.text_vocab_size,
            self.d_model,
            name="token_embedding"
        )
        
        self.positional_encoding = PositionalEncoding(
            self.d_model, 
            config.max_text_length,
            name="positional_encoding"
        )
        
        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(
                self.d_model,
                config.text_encoder_heads,
                self.d_model * 4,
                dropout=0.1,
                use_gradient_checkpointing=config.enable_gradient_checkpointing,
                name=f"transformer_block_{i}"
            )
            for i in range(config.text_encoder_layers)
        ]
        
        self.layer_norm = tf.keras.layers.LayerNormalization(name="layer_norm")
        self.dropout = tf.keras.layers.Dropout(0.1)
        
        # Duration predictor for alignment guidance
        self.duration_predictor = tf.keras.layers.Dense(
            1, 
            activation='relu',
            name="duration_predictor"
        )
    
    def call(
        self,
        inputs: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        training: bool = False,
        return_durations: bool = False
    ) -> tf.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            training: Training mode flag
            return_durations: Whether to return duration predictions
            
        Returns:
            Encoded text representations [batch, seq_len, d_model] or 
            tuple (encoded_text, duration_predictions)
        """
        # Token embedding
        x = self.token_embedding(inputs)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        
        # Positional encoding
        x = self.positional_encoding(x)
        x = self.dropout(x, training=training)
        
        # Transform attention mask for transformer layers
        if attention_mask is not None:
            # Expand mask dimensions for multi-head attention
            attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]
            attention_mask = tf.cast(attention_mask, tf.float32)
            attention_mask = (1.0 - attention_mask) * -1e9
        
        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(
                x, 
                self_attention_mask=attention_mask,
                training=training
            )
        
        # Final layer norm
        x = self.layer_norm(x)
        
        if return_durations:
            # Duration prediction
            duration_pred = self.duration_predictor(x)
            duration_pred = tf.squeeze(duration_pred, axis=-1)  # [batch, seq_len]
            return x, duration_pred
        else:
            return x


class AudioEncoder(tf.keras.layers.Layer):
    """
    Audio encoder for voice conditioning.
    
    Encodes reference audio features for voice cloning.
    """
    
    def __init__(self, config: ModelConfig, name: str = "audio_encoder"):
        """
        Initialize audio encoder.
        
        Args:
            config: Model configuration
            name: Layer name
        """
        super().__init__(name=name)
        
        self.config = config
        self.d_model = config.audio_encoder_dim
        
        # Convolutional layers for audio feature extraction
        self.conv_layers = [
            ConvolutionalLayer(
                filters=64, 
                kernel_size=3, 
                dropout=0.1,
                name="conv1"
            ),
            ConvolutionalLayer(
                filters=128,
                kernel_size=3,
                dropout=0.1, 
                name="conv2"
            ),
            ConvolutionalLayer(
                filters=256,
                kernel_size=3,
                dropout=0.1,
                name="conv3"
            )
        ]
        
        # Projection to model dimension
        self.projection = tf.keras.layers.Dense(
            self.d_model, 
            name="projection"
        )
        
        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(
                self.d_model,
                config.audio_encoder_heads,
                self.d_model * 4,
                dropout=0.1,
                use_gradient_checkpointing=config.enable_gradient_checkpointing,
                name=f"transformer_block_{i}"
            )
            for i in range(config.audio_encoder_layers)
        ]
        
        # Global average pooling for speaker embedding
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.speaker_projection = tf.keras.layers.Dense(
            config.speaker_embedding_dim,
            activation='tanh',
            name="speaker_projection"
        )
    
    def call(
        self,
        inputs: tf.Tensor,
        training: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Forward pass.
        
        Args:
            inputs: Mel spectrogram [batch, time, n_mels]
            training: Training mode flag
            
        Returns:
            Tuple of (contextualized_features, speaker_embedding)
        """
        x = inputs
        
        # Convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x, training=training)
        
        # Project to model dimension
        x = self.projection(x)
        
        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)
        
        # Speaker embedding via global pooling
        speaker_embedding = self.global_pool(x)
        speaker_embedding = self.speaker_projection(speaker_embedding)
        
        return x, speaker_embedding


class MelDecoder(tf.keras.layers.Layer):
    """
    Mel spectrogram decoder.
    
    Generates mel spectrograms from text and audio conditioning.
    """
    
    def __init__(self, config: ModelConfig, name: str = "mel_decoder"):
        """
        Initialize mel decoder.
        
        Args:
            config: Model configuration
            name: Layer name
        """
        super().__init__(name=name)
        
        self.config = config
        self.d_model = config.decoder_dim
        
        # Input projection
        self.input_projection = tf.keras.layers.Dense(
            self.d_model,
            name="input_projection"
        )
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            self.d_model,
            max_length=2000,  # Max mel frames
            name="positional_encoding"
        )
        
        # Transformer decoder blocks
        self.transformer_blocks = [
            TransformerBlock(
                self.d_model,
                config.decoder_heads,
                self.d_model * 4,
                dropout=0.1,
                is_decoder=True,
                use_gradient_checkpointing=config.enable_gradient_checkpointing,
                name=f"transformer_block_{i}"
            )
            for i in range(config.decoder_layers)
        ]
        
        # Output projection
        self.mel_projection = tf.keras.layers.Dense(
            config.n_mels,
            name="mel_projection"
        )
        
        # Stop token prediction
        self.stop_projection = tf.keras.layers.Dense(
            1,
            activation='sigmoid',
            name="stop_projection"
        )
        
        # Speaker conditioning projection (moved from call to enable weight training)
        self.speaker_projection = tf.keras.layers.Dense(
            self.d_model,
            name="speaker_projection"
        )
        
        self.dropout = tf.keras.layers.Dropout(0.1)
    
    def call(
        self,
        decoder_inputs: tf.Tensor,
        encoder_output: tf.Tensor,
        speaker_embedding: Optional[tf.Tensor] = None,
        encoder_mask: Optional[tf.Tensor] = None,
        decoder_mask: Optional[tf.Tensor] = None,
        training: bool = False,
        return_attention_weights: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Forward pass.
        
        Args:
            decoder_inputs: Previous mel frames [batch, mel_len, n_mels]
            encoder_output: Text encoder output [batch, text_len, d_model]
            speaker_embedding: Speaker embedding [batch, speaker_dim]
            encoder_mask: Encoder attention mask
            decoder_mask: Decoder attention mask
            training: Training mode flag
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Tuple of (mel_output, stop_tokens) or (mel_output, stop_tokens, attention_weights)
        """
        # Input projection
        x = self.input_projection(decoder_inputs)
        
        # Add speaker conditioning if provided
        if speaker_embedding is not None:
            # Broadcast speaker embedding to all time steps
            speaker_expanded = tf.expand_dims(speaker_embedding, 1)
            speaker_expanded = tf.tile(
                speaker_expanded, 
                [1, tf.shape(x)[1], 1]
            )
            x = tf.concat([x, speaker_expanded], axis=-1)
            x = self.speaker_projection(x)
        
        # Positional encoding
        x = self.positional_encoding(x)
        x = self.dropout(x, training=training)
        
        # Transform masks for transformer layers
        if encoder_mask is not None:
            encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]
            encoder_mask = tf.cast(encoder_mask, tf.float32)
            encoder_mask = (1.0 - encoder_mask) * -1e9
        
        if decoder_mask is not None:
            decoder_mask = decoder_mask[:, tf.newaxis, :, :]
            decoder_mask = tf.cast(decoder_mask, tf.float32) 
            decoder_mask = (1.0 - decoder_mask) * -1e9
        
        # Transformer decoder blocks
        attention_weights = None
        for i, transformer_block in enumerate(self.transformer_blocks):
            if return_attention_weights and i == len(self.transformer_blocks) - 1:
                # Get attention weights from the last layer
                x, attention_weights = transformer_block(
                    x,
                    encoder_output=encoder_output,
                    self_attention_mask=decoder_mask,
                    cross_attention_mask=encoder_mask,
                    training=training,
                    return_attention_weights=True
                )
            else:
                x = transformer_block(
                    x,
                    encoder_output=encoder_output,
                    self_attention_mask=decoder_mask,
                    cross_attention_mask=encoder_mask,
                    training=training
                )
        
        # Output projections
        mel_output = self.mel_projection(x)
        stop_tokens = self.stop_projection(x)
        
        if return_attention_weights and attention_weights is not None:
            return mel_output, stop_tokens, attention_weights
        else:
            return mel_output, stop_tokens


class XTTS(tf.keras.Model):
    """
    MyXTTS Model - TensorFlow implementation of XTTS.
    
    A transformer-based text-to-speech model with voice cloning capabilities,
    supporting multilingual synthesis and speaker conditioning.
    """
    
    def __init__(self, config: ModelConfig, name: str = "xtts"):
        """
        Initialize XTTS model.
        
        Args:
            config: Model configuration
            name: Model name
        """
        super().__init__(name=name)
        
        self.config = config
        
        # Model components
        self.text_encoder = TextEncoder(config, name="text_encoder")
        
        if config.use_voice_conditioning:
            self.audio_encoder = AudioEncoder(config, name="audio_encoder") 
        
        self.mel_decoder = MelDecoder(config, name="mel_decoder")
    
    def call(
        self,
        text_inputs: tf.Tensor,
        mel_inputs: tf.Tensor,
        audio_conditioning: Optional[tf.Tensor] = None,
        text_lengths: Optional[tf.Tensor] = None,
        mel_lengths: Optional[tf.Tensor] = None,
        training: bool = False
    ) -> Dict[str, tf.Tensor]:
        """
        Forward pass.
        
        Args:
            text_inputs: Text token IDs [batch, text_len]
            mel_inputs: Target mel spectrograms [batch, mel_len, n_mels]
            audio_conditioning: Reference audio for voice cloning [batch, time, n_mels]
            text_lengths: Text sequence lengths [batch]
            mel_lengths: Mel sequence lengths [batch]
            training: Training mode flag
            
        Returns:
            Dictionary with model outputs
        """
        batch_size = tf.shape(text_inputs)[0]
        
        # Create attention masks
        text_mask = None
        if text_lengths is not None:
            text_mask = tf.sequence_mask(
                text_lengths, 
                maxlen=tf.shape(text_inputs)[1],
                dtype=tf.float32
            )
        
        mel_mask = None
        if mel_lengths is not None:
            mel_mask = tf.sequence_mask(
                mel_lengths,
                maxlen=tf.shape(mel_inputs)[1],
                dtype=tf.float32
            )
        
        # Encode text (with optional duration prediction)
        if training:
            text_encoded, duration_pred = self.text_encoder(
                text_inputs,
                attention_mask=text_mask,
                training=training,
                return_durations=True
            )
        else:
            text_encoded = self.text_encoder(
                text_inputs,
                attention_mask=text_mask,
                training=training
            )
            duration_pred = None
        
        # Encode audio for speaker conditioning
        speaker_embedding = None
        if self.config.use_voice_conditioning and audio_conditioning is not None:
            audio_encoded, speaker_embedding = self.audio_encoder(
                audio_conditioning,
                training=training
            )
        
        # Prepare decoder inputs (shifted mel spectrograms)
        # For training: use teacher forcing with previous mel frames
        # For inference: use autoregressive generation
        if training:
            # Teacher forcing: shift mel inputs by one frame
            decoder_inputs = tf.pad(
                mel_inputs[:, :-1, :],
                [[0, 0], [1, 0], [0, 0]],
                mode='CONSTANT'
            )
        else:
            # During inference, start with zeros
            decoder_inputs = tf.zeros_like(mel_inputs)
        
        # Create causal mask for decoder
        mel_len = tf.shape(mel_inputs)[1]
        causal_mask = tf.linalg.band_part(
            tf.ones([mel_len, mel_len]),
            -1, 0
        )  # Lower triangular matrix
        causal_mask = tf.expand_dims(causal_mask, 0)
        causal_mask = tf.tile(causal_mask, [batch_size, 1, 1])
        
        if mel_mask is not None:
            # Combine causal mask with padding mask
            padding_mask = tf.expand_dims(mel_mask, 1)
            causal_mask = causal_mask * padding_mask
        
        # Decode mel spectrograms (with optional attention weights)
        if training:
            mel_decoder_output = self.mel_decoder(
                decoder_inputs,
                text_encoded,
                speaker_embedding=speaker_embedding,
                encoder_mask=text_mask,
                decoder_mask=causal_mask,
                training=training,
                return_attention_weights=True
            )
            if len(mel_decoder_output) == 3:
                mel_output, stop_tokens, attention_weights = mel_decoder_output
            else:
                mel_output, stop_tokens = mel_decoder_output
                attention_weights = None
        else:
            mel_output, stop_tokens = self.mel_decoder(
                decoder_inputs,
                text_encoded,
                speaker_embedding=speaker_embedding,
                encoder_mask=text_mask,
                decoder_mask=causal_mask,
                training=training
            )
            attention_weights = None
        
        outputs = {
            "mel_output": mel_output,
            "stop_tokens": stop_tokens,
            "text_encoded": text_encoded,
        }
        
        if speaker_embedding is not None:
            outputs["speaker_embedding"] = speaker_embedding
            
        # Add duration predictions and attention weights during training
        if training:
            if duration_pred is not None:
                outputs["duration_pred"] = duration_pred
            if attention_weights is not None:
                outputs["attention_weights"] = attention_weights
        
        return outputs
    
    def generate(
        self,
        text_inputs: tf.Tensor,
        audio_conditioning: Optional[tf.Tensor] = None,
        max_length: int = 1000,
        temperature: float = 1.0
    ) -> Dict[str, tf.Tensor]:
        """
        Generate mel spectrograms autoregressively.
        
        Args:
            text_inputs: Text token IDs [batch, text_len]
            audio_conditioning: Reference audio for voice cloning
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            Dictionary with generated outputs
        """
        batch_size = tf.shape(text_inputs)[0]
        
        # Encode text
        text_encoded = self.text_encoder(text_inputs, training=False)
        
        # Encode audio for speaker conditioning
        speaker_embedding = None
        if self.config.use_voice_conditioning and audio_conditioning is not None:
            _, speaker_embedding = self.audio_encoder(
                audio_conditioning, 
                training=False
            )
        
        # Initialize decoder state
        mel_outputs = []
        stop_probs = []
        
        # Start with zero frame
        current_mel = tf.zeros([batch_size, 1, self.config.n_mels])
        
        for step in range(max_length):
            # Decode current step
            mel_output, stop_tokens = self.mel_decoder(
                current_mel,
                text_encoded,
                speaker_embedding=speaker_embedding,
                training=False
            )
            
            # Get last frame prediction
            mel_frame = mel_output[:, -1:, :]  # [batch, 1, n_mels]
            stop_prob = stop_tokens[:, -1:, :]  # [batch, 1, 1]
            
            # Apply temperature sampling if needed
            if temperature != 1.0:
                mel_frame = mel_frame / temperature
            
            mel_outputs.append(mel_frame)
            stop_probs.append(stop_prob)
            
            # Update decoder input for next step
            current_mel = tf.concat([current_mel, mel_frame], axis=1)
            
            # Check for stop condition
            if tf.reduce_all(stop_prob > 0.5):
                break
        
        # Concatenate all frames
        generated_mel = tf.concat(mel_outputs, axis=1)
        generated_stops = tf.concat(stop_probs, axis=1)
        
        return {
            "mel_output": generated_mel,
            "stop_tokens": generated_stops,
            "text_encoded": text_encoded,
            "speaker_embedding": speaker_embedding
        }
    
    def get_config(self) -> Dict:
        """Get model configuration."""
        return {
            "model_config": self.config.__dict__,
            "name": self.name
        }
    
    @classmethod
    def from_config(cls, config: Dict) -> 'XTTS':
        """Create model from configuration."""
        model_config = ModelConfig(**config["model_config"])
        return cls(model_config, name=config.get("name", "xtts"))


def create_xtts_model(config: ModelConfig) -> XTTS:
    """
    Create XTTS model instance.
    
    Args:
        config: Model configuration
        
    Returns:
        XTTS model instance
    """
    return XTTS(config)