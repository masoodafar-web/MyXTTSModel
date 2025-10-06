"""
Neural vocoder implementation for MyXTTS.

This module provides HiFi-GAN vocoder for high-quality mel spectrogram 
to audio conversion.
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Optional, Tuple
from ..config.config import ModelConfig


class Vocoder(tf.keras.layers.Layer):
    """
    HiFi-GAN vocoder for high-quality mel-to-audio synthesis.
    
    Based on "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis"
    https://arxiv.org/abs/2010.05909
    """
    
    def __init__(
        self,
        config: ModelConfig,
        upsample_rates: List[int] = None,
        upsample_kernel_sizes: List[int] = None,
        resblock_kernel_sizes: List[int] = None,
        resblock_dilation_sizes: List[List[int]] = None,
        initial_channel: int = None,
        name: str = "vocoder"
    ):
        """
        Initialize HiFi-GAN vocoder.
        
        Args:
            config: Model configuration
            upsample_rates: Upsampling rates for each upsampling layer
            upsample_kernel_sizes: Kernel sizes for upsampling layers
            resblock_kernel_sizes: Kernel sizes for residual blocks
            resblock_dilation_sizes: Dilation sizes for residual blocks
            initial_channel: Initial number of channels
            name: Layer name
        """
        super().__init__(name=name)
        
        self.config = config
        
        # Use config values if not provided
        self.upsample_rates = upsample_rates or getattr(config, 'vocoder_upsample_rates', [8, 8, 2, 2])
        self.upsample_kernel_sizes = upsample_kernel_sizes or getattr(config, 'vocoder_upsample_kernel_sizes', [16, 16, 4, 4])
        self.resblock_kernel_sizes = resblock_kernel_sizes or getattr(config, 'vocoder_resblock_kernel_sizes', [3, 7, 11])
        self.resblock_dilation_sizes = resblock_dilation_sizes or getattr(config, 'vocoder_resblock_dilation_sizes', [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
        self.initial_channel = initial_channel or getattr(config, 'vocoder_initial_channel', 512)
        
        self._weights_initialized = False
        
        # Pre-convolution
        self.pre_conv = tf.keras.layers.Conv1D(
            self.initial_channel,
            kernel_size=7,
            strides=1,
            padding='same',
            name='pre_conv'
        )
        
        # Upsampling layers
        self.upsampling_layers = []
        self.resblocks = []
        
        channels = self.initial_channel
        for i, (rate, kernel_size) in enumerate(zip(self.upsample_rates, self.upsample_kernel_sizes)):
            # Upsampling layer
            channels = channels // 2
            upsample_layer = tf.keras.layers.Conv1DTranspose(
                channels,
                kernel_size=kernel_size,
                strides=rate,
                padding='same',
                name=f'upsample_{i}'
            )
            self.upsampling_layers.append(upsample_layer)
            
            # Residual blocks for this upsampling layer
            layer_resblocks = []
            for j, (res_kernel_size, dilations) in enumerate(zip(self.resblock_kernel_sizes, self.resblock_dilation_sizes)):
                resblock = ResidualBlock(
                    channels,
                    res_kernel_size,
                    dilations,
                    name=f'resblock_{i}_{j}'
                )
                layer_resblocks.append(resblock)
            self.resblocks.append(layer_resblocks)
        
        # Post-convolution
        self.post_conv = tf.keras.layers.Conv1D(
            1,
            kernel_size=7,
            strides=1,
            padding='same',
            activation='tanh',
            name='post_conv'
        )
        
        # LeakyReLU activation
        self.activation = tf.keras.layers.LeakyReLU(0.1)
    
    def mark_weights_loaded(self):
        """Mark that vocoder weights have been loaded from checkpoint."""
        self._weights_initialized = True
    
    def check_weights_initialized(self) -> bool:
        """Check if vocoder weights are properly initialized."""
        return self._weights_initialized
    
    def call(self, mel: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass.
        
        Args:
            mel: Mel spectrogram [batch, time, n_mels]
            training: Training mode flag
            
        Returns:
            Audio waveform [batch, time * hop_length, 1]
        """
        # Warn if vocoder weights may not be initialized
        if not training and not self._weights_initialized:
            import logging
            logger = logging.getLogger("MyXTTS.Vocoder")
            logger.warning(
                "⚠️ HiFi-GAN vocoder weights may not be properly initialized! "
                "This will produce noise instead of speech. "
                "Make sure to load a trained checkpoint with vocoder weights."
            )
        
        # Pre-convolution
        x = self.pre_conv(mel)
        x = self.activation(x)
        
        # Upsampling and residual blocks
        for i, (upsample_layer, resblocks) in enumerate(zip(self.upsampling_layers, self.resblocks)):
            # Upsample
            x = upsample_layer(x)
            x = self.activation(x)
            
            # Apply residual blocks
            residual_outputs = []
            for resblock in resblocks:
                residual_outputs.append(resblock(x, training=training))
            
            # Sum residual outputs
            x = tf.add_n(residual_outputs) / len(residual_outputs)
        
        # Post-convolution
        audio = self.post_conv(x)
        
        # Validate output is not all zeros or NaNs
        if not training:
            # Check for problematic output
            audio_mean = tf.reduce_mean(tf.abs(audio))
            if tf.math.is_nan(audio_mean) or audio_mean < 1e-8:
                import logging
                logger = logging.getLogger("MyXTTS.Vocoder")
                logger.error(
                    "❌ Vocoder produced invalid output (NaN or near-zero). "
                    "This indicates the vocoder is not properly trained."
                )
        
        return audio


class ResidualBlock(tf.keras.layers.Layer):
    """Residual block for HiFi-GAN generator."""
    
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilations: List[int] = [1, 3, 5],
        name: str = "residual_block"
    ):
        """
        Initialize residual block.
        
        Args:
            channels: Number of channels
            kernel_size: Convolution kernel size
            dilations: Dilation rates for dilated convolutions
            name: Layer name
        """
        super().__init__(name=name)
        
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilations = dilations
        
        # Convolution layers
        self.conv_layers = []
        for i, dilation in enumerate(dilations):
            conv_layer = tf.keras.layers.Conv1D(
                channels,
                kernel_size=kernel_size,
                dilation_rate=dilation,
                padding='same',
                name=f'conv_{i}'
            )
            self.conv_layers.append(conv_layer)
        
        # LeakyReLU activation
        self.activation = tf.keras.layers.LeakyReLU(0.1)
    
    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, time, channels]
            training: Training mode flag
            
        Returns:
            Output tensor [batch, time, channels]
        """
        residual = x
        
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            x = self.activation(x)
        
        return x + residual


# Backward compatibility aliases
HiFiGANGenerator = Vocoder
VocoderInterface = Vocoder



class VocoderLoss(tf.keras.layers.Layer):
    """
    Multi-scale discriminator loss for vocoder training.
    
    Implements discriminator losses for high-quality audio generation.
    """
    
    def __init__(self, name: str = "vocoder_loss"):
        super().__init__(name=name)
        
        # Feature matching loss weight
        self.feature_loss_weight = 2.0
        
        # Mel spectrogram loss weight
        self.mel_loss_weight = 45.0
    
    def call(
        self,
        real_audio: tf.Tensor,
        generated_audio: tf.Tensor,
        real_mel: tf.Tensor,
        generated_mel: tf.Tensor,
        training: bool = False
    ) -> Dict[str, tf.Tensor]:
        """
        Compute vocoder training losses.
        
        Args:
            real_audio: Real audio [batch, time, 1]
            generated_audio: Generated audio [batch, time, 1]
            real_mel: Real mel spectrogram [batch, time, n_mels]
            generated_mel: Generated mel spectrogram [batch, time, n_mels]
            training: Training mode flag
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # L1 loss between mel spectrograms
        mel_loss = tf.reduce_mean(tf.abs(real_mel - generated_mel))
        losses['mel_loss'] = mel_loss
        
        # L1 loss between audio (for early training stages)
        audio_loss = tf.reduce_mean(tf.abs(real_audio - generated_audio))
        losses['audio_loss'] = audio_loss
        
        # Total generator loss
        total_loss = self.mel_loss_weight * mel_loss + audio_loss
        losses['total_loss'] = total_loss
        
        return losses