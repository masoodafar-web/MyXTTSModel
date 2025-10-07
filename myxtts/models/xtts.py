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
from .vocoder import Vocoder
from .non_autoregressive import DecoderStrategy, NonAutoregressiveDecoder
from .speaker_encoder import PretrainedSpeakerEncoder, ContrastiveSpeakerLoss
from .diffusion_decoder import DiffusionDecoder
from .gst import GlobalStyleToken, ProsodyPredictor
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
        
        # Duration predictor for alignment guidance (conditional)
        self.use_duration_predictor = getattr(config, 'use_duration_predictor', True)
        if self.use_duration_predictor:
            self.duration_predictor = tf.keras.layers.Dense(
                1, 
                activation='relu',
                name="duration_predictor"
            )
        else:
            self.duration_predictor = None
    
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
        scale = tf.math.sqrt(tf.cast(self.d_model, x.dtype))
        x = tf.cast(x, scale.dtype) * scale
        
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
        
        if return_durations and self.use_duration_predictor and self.duration_predictor is not None:
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
        self.use_pretrained = getattr(config, 'use_pretrained_speaker_encoder', False)
        
        if self.use_pretrained:
            # Use pre-trained speaker encoder for enhanced voice conditioning
            self.speaker_encoder = PretrainedSpeakerEncoder(
                config,
                pretrained_path=getattr(config, 'pretrained_speaker_encoder_path', None),
                freeze_weights=getattr(config, 'freeze_speaker_encoder', True),
                embedding_dim=config.speaker_embedding_dim,
                encoder_type=getattr(config, 'speaker_encoder_type', 'ecapa_tdnn'),
                name="pretrained_speaker_encoder"
            )
            # Project to audio encoder dimension if needed
            if config.speaker_embedding_dim != self.d_model:
                self.projection = tf.keras.layers.Dense(
                    self.d_model,
                    name="speaker_projection"
                )
            else:
                self.projection = None
        else:
            # Original convolutional + transformer architecture
            self._build_original_encoder()
    
    def _build_original_encoder(self):
        """Build original convolutional + transformer encoder architecture."""
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
                self.config.audio_encoder_heads,
                self.d_model * 4,
                dropout=0.1,
                use_gradient_checkpointing=self.config.enable_gradient_checkpointing,
                name=f"transformer_block_{i}"
            )
            for i in range(self.config.audio_encoder_layers)
        ]
        
        # Global average pooling for speaker embedding
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.speaker_projection = tf.keras.layers.Dense(
            self.config.speaker_embedding_dim,
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
        if self.use_pretrained:
            # Use pre-trained speaker encoder
            speaker_embedding = self.speaker_encoder(inputs, training=training)
            
            # Project if needed
            if self.projection is not None:
                contextualized_features = self.projection(speaker_embedding)
                # Expand to sequence length for consistency
                contextualized_features = tf.expand_dims(contextualized_features, axis=1)
                contextualized_features = tf.tile(
                    contextualized_features,
                    [1, tf.shape(inputs)[1], 1]
                )
            else:
                contextualized_features = tf.expand_dims(speaker_embedding, axis=1)
                contextualized_features = tf.tile(
                    contextualized_features,
                    [1, tf.shape(inputs)[1], 1]
                )
            
            return contextualized_features, speaker_embedding
        else:
            # Original encoder path
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
        
        # Prosody feature projections (like FastSpeech/FastPitch)
        self.pitch_projection = tf.keras.layers.Dense(
            1,
            name="pitch_projection"
        )
        self.energy_projection = tf.keras.layers.Dense(
            1,
            name="energy_projection"
        )
        
        # Speaker conditioning projection (moved from call to enable weight training)
        self.speaker_projection = tf.keras.layers.Dense(
            self.d_model,
            name="speaker_projection"
        )
        
        # Style conditioning projection for GST
        self.style_projection = tf.keras.layers.Dense(
            self.d_model,
            name="style_projection"
        )
        
        self.dropout = tf.keras.layers.Dropout(0.1)
    
    def call(
        self,
        decoder_inputs: tf.Tensor,
        encoder_output: tf.Tensor,
        speaker_embedding: Optional[tf.Tensor] = None,
        style_embedding: Optional[tf.Tensor] = None,
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
            style_embedding: Style embedding for prosody control [batch, style_dim]
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
        
        # Add style conditioning if provided
        if style_embedding is not None:
            # Broadcast style embedding to all time steps
            style_expanded = tf.expand_dims(style_embedding, 1)
            style_expanded = tf.tile(
                style_expanded,
                [1, tf.shape(x)[1], 1]
            )
            # Add style conditioning to decoder features
            style_projected = self.style_projection(style_expanded)
            x = x + style_projected  # Add rather than concatenate to preserve dimensionality
        
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
        
        # Prosody features (pitch and energy)
        pitch_output = self.pitch_projection(x)
        energy_output = self.energy_projection(x)
        
        if return_attention_weights and attention_weights is not None:
            return mel_output, stop_tokens, pitch_output, energy_output, attention_weights
        else:
            return mel_output, stop_tokens, pitch_output, energy_output


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
        
        # Global Style Tokens for prosody controllability
        if getattr(config, 'use_gst', False):
            self.gst = GlobalStyleToken(config, name="global_style_token")
            self.prosody_predictor = ProsodyPredictor(config, name="prosody_predictor")
        else:
            self.gst = None
            self.prosody_predictor = None
        
        # Decoder strategy (autoregressive, non-autoregressive, or diffusion)
        decoder_strategy = getattr(config, 'decoder_strategy', 'autoregressive')
        if decoder_strategy == 'non_autoregressive':
            self.decoder_strategy = DecoderStrategy(
                config, 
                strategy='non_autoregressive',
                name="decoder_strategy"
            )
        elif decoder_strategy == 'diffusion':
            # Diffusion-based decoder for enhanced quality
            self.diffusion_decoder = DiffusionDecoder(config, name="diffusion_decoder")
            self.decoder_strategy = None
        else:
            # Default to autoregressive for backward compatibility
            self.mel_decoder = MelDecoder(config, name="mel_decoder")
            self.decoder_strategy = None
        
        # HiFi-GAN vocoder
        self.vocoder = Vocoder(config, name="vocoder")
    
    def call(
        self,
        text_inputs: tf.Tensor,
        mel_inputs: tf.Tensor,
        audio_conditioning: Optional[tf.Tensor] = None,
        reference_mel: Optional[tf.Tensor] = None,
        style_weights: Optional[tf.Tensor] = None,
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
            reference_mel: Reference mel for prosody conditioning [batch, time, n_mels]
            style_weights: Direct style weights for GST [batch, num_style_tokens]
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
        
        # Encode text (duration prediction controlled by config)
        if training and self.config.use_duration_predictor:
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
                training=training,
                return_durations=False
            )
            duration_pred = None
        
        # Encode audio for speaker conditioning
        speaker_embedding = None
        if self.config.use_voice_conditioning and audio_conditioning is not None:
            audio_encoded, speaker_embedding = self.audio_encoder(
                audio_conditioning,
                training=training
            )
        
        # Global Style Token (GST) processing for prosody control
        style_embedding = None
        style_attention_weights = None
        prosody_pitch = None
        prosody_energy = None
        prosody_speaking_rate = None
        
        if self.gst is not None:
            # Use available signal for prosody extraction; fall back to current mel inputs when needed
            prosody_reference = reference_mel if reference_mel is not None else (
                audio_conditioning if audio_conditioning is not None else mel_inputs
            )
            
            style_embedding, style_attention_weights = self.gst(
                reference_mel=prosody_reference,
                style_weights=style_weights,
                training=training
            )
            
            # Predict prosody features using style embedding
            if self.prosody_predictor is not None:
                prosody_pitch, prosody_energy, prosody_speaking_rate = self.prosody_predictor(
                    text_encoded,
                    style_embedding,
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
        
        # Decode mel spectrograms using appropriate strategy
        if self.decoder_strategy is not None:
            # Use new decoder strategy interface
            mel_output, stop_or_duration, attention_weights = self.decoder_strategy(
                decoder_inputs if hasattr(self, 'mel_decoder') else None,
                text_encoded,
                durations=None,  # TODO: Add duration targets for training
                speaker_embedding=speaker_embedding,
                encoder_mask=text_mask,
                decoder_mask=causal_mask if hasattr(self, 'mel_decoder') else None,
                max_length=mel_len if not training else None,
                training=training
            )
            
            # For non-autoregressive, stop_or_duration contains duration predictions
            if self.config.decoder_strategy == 'non_autoregressive':
                stop_tokens = None
                duration_pred = stop_or_duration
            else:
                stop_tokens = stop_or_duration
        elif hasattr(self, 'diffusion_decoder'):
            # Use diffusion decoder for high-quality generation
            if training:
                # During training, add noise to mel_inputs and predict the noise
                timesteps = tf.random.uniform([batch_size], 0, self.diffusion_decoder.num_timesteps, dtype=tf.int32)
                noisy_mels, noise = self.diffusion_decoder.forward_diffusion(mel_inputs, timesteps)
                
                # Predict the noise
                predicted_noise = self.diffusion_decoder(
                    noisy_mels,
                    timesteps,
                    text_encoded,
                    speaker_embedding,
                    training=training
                )
                
                # For training, we return both mel and noise for different loss computations
                mel_output = mel_inputs  # Original clean mel for other losses
                # Store noise prediction info for diffusion loss
                diffusion_noise_pred = predicted_noise
                diffusion_noise_target = noise
                stop_tokens = None  # Diffusion doesn't use stop tokens
                pitch_output = energy_output = None  # Not supported in basic diffusion
                attention_weights = None
            else:
                # During inference, generate from noise
                mel_output = self.diffusion_decoder.reverse_diffusion(
                    text_encoded,
                    speaker_embedding,
                    shape=(batch_size, tf.shape(mel_inputs)[1], self.config.n_mels)
                )
                stop_tokens = None
                pitch_output = energy_output = None
                attention_weights = None
                diffusion_noise_pred = diffusion_noise_target = None
        else:
            # Use legacy autoregressive decoder
            if training:
                mel_decoder_output = self.mel_decoder(
                    decoder_inputs,
                    text_encoded,
                    speaker_embedding=speaker_embedding,
                    style_embedding=style_embedding,
                    encoder_mask=text_mask,
                    decoder_mask=causal_mask,
                    training=training,
                    return_attention_weights=True
                )
                if len(mel_decoder_output) == 5:
                    mel_output, stop_tokens, pitch_output, energy_output, attention_weights = mel_decoder_output
                elif len(mel_decoder_output) == 4:
                    mel_output, stop_tokens, pitch_output, energy_output = mel_decoder_output
                    attention_weights = None
                elif len(mel_decoder_output) == 3:
                    # Backward compatibility - assume no prosody features
                    mel_output, stop_tokens, attention_weights = mel_decoder_output
                    pitch_output = energy_output = None
                else:
                    mel_output, stop_tokens = mel_decoder_output
                    pitch_output = energy_output = attention_weights = None
            else:
                mel_decoder_output = self.mel_decoder(
                    decoder_inputs,
                    text_encoded,
                    speaker_embedding=speaker_embedding,
                    style_embedding=style_embedding,
                    encoder_mask=text_mask,
                    decoder_mask=causal_mask,
                    training=training
                )
                if len(mel_decoder_output) == 4:
                    mel_output, stop_tokens, pitch_output, energy_output = mel_decoder_output
                elif len(mel_decoder_output) == 2:
                    # Backward compatibility - assume no prosody features
                    mel_output, stop_tokens = mel_decoder_output
                    pitch_output = energy_output = None
                else:
                    mel_output, stop_tokens = mel_decoder_output
                    pitch_output = energy_output = None
                attention_weights = None
        
        outputs = {
            "mel_output": mel_output,
            "stop_tokens": stop_tokens,
            "text_encoded": text_encoded,
        }
        
        # Add prosody features to outputs (during training, always include to ensure gradient participation)
        if training and pitch_output is not None:
            outputs["pitch_output"] = pitch_output
        if training and energy_output is not None:
            outputs["energy_output"] = energy_output
        
        # Add GST-related outputs (during training, always include to ensure gradient participation)
        if style_embedding is not None:
            outputs["style_embedding"] = style_embedding
        if style_attention_weights is not None:
            outputs["style_attention_weights"] = style_attention_weights
        if training and prosody_pitch is not None:
            outputs["prosody_pitch"] = prosody_pitch
        if training and prosody_energy is not None:
            outputs["prosody_energy"] = prosody_energy
        if training and prosody_speaking_rate is not None:
            outputs["prosody_speaking_rate"] = prosody_speaking_rate
        
        # Add diffusion loss outputs
        if training and 'diffusion_noise_pred' in locals():
            outputs["diffusion_noise_pred"] = diffusion_noise_pred
            outputs["diffusion_noise_target"] = diffusion_noise_target
        
        if speaker_embedding is not None:
            outputs["speaker_embedding"] = speaker_embedding
            
        # Add duration predictions during training if enabled
        if training and self.config.use_duration_predictor and duration_pred is not None:
            outputs["duration_pred"] = duration_pred
        if training and attention_weights is not None:
            outputs["attention_weights"] = attention_weights
        
        return outputs
    
    def generate(
        self,
        text_inputs: tf.Tensor,
        audio_conditioning: Optional[tf.Tensor] = None,
        reference_mel: Optional[tf.Tensor] = None,
        style_weights: Optional[tf.Tensor] = None,
        max_length: int = 1000,
        temperature: float = 1.0,
        generate_audio: bool = False
    ) -> Dict[str, tf.Tensor]:
        """
        Generate mel spectrograms and optionally audio.
        
        Args:
            text_inputs: Text token IDs [batch, text_len]
            audio_conditioning: Reference audio for voice cloning
            reference_mel: Reference mel for prosody conditioning [batch, time, n_mels]
            style_weights: Direct style weights for GST [batch, num_style_tokens]
            max_length: Maximum generation length
            temperature: Sampling temperature
            generate_audio: Whether to generate audio using neural vocoder
            
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
        
        # Global Style Token (GST) processing for prosody control
        style_embedding = None
        if self.gst is not None:
            # Use reference_mel for prosody extraction, fallback to audio_conditioning
            prosody_reference = reference_mel if reference_mel is not None else audio_conditioning

            if prosody_reference is None and style_weights is None:
                neutral_weights = tf.ones(
                    [batch_size, self.config.gst_num_style_tokens],
                    dtype=tf.float32
                ) / float(self.config.gst_num_style_tokens)
                style_embedding, _ = self.gst(
                    reference_mel=None,
                    style_weights=neutral_weights,
                    training=False
                )
            else:
                style_embedding, _ = self.gst(
                    reference_mel=prosody_reference,
                    style_weights=style_weights,
                    training=False
                )
        
        # Generate mel spectrograms using appropriate strategy
        if self.decoder_strategy is not None and getattr(self.config, 'decoder_strategy', '') == 'non_autoregressive':
            # Non-autoregressive generation
            mel_output, duration_pred = self.decoder_strategy.decoder(
                text_encoded,
                speaker_embedding=speaker_embedding,
                max_length=max_length,
                training=False
            )
            generated_mel = mel_output
            generated_stops = None
            
        else:
            # Autoregressive generation
            mel_outputs = []
            stop_probs = []
            
            # Start with zero frame
            current_mel = tf.zeros([batch_size, 1, self.config.n_mels])
            
            decoder = self.mel_decoder if hasattr(self, 'mel_decoder') else self.decoder_strategy.decoder
            
            # Calculate minimum frames based on text length
            # Assume roughly 10-15 mel frames per character for reasonable speech
            text_len = tf.shape(text_inputs)[1]
            min_frames_tensor = tf.maximum(20, text_len * 10)  # At least 20 frames, or 10x text length
            # Convert to Python int for range() - safe in eager mode
            try:
                min_frames = int(min_frames_tensor.numpy())
            except (AttributeError, RuntimeError):
                # Fallback for graph mode - use a reasonable default
                min_frames = 50
            
            for step in range(max_length):
                # Decode current step
                if hasattr(self, 'mel_decoder'):
                    mel_output, stop_tokens, pitch_output, energy_output = self.mel_decoder(
                        current_mel,
                        text_encoded,
                        speaker_embedding=speaker_embedding,
                        training=False
                    )
                else:
                    mel_output, stop_tokens, _ = self.decoder_strategy(
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
                
                # Check for stop condition with improved logic
                stop_prob_value = float(tf.reduce_mean(stop_prob))
                
                # More robust stopping criteria:
                # 1. Never stop before minimum frames (based on text length)
                # 2. Require very high confidence (0.95) to stop early
                # 3. Use sliding window average for more stable stop detection
                if step < min_frames:
                    # Don't check stop condition before minimum length
                    continue
                elif step >= min_frames and stop_prob_value > 0.95:
                    # High confidence stop after minimum length
                    break
                elif step > max_length * 0.9:
                    # Safety break if we're near max length
                    break
            
            # Concatenate all frames
            generated_mel = tf.concat(mel_outputs, axis=1)
            generated_stops = tf.concat(stop_probs, axis=1)
        
        results = {
            "mel_output": generated_mel,
            "text_encoded": text_encoded,
            "speaker_embedding": speaker_embedding
        }
        
        if generated_stops is not None:
            results["stop_tokens"] = generated_stops
        
        # Generate audio using HiFi-GAN vocoder if requested
        if generate_audio:
            generated_audio = self.vocoder(generated_mel, training=False)
            results["audio_output"] = generated_audio
        
        return results
    
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
