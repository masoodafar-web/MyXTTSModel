"""
Neural network layers for MyXTTS model.

This module implements transformer-based layers including multi-head attention,
positional encoding, and feed-forward networks used in the XTTS architecture.
"""

import tensorflow as tf
import numpy as np
from typing import Optional, Tuple
import warnings
import os
import importlib.util

# Import device utilities with fallback
try:
    from ..utils.commons import create_dropout_layer, get_device_context
except ImportError:
    # Fallback: load commons directly to avoid dependency issues
    try:
        commons_path = os.path.join(os.path.dirname(__file__), '..', 'utils', 'commons.py')
        spec = importlib.util.spec_from_file_location("commons", commons_path)
        commons = importlib.util.module_from_spec(spec)
        # Add required imports to commons namespace
        commons.tf = tf
        spec.loader.exec_module(commons)
        create_dropout_layer = commons.create_dropout_layer
        get_device_context = commons.get_device_context
        # Apply GPU configuration
        commons.configure_gpus()
    except Exception as e:
        warnings.warn(f"Could not load device utilities: {e}")
        # Fallback implementations
        def get_device_context():
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                return tf.device('/GPU:0')
            else:
                return tf.device('/CPU:0')
        
        def create_dropout_layer(rate: float, seed: Optional[int] = None, name: str = "dropout"):
            with get_device_context():
                return tf.keras.layers.Dropout(rate, seed=seed, name=name)


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-head attention layer compatible with XTTS architecture."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        name: str = "multi_head_attention"
    ):
        """
        Initialize multi-head attention.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            name: Layer name
        """
        super().__init__(name=name)
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.wq = tf.keras.layers.Dense(d_model, name="query_projection")
        self.wk = tf.keras.layers.Dense(d_model, name="key_projection") 
        self.wv = tf.keras.layers.Dense(d_model, name="value_projection")
        self.wo = tf.keras.layers.Dense(d_model, name="output_projection")
        
        # Use device-aware dropout creation to avoid placement conflicts
        self.dropout = create_dropout_layer(dropout, name="dropout")
    
    def scaled_dot_product_attention(
        self,
        q: tf.Tensor,
        k: tf.Tensor,
        v: tf.Tensor,
        mask: Optional[tf.Tensor] = None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute memory-efficient scaled dot-product attention.
        
        Args:
            q: Query tensor [batch, heads, seq_len, d_k]
            k: Key tensor [batch, heads, seq_len, d_k]
            v: Value tensor [batch, heads, seq_len, d_k]
            mask: Attention mask [batch, heads, seq_len, seq_len]
            
        Returns:
            Tuple of (attention_output, attention_weights)
        """
        # Get sequence lengths
        q_seq_len = tf.shape(q)[2]
        k_seq_len = tf.shape(k)[2]
        
        # Memory optimization: limit maximum sequence length for attention
        # Use a generous cap so inference up to ~2k frames is unaffected.
        max_seq_len = 2048
        
        if q_seq_len > max_seq_len or k_seq_len > max_seq_len:
            # Truncate sequences to prevent memory explosion
            q = q[:, :, :max_seq_len, :]
            k = k[:, :, :max_seq_len, :]
            v = v[:, :, :max_seq_len, :]
            if mask is not None:
                mask = mask[:, :, :max_seq_len, :max_seq_len]
        
        # Use tf.nn.scaled_dot_product_attention if available (TF 2.11+)
        try:
            # Modern TensorFlow has built-in memory-efficient attention
            output = tf.nn.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=mask,
                dropout_rate=0.1 if self.dropout.rate > 0 else 0.0
            )
            # Return dummy weights since we can't get them from the built-in function
            attention_weights = tf.zeros([tf.shape(q)[0], tf.shape(q)[1], 
                                        tf.shape(q)[2], tf.shape(k)[2]])
            return output, attention_weights
        except AttributeError:
            # Fallback to manual implementation for older TensorFlow versions
            pass
        
        # Compute attention scores with memory limits
        scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(
            tf.cast(self.d_k, tf.float32)
        )
        
        # Apply mask if provided
        if mask is not None:
            scores += (mask * -1e9)
        
        # Use memory-efficient softmax for large tensors
        attention_weights = tf.nn.softmax(scores, axis=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = tf.matmul(attention_weights, v)
        
        return output, attention_weights
    
    def call(
        self,
        inputs: tf.Tensor,
        key_value: Optional[tf.Tensor] = None,
        mask: Optional[tf.Tensor] = None,
        training: bool = False,
        return_attention_weights: bool = False
    ) -> tf.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Input tensor [batch, seq_len, d_model]
            key_value: Key-value tensor for cross-attention
            mask: Attention mask
            training: Training mode flag
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Output tensor [batch, seq_len, d_model] or tuple (output, attention_weights)
        """
        batch_size = tf.shape(inputs)[0]

        # Linear projections
        q = self.wq(inputs)  # [batch, seq_len, d_model]
        seq_len_q = tf.shape(q)[1]

        if key_value is not None:
            # Cross-attention
            k = self.wk(key_value)
            v = self.wv(key_value)
        else:
            # Self-attention
            k = self.wk(inputs)
            v = self.wv(inputs)
        
        # Reshape for multi-head attention
        q = tf.reshape(q, [batch_size, seq_len_q, self.num_heads, self.d_k])
        q = tf.transpose(q, [0, 2, 1, 3])  # [batch, heads, seq_len, d_k]

        k_seq_len = tf.shape(k)[1]
        k = tf.reshape(k, [batch_size, k_seq_len, self.num_heads, self.d_k])
        k = tf.transpose(k, [0, 2, 1, 3])  # [batch, heads, seq_len, d_k]

        v = tf.reshape(v, [batch_size, k_seq_len, self.num_heads, self.d_k])
        v = tf.transpose(v, [0, 2, 1, 3])  # [batch, heads, seq_len, d_k]
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        
        # Concatenate heads
        attention_output = tf.transpose(attention_output, [0, 2, 1, 3])
        seq_len_out = tf.shape(attention_output)[1]
        attention_output = tf.reshape(
            attention_output, [batch_size, seq_len_out, self.d_model]
        )
        
        # Final linear projection
        output = self.wo(attention_output)
        
        if return_attention_weights:
            return output, attention_weights
        else:
            return output


class PositionalEncoding(tf.keras.layers.Layer):
    """Sinusoidal positional encoding layer."""
    
    def __init__(
        self,
        d_model: int,
        max_length: int = 10000,
        name: str = "positional_encoding"
    ):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_length: Maximum sequence length
            name: Layer name
        """
        super().__init__(name=name)
        
        self.d_model = d_model
        self.max_length = max_length
        
        # Create positional encoding matrix
        pe = np.zeros((max_length, d_model), dtype=np.float32)
        position = np.arange(0, max_length, dtype=np.float32)[:, np.newaxis]
        
        div_term = np.exp(
            np.arange(0, d_model, 2, dtype=np.float32) * 
            -(np.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        # Create positional encoding with proper device placement
        with get_device_context():
            self.pe = tf.Variable(
                pe[np.newaxis, :, :],  # [1, max_length, d_model]
                trainable=False,
                name="positional_encoding"
            )
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Add positional encoding to inputs.
        
        Args:
            inputs: Input tensor [batch, seq_len, d_model]
            
        Returns:
            Output with positional encoding added
        """
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pe[:, :seq_len, :]


class FeedForward(tf.keras.layers.Layer):
    """Feed-forward network layer."""
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
        name: str = "feed_forward"
    ):
        """
        Initialize feed-forward network.
        
        Args:
            d_model: Model dimension
            d_ff: Hidden dimension
            dropout: Dropout rate
            activation: Activation function
            name: Layer name
        """
        super().__init__(name=name)
        
        self.linear1 = tf.keras.layers.Dense(
            d_ff, activation=activation, name="linear1"
        )
        self.linear2 = tf.keras.layers.Dense(d_model, name="linear2")
        # Use device-aware dropout creation to avoid placement conflicts
        self.dropout = create_dropout_layer(dropout, name="dropout")
    
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Input tensor [batch, seq_len, d_model]
            training: Training mode flag
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        x = self.linear1(inputs)
        x = self.dropout(x, training=training)
        x = self.linear2(x)
        return x


class TransformerBlock(tf.keras.layers.Layer):
    """Transformer encoder/decoder block with memory optimization."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        is_decoder: bool = False,
        use_gradient_checkpointing: bool = False,
        name: str = "transformer_block"
    ):
        """
        Initialize transformer block.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout rate
            is_decoder: Whether this is a decoder block
            use_gradient_checkpointing: Enable gradient checkpointing for memory savings
            name: Layer name
        """
        super().__init__(name=name)
        
        self.d_model = d_model
        self.is_decoder = is_decoder
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Self-attention
        self.self_attention = MultiHeadAttention(
            d_model, num_heads, dropout, name="self_attention"
        )
        self.norm1 = tf.keras.layers.LayerNormalization(name="norm1")
        
        # Cross-attention (decoder only)
        if is_decoder:
            self.cross_attention = MultiHeadAttention(
                d_model, num_heads, dropout, name="cross_attention"
            )
            self.norm2 = tf.keras.layers.LayerNormalization(name="norm2")
        
        # Feed-forward
        self.feed_forward = FeedForward(
            d_model, d_ff, dropout, name="feed_forward"
        )
        self.norm3 = tf.keras.layers.LayerNormalization(
            name="norm3" if is_decoder else "norm2"
        )
        
        # Use device-aware dropout creation to avoid placement conflicts
        self.dropout = create_dropout_layer(dropout, name="dropout")
    
    def _self_attention_block(self, inputs, mask, training):
        """Self-attention block that can be checkpointed."""
        attn_output = self.self_attention(
            inputs, mask=mask, training=training
        )
        attn_output = self.dropout(attn_output, training=training)
        return self.norm1(inputs + attn_output)
    
    def _cross_attention_block(self, inputs, encoder_output, mask, training):
        """Cross-attention block that can be checkpointed."""
        attn_output = self.cross_attention(
            inputs, key_value=encoder_output, 
            mask=mask, training=training
        )
        attn_output = self.dropout(attn_output, training=training)
        return self.norm2(inputs + attn_output)
    
    def _feed_forward_block(self, inputs, training):
        """Feed-forward block that can be checkpointed."""
        ff_output = self.feed_forward(inputs, training=training)
        ff_output = self.dropout(ff_output, training=training)
        return self.norm3(inputs + ff_output)
    
    def call(
        self,
        inputs: tf.Tensor,
        encoder_output: Optional[tf.Tensor] = None,
        self_attention_mask: Optional[tf.Tensor] = None,
        cross_attention_mask: Optional[tf.Tensor] = None,
        training: bool = False,
        return_attention_weights: bool = False
    ) -> tf.Tensor:
        """
        Forward pass with optional gradient checkpointing.
        
        Args:
            inputs: Input tensor [batch, seq_len, d_model]
            encoder_output: Encoder output for cross-attention
            self_attention_mask: Self-attention mask
            cross_attention_mask: Cross-attention mask
            training: Training mode flag
            return_attention_weights: Whether to return cross-attention weights
            
        Returns:
            Output tensor [batch, seq_len, d_model] or tuple (output, attention_weights)
        """
        attention_weights = None
        
        # Self-attention with optional gradient checkpointing
        if self.use_gradient_checkpointing and training:
            # Correct usage: wrap the block function, then call with kwargs
            x = tf.recompute_grad(self._self_attention_block)(
                inputs, self_attention_mask, training=training
            )
        else:
            x = self._self_attention_block(inputs, self_attention_mask, training)
        
        # Cross-attention (decoder only)
        if self.is_decoder and encoder_output is not None:
            if return_attention_weights:
                # For cross-attention, we need to capture attention weights
                attn_output, attention_weights = self.cross_attention(
                    x, key_value=encoder_output, 
                    mask=cross_attention_mask, training=training,
                    return_attention_weights=True
                )
                attn_output = self.dropout(attn_output, training=training)
                x = self.norm2(x + attn_output)
            else:
                if self.use_gradient_checkpointing and training:
                    x = tf.recompute_grad(self._cross_attention_block)(
                        x, encoder_output, cross_attention_mask, training=training
                    )
                else:
                    x = self._cross_attention_block(x, encoder_output, cross_attention_mask, training)
        
        # Feed-forward with optional gradient checkpointing
        if self.use_gradient_checkpointing and training:
            x = tf.recompute_grad(self._feed_forward_block)(
                x, training=training
            )
        else:
            x = self._feed_forward_block(x, training)
        
        if return_attention_weights and attention_weights is not None:
            return x, attention_weights
        else:
            return x


class ConvolutionalLayer(tf.keras.layers.Layer):
    """1D Convolutional layer with normalization."""
    
    def __init__(
        self,
        filters: int,
        kernel_size: int,
        strides: int = 1,
        padding: str = "same",
        activation: str = "relu",
        dropout: float = 0.0,
        name: str = "conv1d"
    ):
        """
        Initialize convolutional layer.
        
        Args:
            filters: Number of filters
            kernel_size: Kernel size
            strides: Stride size
            padding: Padding type
            activation: Activation function
            dropout: Dropout rate
            name: Layer name
        """
        super().__init__(name=name)
        
        self.conv = tf.keras.layers.Conv1D(
            filters, kernel_size, strides, padding, name="conv"
        )
        self.batch_norm = tf.keras.layers.BatchNormalization(name="batch_norm")
        self.activation = tf.keras.layers.Activation(activation)
        # Use device-aware dropout creation to avoid placement conflicts
        self.dropout = create_dropout_layer(dropout, name="dropout")
    
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass."""
        x = self.conv(inputs)
        x = self.batch_norm(x, training=training)
        x = self.activation(x)
        x = self.dropout(x, training=training)
        return x
