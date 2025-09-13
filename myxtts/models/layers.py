"""
Neural network layers for MyXTTS model.

This module implements transformer-based layers including multi-head attention,
positional encoding, and feed-forward networks used in the XTTS architecture.
"""

import tensorflow as tf
import numpy as np
from typing import Optional, Tuple

# Import device utilities
from ..utils.commons import create_dropout_layer, get_device_context


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
        Compute scaled dot-product attention.
        
        Args:
            q: Query tensor [batch, seq_len, d_k]
            k: Key tensor [batch, seq_len, d_k]
            v: Value tensor [batch, seq_len, d_k]
            mask: Attention mask [batch, seq_len, seq_len]
            
        Returns:
            Tuple of (attention_output, attention_weights)
        """
        # Compute attention scores
        scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(
            tf.cast(self.d_k, tf.float32)
        )
        
        # Apply mask if provided
        if mask is not None:
            scores += (mask * -1e9)
        
        # Softmax
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
        training: bool = False
    ) -> tf.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Input tensor [batch, seq_len, d_model]
            key_value: Key-value tensor for cross-attention
            mask: Attention mask
            training: Training mode flag
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        # Linear projections
        q = self.wq(inputs)  # [batch, seq_len, d_model]
        
        if key_value is not None:
            # Cross-attention
            k = self.wk(key_value)
            v = self.wv(key_value)
        else:
            # Self-attention
            k = self.wk(inputs)
            v = self.wv(inputs)
        
        # Reshape for multi-head attention
        q = tf.reshape(q, [batch_size, seq_len, self.num_heads, self.d_k])
        q = tf.transpose(q, [0, 2, 1, 3])  # [batch, heads, seq_len, d_k]
        
        k_seq_len = tf.shape(k)[1]
        k = tf.reshape(k, [batch_size, k_seq_len, self.num_heads, self.d_k])
        k = tf.transpose(k, [0, 2, 1, 3])  # [batch, heads, seq_len, d_k]
        
        v = tf.reshape(v, [batch_size, k_seq_len, self.num_heads, self.d_k])
        v = tf.transpose(v, [0, 2, 1, 3])  # [batch, heads, seq_len, d_k]
        
        # Apply attention
        attention_output, _ = self.scaled_dot_product_attention(q, k, v, mask)
        
        # Concatenate heads
        attention_output = tf.transpose(attention_output, [0, 2, 1, 3])
        attention_output = tf.reshape(
            attention_output, [batch_size, seq_len, self.d_model]
        )
        
        # Final linear projection
        output = self.wo(attention_output)
        
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
    """Transformer encoder/decoder block."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        is_decoder: bool = False,
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
            name: Layer name
        """
        super().__init__(name=name)
        
        self.d_model = d_model
        self.is_decoder = is_decoder
        
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
    
    def call(
        self,
        inputs: tf.Tensor,
        encoder_output: Optional[tf.Tensor] = None,
        self_attention_mask: Optional[tf.Tensor] = None,
        cross_attention_mask: Optional[tf.Tensor] = None,
        training: bool = False
    ) -> tf.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Input tensor [batch, seq_len, d_model]
            encoder_output: Encoder output for cross-attention
            self_attention_mask: Self-attention mask
            cross_attention_mask: Cross-attention mask
            training: Training mode flag
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # Self-attention
        attn_output = self.self_attention(
            inputs, mask=self_attention_mask, training=training
        )
        attn_output = self.dropout(attn_output, training=training)
        x = self.norm1(inputs + attn_output)
        
        # Cross-attention (decoder only)
        if self.is_decoder and encoder_output is not None:
            attn_output = self.cross_attention(
                x, key_value=encoder_output, 
                mask=cross_attention_mask, training=training
            )
            attn_output = self.dropout(attn_output, training=training)
            x = self.norm2(x + attn_output)
        
        # Feed-forward
        ff_output = self.feed_forward(x, training=training)
        ff_output = self.dropout(ff_output, training=training)
        
        # Final normalization: use the third norm layer for decoder blocks,
        # and for encoder blocks this is the second norm (named "norm2").
        # We always apply self.norm3, which is constructed as
        # name="norm3" if decoder else name="norm2".
        output = self.norm3(x + ff_output)
        
        return output


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
