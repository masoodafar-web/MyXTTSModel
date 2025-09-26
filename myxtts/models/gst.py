"""
Global Style Tokens (GST) for MyXTTS Model.

This module implements Global Style Tokens as described in the paper:
"Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis"

The GST allows controllable prosody synthesis by learning a bank of style embeddings
that can be selectively activated to control emotion, speaking rate, and other prosodic features.

Similar to Coqui's approach with separate vector conditioning for expressive TTS.
"""

import tensorflow as tf
from typing import Optional, Tuple
from ..config.config import ModelConfig


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-head attention for style token selection."""
    
    def __init__(self, d_model: int, num_heads: int, name: str = "multi_head_attention"):
        super().__init__(name=name)
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        
        self.wq = tf.keras.layers.Dense(d_model, name="query_projection")
        self.wk = tf.keras.layers.Dense(d_model, name="key_projection") 
        self.wv = tf.keras.layers.Dense(d_model, name="value_projection")
        self.dense = tf.keras.layers.Dense(d_model, name="output_projection")
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, query, key, value, mask=None, training=False):
        batch_size = tf.shape(query)[0]
        
        # Linear projections
        q = self.wq(query)  # (batch_size, seq_len_q, d_model)
        k = self.wk(key)    # (batch_size, seq_len_k, d_model)
        v = self.wv(value)  # (batch_size, seq_len_v, d_model)
        
        # Split into multiple heads
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        
        # Scaled dot-product attention in float32 for stability
        q32 = tf.cast(q, tf.float32)
        k32 = tf.cast(k, tf.float32)
        v32 = tf.cast(v, tf.float32)
        attn_scores32 = tf.matmul(q32, k32, transpose_b=True)  # (batch, heads, q_len, k_len)
        scale32 = tf.math.sqrt(tf.cast(self.depth, tf.float32))
        attn_weights32 = attn_scores32 / scale32

        if mask is not None:
            mask32 = tf.cast(mask, tf.float32)
            attn_weights32 += (mask32 * tf.cast(-1e9, tf.float32))

        attn_weights32 = tf.nn.softmax(attn_weights32, axis=-1)
        
        # Apply attention to values
        attention_output32 = tf.matmul(attn_weights32, v32)  # (batch_size, num_heads, seq_len_q, depth)
        
        # Concatenate heads
        attention_output = tf.transpose(attention_output32, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, (batch_size, -1, self.d_model))
        attention_output = tf.cast(attention_output, query.dtype)

        # Cast attention weights to match input dtype for consistency
        attention_weights = tf.cast(attn_weights32, query.dtype)
        
        # Final linear projection
        output = self.dense(attention_output)
        
        return output, attention_weights


class ReferenceEncoder(tf.keras.layers.Layer):
    """Reference encoder to extract prosody features from reference audio."""
    
    def __init__(self, config: ModelConfig, name: str = "reference_encoder"):
        super().__init__(name=name)
        
        self.config = config
        self.conv_layers = []
        
        # Convolutional layers for feature extraction
        filters = [32, 32, 64, 64, 128, 128]
        for i, num_filters in enumerate(filters):
            self.conv_layers.append(
                tf.keras.layers.Conv2D(
                    num_filters, 
                    (3, 3), 
                    strides=(2, 2), 
                    padding='same',
                    activation='relu',
                    name=f"conv_{i}"
                )
            )
            self.conv_layers.append(
                tf.keras.layers.BatchNormalization(name=f"bn_{i}")
            )
        
        # GRU for temporal modeling
        self.gru = tf.keras.layers.GRU(
            config.gst_reference_encoder_dim,
            return_sequences=False,
            return_state=False,
            name="gru"
        )
        
        # Final projection
        self.projection = tf.keras.layers.Dense(
            config.gst_reference_encoder_dim,
            name="reference_projection"
        )
        
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Extract prosody features from reference mel spectrogram.
        
        Args:
            inputs: Reference mel spectrogram [batch, time, n_mels]
            training: Training mode flag
            
        Returns:
            Reference embedding [batch, reference_dim]
        """
        # Add channel dimension: [batch, time, n_mels, 1]
        x = tf.expand_dims(inputs, -1)
        
        # Apply convolutional layers
        for layer in self.conv_layers:
            x = layer(x, training=training)
        
        # Reshape for RNN: [batch, time', features]
        batch_size = tf.shape(x)[0]
        time_dim = tf.shape(x)[1]
        x = tf.reshape(x, [batch_size, time_dim, -1])
        
        # Apply GRU and project to embedding space
        gru_output = self.gru(x, training=training)
        reference_embedding = self.projection(gru_output, training=training)
        
        return reference_embedding


class GlobalStyleToken(tf.keras.layers.Layer):
    """
    Global Style Token (GST) module for controllable prosody synthesis.
    
    This module learns a bank of style embeddings and uses attention to 
    select appropriate styles based on reference audio or style control.
    """
    
    def __init__(self, config: ModelConfig, name: str = "global_style_token"):
        super().__init__(name=name)
        
        self.config = config
        self.num_style_tokens = config.gst_num_style_tokens
        self.style_token_dim = config.gst_style_token_dim
        self.num_heads = config.gst_num_heads
        
        # Reference encoder for extracting prosody from reference audio
        self.reference_encoder = ReferenceEncoder(config, name="reference_encoder")
        
        # Bank of learnable style tokens
        self.style_tokens = self.add_weight(
            name="style_tokens",
            shape=(self.num_style_tokens, self.style_token_dim),
            initializer="glorot_uniform",
            trainable=True
        )
        
        # Multi-head attention for style selection
        self.multihead_attention = MultiHeadAttention(
            self.style_token_dim,
            self.num_heads,
            name="style_attention"
        )
        
        # Output projection
        self.output_projection = tf.keras.layers.Dense(
            config.gst_style_embedding_dim,
            name="style_output_projection"
        )
        
    def call(
        self, 
        reference_mel: Optional[tf.Tensor] = None,
        style_weights: Optional[tf.Tensor] = None,
        training: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Generate style embedding from reference audio or style weights.
        
        Args:
            reference_mel: Reference mel spectrogram [batch, time, n_mels]
            style_weights: Direct style weights [batch, num_style_tokens] 
            training: Training mode flag
            
        Returns:
            Tuple of (style_embedding, attention_weights)
            - style_embedding: [batch, style_embedding_dim] 
            - attention_weights: [batch, num_heads, 1, num_style_tokens]
        """
        batch_size = tf.shape(reference_mel)[0] if reference_mel is not None else tf.shape(style_weights)[0]
        
        if reference_mel is not None:
            # Extract reference embedding from mel spectrogram
            reference_embedding = self.reference_encoder(reference_mel, training=training)
            # Add sequence dimension for attention: [batch, 1, dim]
            query = tf.expand_dims(reference_embedding, 1)
        elif style_weights is not None:
            # Use provided style weights directly
            # Convert to embedding by weighted combination of style tokens
            style_embedding = tf.matmul(style_weights, self.style_tokens)  # [batch, style_token_dim]
            query = tf.expand_dims(style_embedding, 1)  # [batch, 1, style_token_dim]
        else:
            # Default to neutral style (average of all style tokens)
            neutral_weights = tf.ones((batch_size, self.num_style_tokens)) / self.num_style_tokens
            style_embedding = tf.matmul(neutral_weights, self.style_tokens)
            query = tf.expand_dims(style_embedding, 1)
        
        # Prepare style tokens as keys and values: [batch, num_tokens, token_dim]
        style_tokens_batch = tf.tile(
            tf.expand_dims(self.style_tokens, 0), 
            [batch_size, 1, 1]
        )
        
        # Apply multi-head attention to select relevant styles
        style_output, attention_weights = self.multihead_attention(
            query=query,
            key=style_tokens_batch,
            value=style_tokens_batch,
            training=training
        )
        
        # Remove sequence dimension and project to final embedding size
        style_output = tf.squeeze(style_output, axis=1)  # [batch, style_token_dim]
        style_embedding = self.output_projection(style_output, training=training)
        
        return style_embedding, attention_weights


class ProsodyPredictor(tf.keras.layers.Layer):
    """Prosody predictor that conditions on style embedding."""
    
    def __init__(self, config: ModelConfig, name: str = "prosody_predictor"):
        super().__init__(name=name)
        
        self.config = config
        
        # Style conditioning layer (match text encoder dimensionality)
        projection_dim = getattr(config, "text_encoder_dim", config.decoder_dim)
        self.style_projection = tf.keras.layers.Dense(
            projection_dim,
            name="style_projection"
        )
        
        # Prosody prediction layers
        self.prosody_layers = [
            tf.keras.layers.Dense(
                config.decoder_dim,
                activation='relu',
                name=f"prosody_layer_{i}"
            )
            for i in range(2)
        ]
        
        # Specific prosody feature predictions
        self.pitch_predictor = tf.keras.layers.Dense(1, name="pitch_predictor")
        self.energy_predictor = tf.keras.layers.Dense(1, name="energy_predictor") 
        self.speaking_rate_predictor = tf.keras.layers.Dense(1, name="speaking_rate_predictor")
        
    def call(
        self, 
        text_features: tf.Tensor,
        style_embedding: tf.Tensor,
        training: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Predict prosody features conditioned on style embedding.
        
        Args:
            text_features: Text encoder features [batch, seq_len, dim]
            style_embedding: Style embedding [batch, style_dim]
            training: Training mode flag
            
        Returns:
            Tuple of (pitch, energy, speaking_rate) predictions
        """
        # Project style embedding and broadcast to text sequence
        style_projected = self.style_projection(style_embedding, training=training)
        style_expanded = tf.expand_dims(style_projected, 1)  # [batch, 1, dim]
        
        seq_len = tf.shape(text_features)[1]
        style_broadcast = tf.tile(style_expanded, [1, seq_len, 1])  # [batch, seq_len, dim]
        
        # Combine text features with style conditioning
        combined_features = text_features + style_broadcast
        
        # Apply prosody prediction layers
        x = combined_features
        for layer in self.prosody_layers:
            x = layer(x, training=training)
        
        # Predict specific prosody features
        pitch = self.pitch_predictor(x, training=training)
        energy = self.energy_predictor(x, training=training)
        speaking_rate = self.speaking_rate_predictor(x, training=training)
        
        return pitch, energy, speaking_rate
