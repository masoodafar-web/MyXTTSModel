"""Model compression utilities for creating lightweight XTTS models.

This module provides tools for:
- Weight pruning to reduce model size
- Quantization-aware training for efficient inference
- Model architecture modifications for deployment
"""

import os
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class CompressionConfig:
    """Configuration for model compression."""
    # Pruning configuration
    enable_pruning: bool = True
    pruning_schedule: str = "polynomial"  # "polynomial" or "constant"
    initial_sparsity: float = 0.0
    final_sparsity: float = 0.5  # Remove 50% of weights
    pruning_frequency: int = 100  # Steps between pruning updates
    
    # Quantization configuration
    enable_quantization: bool = True
    quantization_method: str = "int8"  # "int8" or "int16"
    calibration_steps: int = 100
    
    # Architecture modifications
    reduce_decoder_layers: bool = True
    target_decoder_layers: int = 8  # Reduce from 16 to 8
    reduce_attention_heads: bool = True
    target_attention_heads: int = 12  # Reduce from 24 to 12
    reduce_decoder_dim: bool = True
    target_decoder_dim: int = 768  # Reduce from 1536 to 768
    
    # Performance targets
    target_speedup: float = 2.0  # Target 2x speedup
    max_quality_loss: float = 0.1  # Maximum 10% quality loss


class ModelCompressor:
    """Main model compression class for XTTS optimization."""
    
    def __init__(self, config: CompressionConfig):
        """Initialize model compressor.
        
        Args:
            config: Compression configuration
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required for model compression")
        
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ModelCompressor")
        
        # Import TensorFlow Model Optimization toolkit
        try:
            import tensorflow_model_optimization as tfmot
            self.tfmot = tfmot
        except ImportError:
            self.logger.warning("TensorFlow Model Optimization not available - some features disabled")
            self.tfmot = None
    
    def compress_model(self, 
                      model: tf.keras.Model,
                      train_dataset: Optional[tf.data.Dataset] = None) -> tf.keras.Model:
        """Apply comprehensive compression to XTTS model.
        
        Args:
            model: Original XTTS model
            train_dataset: Training dataset for calibration
            
        Returns:
            Compressed model
        """
        self.logger.info("Starting model compression...")
        
        compressed_model = model
        
        # Step 1: Architecture modifications
        if self._should_modify_architecture():
            self.logger.info("Applying architecture modifications...")
            compressed_model = self._modify_architecture(compressed_model)
        
        # Step 2: Pruning
        if self.config.enable_pruning and self.tfmot:
            self.logger.info("Applying weight pruning...")
            compressed_model = self._apply_pruning(compressed_model)
        
        # Step 3: Quantization-aware training preparation
        if self.config.enable_quantization and self.tfmot:
            self.logger.info("Preparing for quantization-aware training...")
            compressed_model = self._prepare_quantization(compressed_model)
        
        self.logger.info("Model compression completed")
        return compressed_model
    
    def _should_modify_architecture(self) -> bool:
        """Check if architecture modifications should be applied."""
        return (self.config.reduce_decoder_layers or 
                self.config.reduce_attention_heads or 
                self.config.reduce_decoder_dim)
    
    def _modify_architecture(self, model: tf.keras.Model) -> tf.keras.Model:
        """Modify model architecture for efficiency."""
        # This is a placeholder for architecture modification
        # In practice, this would involve rebuilding the model with smaller dimensions
        
        self.logger.info("Architecture modifications:")
        if self.config.reduce_decoder_layers:
            self.logger.info(f"  - Decoder layers: ? → {self.config.target_decoder_layers}")
        if self.config.reduce_attention_heads:
            self.logger.info(f"  - Attention heads: ? → {self.config.target_attention_heads}")
        if self.config.reduce_decoder_dim:
            self.logger.info(f"  - Decoder dimension: ? → {self.config.target_decoder_dim}")
        
        # For now, return the original model
        # TODO: Implement actual architecture modification
        self.logger.warning("Architecture modification not yet implemented - returning original model")
        return model
    
    def _apply_pruning(self, model: tf.keras.Model) -> tf.keras.Model:
        """Apply structured pruning to model weights."""
        if not self.tfmot:
            self.logger.warning("TensorFlow Model Optimization not available - skipping pruning")
            return model
        
        # Define pruning schedule
        if self.config.pruning_schedule == "polynomial":
            pruning_schedule = self.tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=self.config.initial_sparsity,
                final_sparsity=self.config.final_sparsity,
                begin_step=0,
                end_step=1000,  # Will be adjusted based on training
                frequency=self.config.pruning_frequency
            )
        else:
            pruning_schedule = self.tfmot.sparsity.keras.ConstantSparsity(
                target_sparsity=self.config.final_sparsity,
                begin_step=0,
                frequency=self.config.pruning_frequency
            )
        
        # Apply pruning to dense layers
        def apply_pruning_to_dense(layer):
            if isinstance(layer, tf.keras.layers.Dense):
                return self.tfmot.sparsity.keras.prune_low_magnitude(
                    layer, pruning_schedule=pruning_schedule
                )
            return layer
        
        # Clone model with pruning
        pruned_model = tf.keras.models.clone_model(
            model,
            clone_function=apply_pruning_to_dense
        )
        
        self.logger.info(f"Applied pruning with {self.config.final_sparsity:.1%} target sparsity")
        return pruned_model
    
    def _prepare_quantization(self, model: tf.keras.Model) -> tf.keras.Model:
        """Prepare model for quantization-aware training."""
        if not self.tfmot:
            self.logger.warning("TensorFlow Model Optimization not available - skipping quantization")
            return model
        
        # Apply quantization-aware training
        q_aware_model = self.tfmot.quantization.keras.quantize_model(model)
        
        self.logger.info(f"Prepared model for {self.config.quantization_method} quantization")
        return q_aware_model
    
    def finalize_compression(self, model: tf.keras.Model) -> tf.keras.Model:
        """Finalize compression after training."""
        if not self.tfmot:
            return model
        
        # Remove pruning wrappers and apply final sparsity
        if self.config.enable_pruning:
            model = self.tfmot.sparsity.keras.strip_pruning(model)
            self.logger.info("Removed pruning wrappers")
        
        return model
    
    def get_compression_stats(self, 
                            original_model: tf.keras.Model,
                            compressed_model: tf.keras.Model) -> Dict:
        """Get compression statistics."""
        # Count parameters
        original_params = original_model.count_params()
        compressed_params = compressed_model.count_params()
        
        # Estimate model sizes (assuming float32)
        original_size_mb = original_params * 4 / (1024 * 1024)
        compressed_size_mb = compressed_params * 4 / (1024 * 1024)
        
        compression_ratio = original_params / compressed_params if compressed_params > 0 else 1.0
        size_reduction = (original_size_mb - compressed_size_mb) / original_size_mb if original_size_mb > 0 else 0.0
        
        return {
            'original_parameters': original_params,
            'compressed_parameters': compressed_params,
            'compression_ratio': compression_ratio,
            'original_size_mb': original_size_mb,
            'compressed_size_mb': compressed_size_mb,
            'size_reduction_percent': size_reduction * 100,
            'estimated_speedup': min(compression_ratio, self.config.target_speedup)
        }


class QuantizationAwareTrainer:
    """Trainer for quantization-aware training of compressed models."""
    
    def __init__(self, config: CompressionConfig):
        """Initialize quantization-aware trainer.
        
        Args:
            config: Compression configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.QuantizationAwareTrainer")
    
    def train(self, 
              model: tf.keras.Model,
              train_dataset: tf.data.Dataset,
              val_dataset: tf.data.Dataset,
              epochs: int = 10,
              **kwargs) -> tf.keras.Model:
        """Train model with quantization awareness.
        
        Args:
            model: Quantization-aware model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of training epochs
            **kwargs: Additional training parameters
            
        Returns:
            Trained quantization-aware model
        """
        self.logger.info(f"Starting quantization-aware training for {epochs} epochs")
        
        # Configure optimizer for QAT
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)  # Lower learning rate for QAT
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=kwargs.get('loss', 'mse'),
            metrics=kwargs.get('metrics', ['mae'])
        )
        
        # Training callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-7
            )
        ]
        
        # Train model
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        self.logger.info("Quantization-aware training completed")
        return model
    
    def convert_to_tflite(self, 
                         model: tf.keras.Model,
                         calibration_dataset: Optional[tf.data.Dataset] = None) -> bytes:
        """Convert trained model to TensorFlow Lite format.
        
        Args:
            model: Trained quantization-aware model
            calibration_dataset: Dataset for post-training quantization calibration
            
        Returns:
            TensorFlow Lite model as bytes
        """
        self.logger.info("Converting to TensorFlow Lite...")
        
        # Create TFLite converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Configure quantization
        if self.config.quantization_method == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            if calibration_dataset:
                converter.representative_dataset = self._create_representative_dataset(calibration_dataset)
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
        
        # Convert model
        tflite_model = converter.convert()
        
        self.logger.info(f"TensorFlow Lite model created ({len(tflite_model)} bytes)")
        return tflite_model
    
    def _create_representative_dataset(self, dataset: tf.data.Dataset):
        """Create representative dataset for quantization calibration."""
        def representative_data_gen():
            count = 0
            for sample in dataset.take(self.config.calibration_steps):
                if isinstance(sample, tuple):
                    yield [sample[0]]  # Use input data
                else:
                    yield [sample]
                count += 1
                if count >= self.config.calibration_steps:
                    break
        
        return representative_data_gen


def create_lightweight_config(base_config) -> Dict:
    """Create a lightweight model configuration for real-time inference.
    
    Args:
        base_config: Original model configuration
        
    Returns:
        Lightweight configuration dictionary
    """
    # Create a copy of base configuration
    lightweight_config = {}
    
    # Model architecture reductions
    lightweight_config.update({
        # Reduce decoder complexity (main bottleneck)
        'decoder_dim': 768,          # Reduced from 1536 (50% reduction)
        'decoder_layers': 8,         # Reduced from 16 (50% reduction)
        'decoder_heads': 12,         # Reduced from 24 (50% reduction)
        
        # Reduce text encoder
        'text_encoder_layers': 4,    # Reduced from 8 (50% reduction)
        'text_encoder_heads': 4,     # Reduced from 8 (50% reduction)
        
        # Reduce audio encoder
        'audio_encoder_dim': 512,    # Reduced from 768
        'audio_encoder_layers': 4,   # Reduced from 8
        'audio_encoder_heads': 8,    # Reduced from 12
        
        # Reduce speaker embedding
        'speaker_embedding_dim': 256, # Reduced from 512
        
        # Optimization settings
        'enable_gradient_checkpointing': True,
        'max_attention_sequence_length': 256,  # Reduced from 512
        
        # Reduce mel spectrogram resolution for faster processing
        'n_mels': 64,               # Reduced from 80
        'hop_length': 512,          # Increased from 256 (less time resolution)
        
        # Faster vocoder settings
        'vocoder_type': 'griffin_lim',  # Faster than neural vocoder
        'decoder_strategy': 'non_autoregressive',  # Faster than autoregressive
    })
    
    return lightweight_config
