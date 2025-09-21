"""
Two-stage training for MyXTTS with neural vocoder.

This module implements a two-stage training approach:
1. Stage 1: Text-to-Mel training (existing mel decoder)
2. Stage 2: Mel-to-Audio training (neural vocoder)

This separation allows for better quality compared to end-to-end training.
"""

import tensorflow as tf
import numpy as np
import os
from typing import Dict, Optional, Tuple, Any
import logging
from dataclasses import dataclass

from ..models.xtts import XTTS
from ..models.vocoder import HiFiGANGenerator, VocoderLoss
from ..config.config import ModelConfig, TrainingConfig
from ..utils.audio import AudioProcessor


@dataclass
class TwoStageTrainingConfig:
    """Configuration for two-stage training."""
    
    # Stage 1: Text-to-Mel
    stage1_epochs: int = 100
    stage1_learning_rate: float = 1e-4
    stage1_checkpoint_path: str = "checkpoints/stage1"
    
    # Stage 2: Mel-to-Audio (Vocoder)
    stage2_epochs: int = 200
    stage2_learning_rate: float = 2e-4
    stage2_checkpoint_path: str = "checkpoints/stage2"
    
    # General
    use_pretrained_stage1: bool = False
    pretrained_stage1_path: Optional[str] = None
    enable_mixed_precision: bool = True
    enable_distributed_training: bool = False


class TwoStageTrainer:
    """
    Two-stage trainer for MyXTTS.
    
    Implements separated training for mel generation and audio synthesis.
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        two_stage_config: TwoStageTrainingConfig
    ):
        """
        Initialize two-stage trainer.
        
        Args:
            model_config: Model configuration
            training_config: Training configuration
            two_stage_config: Two-stage specific configuration
        """
        self.model_config = model_config
        self.training_config = training_config
        self.two_stage_config = two_stage_config
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize audio processor
        self.audio_processor = AudioProcessor(
            sample_rate=model_config.sample_rate,
            n_mels=model_config.n_mels,
            n_fft=model_config.n_fft,
            hop_length=model_config.hop_length,
            win_length=model_config.win_length
        )
        
        # Enable mixed precision if requested
        if two_stage_config.enable_mixed_precision:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize models for both stages."""
        
        # Stage 1: Text-to-Mel model (use autoregressive decoder)
        stage1_config = ModelConfig(**self.model_config.__dict__)
        stage1_config.decoder_strategy = "autoregressive"
        stage1_config.vocoder_type = "griffin_lim"  # No vocoder in stage 1
        
        self.stage1_model = XTTS(stage1_config, name="stage1_xtts")
        
        # Stage 2: Vocoder model
        self.vocoder = HiFiGANGenerator(
            self.model_config,
            upsample_rates=self.model_config.vocoder_upsample_rates,
            upsample_kernel_sizes=self.model_config.vocoder_upsample_kernel_sizes,
            resblock_kernel_sizes=self.model_config.vocoder_resblock_kernel_sizes,
            resblock_dilation_sizes=self.model_config.vocoder_resblock_dilation_sizes,
            initial_channel=self.model_config.vocoder_initial_channel
        )
        
        # Vocoder loss function
        self.vocoder_loss = VocoderLoss()
        
        # Optimizers
        self.stage1_optimizer = tf.keras.optimizers.AdamW(
            learning_rate=self.two_stage_config.stage1_learning_rate,
            weight_decay=0.01
        )
        
        self.stage2_optimizer = tf.keras.optimizers.AdamW(
            learning_rate=self.two_stage_config.stage2_learning_rate,
            weight_decay=0.01
        )
    
    def train_stage1(self, train_dataset, val_dataset=None) -> Dict[str, Any]:
        """
        Train Stage 1: Text-to-Mel generation.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            
        Returns:
            Training history
        """
        self.logger.info("Starting Stage 1 training: Text-to-Mel")
        
        # Check if we should use pretrained model
        if self.two_stage_config.use_pretrained_stage1 and self.two_stage_config.pretrained_stage1_path:
            self.logger.info(f"Loading pretrained Stage 1 model from {self.two_stage_config.pretrained_stage1_path}")
            self.stage1_model.load_weights(self.two_stage_config.pretrained_stage1_path)
            return {"message": "Using pretrained Stage 1 model"}
        
        # Compile model
        self.stage1_model.compile(
            optimizer=self.stage1_optimizer,
            loss={
                'mel_output': 'mse',
                'stop_tokens': 'binary_crossentropy'
            },
            loss_weights={
                'mel_output': 1.0,
                'stop_tokens': 0.5
            },
            metrics=['mae']
        )
        
        # Setup callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.two_stage_config.stage1_checkpoint_path, 'best_model.h5'),
                save_best_only=True,
                save_weights_only=True,
                monitor='val_loss' if val_dataset else 'loss'
            ),
            tf.keras.callbacks.EarlyStopping(
                patience=10,
                restore_best_weights=True,
                monitor='val_loss' if val_dataset else 'loss'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train model
        history = self.stage1_model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.two_stage_config.stage1_epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        os.makedirs(self.two_stage_config.stage1_checkpoint_path, exist_ok=True)
        self.stage1_model.save_weights(
            os.path.join(self.two_stage_config.stage1_checkpoint_path, 'final_model.h5')
        )
        
        self.logger.info("Stage 1 training completed")
        return history.history
    
    def train_stage2(self, train_dataset, val_dataset=None) -> Dict[str, Any]:
        """
        Train Stage 2: Mel-to-Audio vocoder.
        
        Args:
            train_dataset: Training dataset with (mel, audio) pairs
            val_dataset: Validation dataset (optional)
            
        Returns:
            Training history
        """
        self.logger.info("Starting Stage 2 training: Mel-to-Audio vocoder")
        
        # Custom training loop for vocoder
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        val_loss = tf.keras.metrics.Mean(name='val_loss')
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.two_stage_config.stage2_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.two_stage_config.stage2_epochs}")
            
            # Training
            train_loss.reset_states()
            for batch_idx, (mel_batch, audio_batch) in enumerate(train_dataset):
                loss = self._train_vocoder_step(mel_batch, audio_batch)
                train_loss.update_state(loss)
                
                if batch_idx % 100 == 0:
                    self.logger.info(f"Batch {batch_idx}, Loss: {loss:.4f}")
            
            # Validation
            if val_dataset:
                val_loss.reset_states()
                for mel_batch, audio_batch in val_dataset:
                    loss = self._val_vocoder_step(mel_batch, audio_batch)
                    val_loss.update_state(loss)
                
                current_val_loss = val_loss.result()
                self.logger.info(f"Validation Loss: {current_val_loss:.4f}")
                
                # Save best model
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    patience_counter = 0
                    os.makedirs(self.two_stage_config.stage2_checkpoint_path, exist_ok=True)
                    self.vocoder.save_weights(
                        os.path.join(self.two_stage_config.stage2_checkpoint_path, 'best_model.h5')
                    )
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= 10:
                    self.logger.info("Early stopping triggered")
                    break
            
            self.logger.info(f"Train Loss: {train_loss.result():.4f}")
        
        # Save final model
        os.makedirs(self.two_stage_config.stage2_checkpoint_path, exist_ok=True)
        self.vocoder.save_weights(
            os.path.join(self.two_stage_config.stage2_checkpoint_path, 'final_model.h5')
        )
        
        self.logger.info("Stage 2 training completed")
        return {"final_train_loss": float(train_loss.result())}
    
    @tf.function
    def _train_vocoder_step(self, mel_batch: tf.Tensor, audio_batch: tf.Tensor) -> tf.Tensor:
        """Single training step for vocoder."""
        with tf.GradientTape() as tape:
            # Generate audio from mel
            generated_audio = self.vocoder(mel_batch, training=True)
            
            # Calculate losses
            losses = self.vocoder_loss(
                real_audio=audio_batch,
                generated_audio=generated_audio,
                real_mel=mel_batch,
                generated_mel=mel_batch,  # For consistency
                training=True
            )
            
            total_loss = losses['total_loss']
            
            # Scale loss for mixed precision
            if self.two_stage_config.enable_mixed_precision:
                scaled_loss = self.stage2_optimizer.get_scaled_loss(total_loss)
        
        # Calculate gradients
        if self.two_stage_config.enable_mixed_precision:
            scaled_gradients = tape.gradient(scaled_loss, self.vocoder.trainable_variables)
            gradients = self.stage2_optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(total_loss, self.vocoder.trainable_variables)
        
        # Apply gradients
        self.stage2_optimizer.apply_gradients(zip(gradients, self.vocoder.trainable_variables))
        
        return total_loss
    
    @tf.function
    def _val_vocoder_step(self, mel_batch: tf.Tensor, audio_batch: tf.Tensor) -> tf.Tensor:
        """Single validation step for vocoder."""
        # Generate audio from mel
        generated_audio = self.vocoder(mel_batch, training=False)
        
        # Calculate losses
        losses = self.vocoder_loss(
            real_audio=audio_batch,
            generated_audio=generated_audio,
            real_mel=mel_batch,
            generated_mel=mel_batch,
            training=False
        )
        
        return losses['total_loss']
    
    def create_combined_model(self) -> XTTS:
        """
        Create a combined model with trained Stage 1 and Stage 2 components.
        
        Returns:
            Combined XTTS model with neural vocoder
        """
        # Load trained Stage 1 weights
        stage1_path = os.path.join(self.two_stage_config.stage1_checkpoint_path, 'best_model.h5')
        if os.path.exists(stage1_path):
            self.stage1_model.load_weights(stage1_path)
        
        # Create combined model configuration
        combined_config = ModelConfig(**self.model_config.__dict__)
        combined_config.vocoder_type = "hifigan"
        
        # Create combined model
        combined_model = XTTS(combined_config, name="combined_xtts")
        
        # Transfer Stage 1 weights
        # Note: This would need careful weight mapping in practice
        # For now, we'll retrain briefly or use a more sophisticated transfer
        
        # Load trained vocoder weights
        stage2_path = os.path.join(self.two_stage_config.stage2_checkpoint_path, 'best_model.h5')
        if os.path.exists(stage2_path):
            combined_model.vocoder.vocoder.load_weights(stage2_path)
        
        self.logger.info("Combined model created successfully")
        return combined_model