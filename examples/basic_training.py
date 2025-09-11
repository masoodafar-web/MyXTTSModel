"""
Basic Training Example for MyXTTS.

This example demonstrates how to train a MyXTTS model on the LJSpeech dataset
with default configurations.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from myxtts.config.config import XTTSConfig
from myxtts.models.xtts import XTTS
from myxtts.training.trainer import XTTSTrainer
from myxtts.utils.commons import set_random_seed


def main():
    """Basic training example."""
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Create configuration with default settings
    config = XTTSConfig()
    
    # Update data configuration for LJSpeech
    config.data.dataset_path = "./data/ljspeech"
    config.data.dataset_name = "ljspeech"
    config.data.batch_size = 16  # Adjust based on GPU memory
    config.data.language = "en"
    
    # Update training configuration
    config.training.epochs = 100
    config.training.learning_rate = 1e-4
    config.training.checkpoint_dir = "./checkpoints"
    config.training.save_step = 5000
    config.training.val_step = 1000
    config.training.log_step = 100
    
    # Create model
    print("Creating XTTS model...")
    model = XTTS(config.model)
    
    # Print model information
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Text vocab size: {config.model.text_vocab_size}")
    print(f"Sample rate: {config.model.sample_rate}")
    print(f"Languages: {config.model.languages}")
    
    # Create trainer
    print("Creating trainer...")
    trainer = XTTSTrainer(config=config, model=model)
    
    # Prepare datasets
    print("Preparing datasets...")
    train_dataset, val_dataset = trainer.prepare_datasets(
        train_data_path=config.data.dataset_path
    )
    
    # Start training
    print("Starting training...")
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=config.training.epochs
    )
    
    print("Training completed!")


if __name__ == "__main__":
    main()