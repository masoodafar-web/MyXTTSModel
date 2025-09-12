#!/usr/bin/env python3
"""
Direct usage example matching the pattern requested in the problem statement.

This example shows the exact usage pattern requested:
- Load configuration from API (not YAML)
- Smart dataset handling
- Same interface as YAML-based approach
"""

from myxtts import XTTS, XTTSConfig, XTTSInference, XTTSTrainer

# Example API configuration URL
# In real usage, replace with your actual API endpoint
API_CONFIG_URL = "https://your-api-server.com/config"
API_KEY = "your-api-key"  # Optional

def main():
    """Direct usage example as requested in the problem statement."""
    
    # Load configuration from API (not from YAML file)
    # The API should return a JSON with the same structure as the YAML config
    config = XTTSConfig.from_api(API_CONFIG_URL, api_key=API_KEY)
    
    # Important note: Smart dataset handling is built-in
    # If the dataset path from config exists locally, it reads from local path
    # If the dataset path doesn't exist, it will download the dataset automatically
    
    # Create and train model
    model = XTTS(config.model)
    trainer = XTTSTrainer(config, model)
    
    # Prepare datasets with smart handling (auto-download if needed)
    train_dataset, val_dataset = trainer.prepare_datasets(config.data.dataset_path)
    
    # Train the model
    trainer.train(train_dataset, val_dataset)
    
    # Inference
    inference = XTTSInference(config, checkpoint_path="model.ckpt")
    result = inference.synthesize("Hello world!", reference_audio="speaker.wav")
    
    return result


if __name__ == "__main__":
    # Note: This example requires a real API endpoint and trained model checkpoint
    # For testing, use the api_config_example.py which includes a mock API server
    print("This is a template showing the exact usage pattern.")
    print("For a working example with mock API, run: python examples/api_config_example.py")