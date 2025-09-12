#!/usr/bin/env python3
"""
Real-world example demonstrating API-based configuration.

This example shows how to use MyXTTS with configuration loaded from an API
instead of a YAML file, with smart dataset handling.
"""

from myxtts import XTTS, XTTSConfig, XTTSInference, XTTSTrainer

def example_with_real_api():
    """
    Example using a real API endpoint for configuration.
    Replace with your actual API endpoint.
    """
    
    # Example API endpoint that returns configuration JSON
    API_URL = "https://your-config-api.com/xtts/config"
    API_KEY = "your-api-key"  # Optional
    
    try:
        # Load configuration from API (not from YAML file)
        # The API should return a JSON with the same structure as example_config.yaml
        config = XTTSConfig.from_api(API_URL, api_key=API_KEY)
        
        print(f"✅ Configuration loaded from API: {API_URL}")
        print(f"📊 Dataset: {config.data.dataset_name}")
        print(f"📁 Dataset Path: {config.data.dataset_path}")
        
        # Smart dataset handling is automatic:
        # - If dataset exists at config.data.dataset_path -> reads from local
        # - If dataset doesn't exist -> downloads automatically
        
        # Create and train model
        model = XTTS(config.model)
        trainer = XTTSTrainer(config, model)
        
        # Prepare datasets (with smart dataset handling)
        train_dataset, val_dataset = trainer.prepare_datasets(config.data.dataset_path)
        
        # Train the model
        trainer.train(train_dataset, val_dataset)
        
        # Inference
        inference = XTTSInference(config, checkpoint_path="model.ckpt")
        result = inference.synthesize("Hello world!", reference_audio="speaker.wav")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure your API endpoint returns valid configuration JSON")
        return None


def example_api_json_format():
    """
    Shows the expected JSON format for the API response.
    Your API endpoint should return JSON in this format.
    """
    
    expected_format = {
        "data": {
            "dataset_path": "./data/ljspeech",  # Will be checked for local existence
            "dataset_name": "ljspeech",
            "batch_size": 32,
            "language": "en",
            "sample_rate": 22050,
            "text_cleaners": ["english_cleaners"],
            "train_split": 0.9,
            "val_split": 0.1,
            # ... other data config fields
        },
        "model": {
            "sample_rate": 22050,
            "languages": ["en", "es", "fr", "de", "it"],
            "text_encoder_dim": 512,
            "decoder_dim": 1024,
            # ... other model config fields
        },
        "training": {
            "epochs": 200,
            "learning_rate": 5e-05,
            "optimizer": "adamw",
            "checkpoint_dir": "./checkpoints",
            # ... other training config fields
        }
    }
    
    print("📋 Expected API JSON Format:")
    print("="*40)
    import json
    print(json.dumps(expected_format, indent=2))
    
    return expected_format


if __name__ == "__main__":
    print("🚀 MyXTTS API Configuration Example")
    print("="*50)
    
    print("\n📖 Usage Pattern:")
    print("""
from myxtts import XTTS, XTTSConfig, XTTSInference, XTTSTrainer

# Load configuration from API (not YAML file)
config = XTTSConfig.from_api("https://your-api.com/config", api_key="your-key")

# Important: Smart dataset handling built-in
# - If dataset exists locally -> read from local path
# - If dataset doesn't exist -> download automatically

# Create and train model  
model = XTTS(config.model)
trainer = XTTSTrainer(config, model)
train_dataset, val_dataset = trainer.prepare_datasets(config.data.dataset_path)
trainer.train(train_dataset, val_dataset)

# Inference
inference = XTTSInference(config, checkpoint_path="model.ckpt")
result = inference.synthesize("Hello world!", reference_audio="speaker.wav")
    """)
    
    print("\n📝 API Configuration Format:")
    example_api_json_format()
    
    print("\n💡 Key Features:")
    print("✅ API-based configuration loading")
    print("✅ Smart dataset handling (auto-download or local)")
    print("✅ Same interface as YAML-based configuration")
    print("✅ Direct imports: XTTS, XTTSConfig, XTTSInference, XTTSTrainer")
    
    print("\n🧪 For testing with mock API, run:")
    print("python examples/api_config_example.py --use-mock")