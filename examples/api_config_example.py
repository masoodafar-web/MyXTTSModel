#!/usr/bin/env python3
"""
Example of using MyXTTS with API-based configuration.

This example demonstrates how to:
1. Load configuration from an API endpoint instead of a YAML file
2. Smart dataset handling (download if doesn't exist, read from local if exists)
3. Use the same interface pattern as YAML-based configuration

Usage:
    python examples/api_config_example.py --api-url YOUR_API_URL [--api-key YOUR_API_KEY]
"""

import argparse
import os
import tempfile
from myxtts import XTTS, XTTSConfig, XTTSInference, XTTSTrainer


def create_sample_config_api_response():
    """
    Create a sample configuration that would be returned by an API.
    In a real scenario, this would come from your configuration API endpoint.
    """
    return {
        "data": {
            "add_blank": True,
            "batch_size": 32,
            "dataset_name": "ljspeech",
            "dataset_path": "./data/ljspeech",
            "language": "en",
            "max_audio_length": 11.0,
            "min_audio_length": 1.0,
            "normalize_audio": True,
            "num_workers": 4,
            "reference_audio_length": 3.0,
            "sample_rate": 22050,
            "text_cleaners": ["english_cleaners"],
            "train_split": 0.9,
            "trim_silence": True,
            "val_split": 0.1
        },
        "model": {
            "audio_encoder_dim": 512,
            "audio_encoder_heads": 8,
            "audio_encoder_layers": 6,
            "decoder_dim": 1024,
            "decoder_heads": 16,
            "decoder_layers": 12,
            "hop_length": 256,
            "languages": [
                "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "hu", "ko"
            ],
            "max_text_length": 500,
            "n_fft": 1024,
            "n_mels": 80,
            "sample_rate": 22050,
            "speaker_embedding_dim": 256,
            "text_encoder_dim": 512,
            "text_encoder_heads": 8,
            "text_encoder_layers": 6,
            "text_vocab_size": 256256,
            "tokenizer_model": "facebook/nllb-200-distilled-600M",
            "tokenizer_type": "nllb",
            "use_voice_conditioning": True,
            "win_length": 1024
        },
        "training": {
            "beta1": 0.9,
            "beta2": 0.999,
            "checkpoint_dir": "./checkpoints",
            "duration_loss_weight": 1.0,
            "epochs": 200,
            "eps": 1e-08,
            "gradient_clip_norm": 1.0,
            "kl_loss_weight": 1.0,
            "learning_rate": 5e-05,
            "log_step": 100,
            "mel_loss_weight": 45.0,
            "optimizer": "adamw",
            "save_step": 1000,
            "scheduler": "noam",
            "scheduler_params": {},
            "use_wandb": False,
            "val_step": 500,
            "wandb_project": "myxtts",
            "warmup_steps": 4000,
            "weight_decay": 1e-06
        }
    }


def run_mock_api_server():
    """
    For demonstration purposes, create a simple mock API server.
    In real usage, you would use your actual API endpoint.
    """
    import json
    import tempfile
    import http.server
    import socketserver
    import threading
    import time
    
    class MockAPIHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/config':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                config_data = create_sample_config_api_response()
                self.wfile.write(json.dumps(config_data).encode())
            else:
                self.send_response(404)
                self.end_headers()
        
        def log_message(self, format, *args):
            # Suppress default logging
            pass
    
    # Start mock server on a random available port
    with socketserver.TCPServer(("", 0), MockAPIHandler) as httpd:
        port = httpd.server_address[1]
        
        # Run server in background thread
        server_thread = threading.Thread(target=httpd.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        
        return f"http://localhost:{port}/config", httpd


def main():
    parser = argparse.ArgumentParser(description='MyXTTS API Configuration Example')
    parser.add_argument('--api-url', type=str, help='API URL for configuration')
    parser.add_argument('--api-key', type=str, help='API key for authentication')
    parser.add_argument('--use-mock', action='store_true', default=True,
                        help='Use mock API server for demo (default: True)')
    
    args = parser.parse_args()
    
    # Determine API URL
    if args.use_mock or not args.api_url:
        print("🚀 Starting mock API server for demonstration...")
        api_url, mock_server = run_mock_api_server()
        print(f"📡 Mock API server running at: {api_url}")
    else:
        api_url = args.api_url
        mock_server = None
    
    try:
        print("\n" + "="*60)
        print("MyXTTS API Configuration Example")
        print("="*60)
        
        # Load configuration from API (not from YAML file)
        print(f"📥 Loading configuration from API: {api_url}")
        config = XTTSConfig.from_api(api_url, api_key=args.api_key)
        print("✅ Configuration loaded successfully from API!")
        
        print(f"\n📊 Configuration Summary:")
        print(f"   Dataset: {config.data.dataset_name}")
        print(f"   Dataset Path: {config.data.dataset_path}")
        print(f"   Languages: {config.model.languages[:3]}... ({len(config.model.languages)} total)")
        print(f"   Sample Rate: {config.model.sample_rate}")
        print(f"   Training Epochs: {config.training.epochs}")
        
        # Create and initialize model
        print(f"\n🤖 Creating XTTS model...")
        model = XTTS(config.model)
        print("✅ Model created successfully!")
        
        # Create trainer with smart dataset handling
        print(f"\n🎯 Initializing trainer with smart dataset handling...")
        trainer = XTTSTrainer(config, model)
        print("✅ Trainer initialized successfully!")
        
        # The smart dataset handling happens when preparing datasets
        print(f"\n📁 Preparing datasets (with smart local/download logic)...")
        # Note: This would normally download the dataset if not present locally
        # For this example, we'll just demonstrate the interface
        print(f"   Dataset path: {config.data.dataset_path}")
        
        if os.path.exists(config.data.dataset_path):
            print("   ✅ Dataset found locally - will read from local path")
        else:
            print("   📥 Dataset not found locally - will be downloaded when train() is called")
        
        # Uncomment the following lines to actually prepare datasets and train:
        # train_dataset, val_dataset = trainer.prepare_datasets(config.data.dataset_path)
        # trainer.train(train_dataset, val_dataset)
        
        # Initialize inference engine
        print(f"\n🎙️  Setting up inference engine...")
        # For this example, we'll create the inference without a checkpoint
        inference = XTTSInference(config, checkpoint_path=None)
        print("✅ Inference engine created successfully!")
        
        # Example inference call (would need a trained model checkpoint)
        print(f"\n🗣️  Inference example:")
        print('   inference = XTTSInference(config, checkpoint_path="model.ckpt")')
        print('   result = inference.synthesize("Hello world!", reference_audio="speaker.wav")')
        
        print(f"\n🎉 Example completed successfully!")
        print(f"\n💡 Key Features Demonstrated:")
        print(f"   ✅ API-based configuration loading (not YAML)")
        print(f"   ✅ Smart dataset handling (local vs download)")
        print(f"   ✅ Same interface pattern as YAML-based config")
        print(f"   ✅ Integrated model, trainer, and inference initialization")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    finally:
        # Clean up mock server if it was started
        if mock_server:
            mock_server.shutdown()
            print(f"\n🛑 Mock API server stopped")
    
    return 0


if __name__ == "__main__":
    exit(main())