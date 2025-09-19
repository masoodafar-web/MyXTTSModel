"""
Basic Inference Example for MyXTTS.

This example demonstrates how to use a trained MyXTTS model for 
text-to-speech synthesis and voice cloning.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from myxtts.config.config import XTTSConfig
from myxtts.inference.synthesizer import XTTSInference


def basic_synthesis_example():
    """Basic text-to-speech synthesis."""
    print("=== Basic Synthesis Example ===")
    
    # Load configuration
    config = XTTSConfig()
    
    # Create inference engine
    # Note: Replace with actual checkpoint path when available
    checkpoint_path = "./checkpoints/checkpoint_best_model.weights.h5"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please train a model first or provide a valid checkpoint path.")
        return
    
    inference = XTTSInference(
        config=config,
        checkpoint_path=checkpoint_path
    )
    
    # Text to synthesize
    text = "Hello, this is MyXTTS, a TensorFlow-based text-to-speech system with voice cloning capabilities."
    
    print(f"Synthesizing: {text}")
    
    # Synthesize speech
    result = inference.synthesize(
        text=text,
        language="en",
        temperature=1.0
    )
    
    # Save output
    output_path = "output_basic.wav"
    inference.save_audio(result["audio"], output_path)
    
    print(f"Audio saved to: {output_path}")
    print(f"Duration: {len(result['audio']) / result['sample_rate']:.2f} seconds")


def voice_cloning_example():
    """Voice cloning example."""
    print("\n=== Voice Cloning Example ===")
    
    # Load configuration
    config = XTTSConfig()
    
    # Make sure voice conditioning is enabled
    config.model.use_voice_conditioning = True
    
    # Create inference engine
    checkpoint_path = "./checkpoints/checkpoint_best_model.weights.h5"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please train a model first or provide a valid checkpoint path.")
        return
    
    inference = XTTSInference(
        config=config,
        checkpoint_path=checkpoint_path
    )
    
    # Reference audio for voice cloning
    reference_audio_path = "./examples/reference_audio.wav"
    
    if not os.path.exists(reference_audio_path):
        print(f"Reference audio not found: {reference_audio_path}")
        print("Please provide a reference audio file for voice cloning.")
        return
    
    # Text to synthesize with cloned voice
    text = "This speech is synthesized using voice cloning from the reference audio."
    
    print(f"Cloning voice for: {text}")
    print(f"Reference audio: {reference_audio_path}")
    
    # Clone voice
    result = inference.clone_voice(
        text=text,
        reference_audio=reference_audio_path,
        language="en",
        temperature=0.7  # Lower temperature for more consistent voice
    )
    
    # Save output
    output_path = "output_cloned.wav"
    inference.save_audio(result["audio"], output_path)
    
    print(f"Cloned voice saved to: {output_path}")
    print(f"Duration: {len(result['audio']) / result['sample_rate']:.2f} seconds")


def multilingual_example():
    """Multilingual synthesis example."""
    print("\n=== Multilingual Example ===")
    
    # Load configuration
    config = XTTSConfig()
    
    # Create inference engine
    checkpoint_path = "./checkpoints/checkpoint_best_model.weights.h5"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please train a model first or provide a valid checkpoint path.")
        return
    
    inference = XTTSInference(
        config=config,
        checkpoint_path=checkpoint_path
    )
    
    # Texts in different languages
    texts_and_languages = [
        ("Hello, how are you today?", "en"),
        ("Hola, ¿cómo estás hoy?", "es"),
        ("Bonjour, comment allez-vous aujourd'hui?", "fr"),
        ("Hallo, wie geht es dir heute?", "de"),
    ]
    
    for i, (text, language) in enumerate(texts_and_languages):
        print(f"Synthesizing {language}: {text}")
        
        # Synthesize speech
        result = inference.synthesize(
            text=text,
            language=language,
            temperature=1.0
        )
        
        # Save output
        output_path = f"output_multilingual_{language}.wav"
        inference.save_audio(result["audio"], output_path)
        
        print(f"Audio saved to: {output_path}")


def batch_synthesis_example():
    """Batch synthesis example."""
    print("\n=== Batch Synthesis Example ===")
    
    # Load configuration
    config = XTTSConfig()
    
    # Create inference engine
    checkpoint_path = "./checkpoints/checkpoint_best_model.weights.h5"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please train a model first or provide a valid checkpoint path.")
        return
    
    inference = XTTSInference(
        config=config,
        checkpoint_path=checkpoint_path
    )
    
    # Multiple texts to synthesize
    texts = [
        "This is the first sentence.",
        "This is the second sentence.",
        "This is the third sentence.",
        "This is the fourth sentence."
    ]
    
    print(f"Synthesizing {len(texts)} texts...")
    
    # Batch synthesis
    results = inference.synthesize_batch(
        texts=texts,
        language="en",
        temperature=1.0
    )
    
    # Save outputs
    for i, result in enumerate(results):
        output_path = f"output_batch_{i+1}.wav"
        inference.save_audio(result["audio"], output_path)
        print(f"Audio {i+1} saved to: {output_path}")


def benchmark_example():
    """Benchmark inference speed."""
    print("\n=== Benchmark Example ===")
    
    # Load configuration
    config = XTTSConfig()
    
    # Create inference engine
    checkpoint_path = "./checkpoints/checkpoint_best_model.weights.h5"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please train a model first or provide a valid checkpoint path.")
        return
    
    inference = XTTSInference(
        config=config,
        checkpoint_path=checkpoint_path
    )
    
    # Test texts
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In a hole in the ground there lived a hobbit.",
        "To be or not to be, that is the question.",
        "The weather today is sunny with clouds."
    ]
    
    print("Running benchmark...")
    
    # Run benchmark
    results = inference.benchmark(
        texts=test_texts,
        num_runs=3
    )
    
    print("Benchmark Results:")
    print(f"  Real-time factor: {results['real_time_factor']:.2f}")
    print(f"  Average synthesis time: {results['average_synthesis_time']:.2f}s")
    print(f"  Average audio length: {results['average_audio_length']:.2f}s")
    print(f"  Throughput: {results['throughput_samples_per_second']:.2f} samples/s")


def main():
    """Run all examples."""
    print("MyXTTS Inference Examples")
    print("========================")
    
    # Run examples
    basic_synthesis_example()
    voice_cloning_example()
    multilingual_example()
    batch_synthesis_example()
    benchmark_example()
    
    print("\nAll examples completed!")
    print("Note: Make sure to train a model first and provide valid checkpoint paths.")


if __name__ == "__main__":
    main()