#!/usr/bin/env python3
"""Standalone TTS evaluation script for automatic quality assessment.

This script provides comprehensive evaluation of TTS outputs using multiple metrics:
- MOSNet-based perceptual quality scoring
- ASR Word Error Rate evaluation using Whisper
- CMVN (Cepstral Mean and Variance Normalization) analysis
- Spectral quality assessment

Usage Examples:
    # Evaluate single audio file
    python evaluate_tts.py --audio output.wav --text "Hello world"
    
    # Evaluate multiple files with batch processing
    python evaluate_tts.py --audio-dir outputs/ --text-file texts.txt
    
    # Save detailed results
    python evaluate_tts.py --audio output.wav --text "Hello world" --output report.json
    
    # Use specific metrics only
    python evaluate_tts.py --audio output.wav --text "Hello world" --metrics mosnet,spectral
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional

# Add myxtts to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent))

from myxtts.evaluation import TTSEvaluator


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Automatic TTS quality evaluation using multiple metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--audio", "-a",
        help="Single audio file to evaluate"
    )
    input_group.add_argument(
        "--audio-dir", "-d",
        help="Directory containing audio files to evaluate"
    )
    input_group.add_argument(
        "--audio-list",
        help="Text file with list of audio file paths"
    )
    
    # Text options
    text_group = parser.add_mutually_exclusive_group()
    text_group.add_argument(
        "--text", "-t",
        help="Reference text for single audio file"
    )
    text_group.add_argument(
        "--text-file",
        help="Text file with reference texts (one per line, matching audio order)"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        help="Output file for detailed results (JSON format)"
    )
    
    # Evaluation options
    parser.add_argument(
        "--metrics",
        default="mosnet,asr_wer,cmvn,spectral",
        help="Comma-separated list of metrics to use (default: all)"
    )
    parser.add_argument(
        "--whisper-model",
        default="openai/whisper-base",
        help="Whisper model for ASR evaluation (default: openai/whisper-base)"
    )
    
    # Processing options
    parser.add_argument(
        "--audio-extensions",
        default="wav,mp3,flac",
        help="Audio file extensions to process (default: wav,mp3,flac)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def load_audio_files(args: argparse.Namespace) -> List[str]:
    """Load list of audio files based on arguments."""
    audio_files = []
    
    if args.audio:
        # Single file
        if not os.path.exists(args.audio):
            raise FileNotFoundError(f"Audio file not found: {args.audio}")
        audio_files = [args.audio]
        
    elif args.audio_dir:
        # Directory
        if not os.path.isdir(args.audio_dir):
            raise NotADirectoryError(f"Directory not found: {args.audio_dir}")
        
        extensions = args.audio_extensions.split(',')
        for ext in extensions:
            pattern = f"*.{ext.strip()}"
            audio_files.extend(Path(args.audio_dir).glob(pattern))
        
        audio_files = [str(f) for f in sorted(audio_files)]
        
    elif args.audio_list:
        # File with list
        if not os.path.exists(args.audio_list):
            raise FileNotFoundError(f"Audio list file not found: {args.audio_list}")
        
        with open(args.audio_list, 'r', encoding='utf-8') as f:
            audio_files = [line.strip() for line in f if line.strip()]
    
    if not audio_files:
        raise ValueError("No audio files found to evaluate")
    
    return audio_files


def load_reference_texts(args: argparse.Namespace, num_files: int) -> Optional[List[str]]:
    """Load reference texts if provided."""
    if args.text:
        # Single text for single file
        if num_files != 1:
            raise ValueError("Single text provided but multiple audio files found")
        return [args.text]
        
    elif args.text_file:
        # Text file
        if not os.path.exists(args.text_file):
            raise FileNotFoundError(f"Text file not found: {args.text_file}")
        
        with open(args.text_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        if len(texts) != num_files:
            raise ValueError(f"Number of texts ({len(texts)}) doesn't match number of audio files ({num_files})")
        
        return texts
    
    return None


def create_evaluator(metrics: str, whisper_model: str) -> TTSEvaluator:
    """Create TTS evaluator with specified metrics."""
    metric_list = [m.strip().lower() for m in metrics.split(',')]
    
    return TTSEvaluator(
        enable_mosnet='mosnet' in metric_list,
        enable_asr_wer='asr_wer' in metric_list,
        enable_cmvn='cmvn' in metric_list,
        enable_spectral='spectral' in metric_list,
        whisper_model=whisper_model
    )


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level)
    
    try:
        # Load audio files
        logger.info("Loading audio files...")
        audio_files = load_audio_files(args)
        logger.info(f"Found {len(audio_files)} audio files")
        
        # Load reference texts
        reference_texts = load_reference_texts(args, len(audio_files))
        if reference_texts:
            logger.info(f"Loaded {len(reference_texts)} reference texts")
        else:
            logger.warning("No reference texts provided - ASR-WER evaluation will be skipped")
        
        # Create evaluator
        logger.info(f"Initializing evaluator with metrics: {args.metrics}")
        evaluator = create_evaluator(args.metrics, args.whisper_model)
        
        # Run evaluation
        if len(audio_files) == 1:
            # Single file evaluation
            logger.info(f"Evaluating: {audio_files[0]}")
            ref_text = reference_texts[0] if reference_texts else None
            report = evaluator.evaluate_single(audio_files[0], ref_text)
            
            # Print results
            print(f"\nEvaluation Results for: {report.audio_path}")
            print(f"Overall Score: {report.overall_score:.3f}")
            print(f"Evaluation Time: {report.evaluation_time:.2f}s")
            
            for metric_name, result in report.results.items():
                if result.error:
                    print(f"{metric_name.upper()}: ERROR - {result.error}")
                else:
                    print(f"{metric_name.upper()}: {result.score:.3f}")
            
            # Save detailed results
            if args.output:
                evaluator.save_reports([report], args.output)
                logger.info(f"Detailed results saved to: {args.output}")
        
        else:
            # Batch evaluation
            logger.info(f"Starting batch evaluation of {len(audio_files)} files")
            reports = evaluator.evaluate_batch(
                audio_files, 
                reference_texts, 
                args.output
            )
            
            # Print summary
            evaluator.print_summary(reports)
            
            if args.output:
                logger.info(f"Detailed results saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()