"""Main TTS evaluator that combines multiple quality metrics."""

import os
import json
import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path

from .metrics import (
    EvaluationResult,
    MOSNetEvaluator,
    ASRWordErrorRateEvaluator,
    CMVNEvaluator,
    SpectralQualityEvaluator
)

logger = logging.getLogger(__name__)


@dataclass
class TTSEvaluationReport:
    """Complete TTS evaluation report."""
    audio_path: str
    reference_text: Optional[str]
    results: Dict[str, EvaluationResult]
    overall_score: float
    evaluation_time: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'audio_path': self.audio_path,
            'reference_text': self.reference_text,
            'results': {
                name: {
                    'metric_name': result.metric_name,
                    'score': result.score,
                    'details': result.details,
                    'error': result.error
                }
                for name, result in self.results.items()
            },
            'overall_score': self.overall_score,
            'evaluation_time': self.evaluation_time
        }


class TTSEvaluator:
    """Comprehensive TTS evaluator combining multiple quality metrics."""
    
    def __init__(self, 
                 enable_mosnet: bool = True,
                 enable_asr_wer: bool = True,
                 enable_cmvn: bool = True,
                 enable_spectral: bool = True,
                 whisper_model: str = "openai/whisper-base"):
        """Initialize TTS evaluator.
        
        Args:
            enable_mosnet: Enable MOSNet-based quality evaluation
            enable_asr_wer: Enable ASR Word Error Rate evaluation
            enable_cmvn: Enable CMVN quality evaluation
            enable_spectral: Enable spectral quality evaluation
            whisper_model: Whisper model for ASR evaluation
        """
        self.logger = logging.getLogger(f"{__name__}.TTSEvaluator")
        
        # Initialize evaluators
        self.evaluators = {}
        
        if enable_mosnet:
            self.evaluators['mosnet'] = MOSNetEvaluator()
            
        if enable_asr_wer:
            try:
                self.evaluators['asr_wer'] = ASRWordErrorRateEvaluator(whisper_model)
            except ImportError as e:
                self.logger.warning(f"ASR-WER evaluator disabled: {e}")
                
        if enable_cmvn:
            self.evaluators['cmvn'] = CMVNEvaluator()
            
        if enable_spectral:
            self.evaluators['spectral'] = SpectralQualityEvaluator()
        
        self.logger.info(f"Initialized TTS evaluator with {len(self.evaluators)} metrics: {list(self.evaluators.keys())}")
    
    def evaluate_single(self, 
                       audio_path: str, 
                       reference_text: Optional[str] = None,
                       **kwargs) -> TTSEvaluationReport:
        """Evaluate a single TTS audio output.
        
        Args:
            audio_path: Path to generated audio file
            reference_text: Original text (required for WER)
            **kwargs: Additional evaluation parameters
            
        Returns:
            TTSEvaluationReport with all metric results
        """
        import time
        start_time = time.time()
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        results = {}
        
        # Run each evaluator
        for name, evaluator in self.evaluators.items():
            try:
                self.logger.debug(f"Running {name} evaluation on {audio_path}")
                result = evaluator.evaluate(audio_path, reference_text, **kwargs)
                results[name] = result
                
                if result.error:
                    self.logger.warning(f"{name} evaluation failed: {result.error}")
                else:
                    self.logger.debug(f"{name} score: {result.score:.3f}")
                    
            except Exception as e:
                self.logger.error(f"Error in {name} evaluation: {e}")
                results[name] = EvaluationResult(
                    metric_name=name,
                    score=0.0,
                    error=str(e)
                )
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(results)
        
        evaluation_time = time.time() - start_time
        
        return TTSEvaluationReport(
            audio_path=audio_path,
            reference_text=reference_text,
            results=results,
            overall_score=overall_score,
            evaluation_time=evaluation_time
        )
    
    def evaluate_batch(self, 
                      audio_files: List[str],
                      reference_texts: Optional[List[str]] = None,
                      output_file: Optional[str] = None,
                      **kwargs) -> List[TTSEvaluationReport]:
        """Evaluate multiple TTS audio outputs.
        
        Args:
            audio_files: List of audio file paths
            reference_texts: List of reference texts (same order as audio_files)
            output_file: Optional file to save results
            **kwargs: Additional evaluation parameters
            
        Returns:
            List of TTSEvaluationReport for each audio file
        """
        if reference_texts and len(reference_texts) != len(audio_files):
            raise ValueError("Number of reference texts must match number of audio files")
        
        reports = []
        
        self.logger.info(f"Starting batch evaluation of {len(audio_files)} files")
        
        for i, audio_path in enumerate(audio_files):
            ref_text = reference_texts[i] if reference_texts else None
            
            try:
                report = self.evaluate_single(audio_path, ref_text, **kwargs)
                reports.append(report)
                
                self.logger.info(f"Evaluated {i+1}/{len(audio_files)}: {audio_path} (score: {report.overall_score:.3f})")
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate {audio_path}: {e}")
                # Add failed report
                reports.append(TTSEvaluationReport(
                    audio_path=audio_path,
                    reference_text=ref_text,
                    results={},
                    overall_score=0.0,
                    evaluation_time=0.0
                ))
        
        # Save results if requested
        if output_file:
            self.save_reports(reports, output_file)
        
        self.logger.info(f"Batch evaluation completed. Average score: {self._calculate_batch_average(reports):.3f}")
        
        return reports
    
    def _calculate_overall_score(self, results: Dict[str, EvaluationResult]) -> float:
        """Calculate overall quality score from individual metrics."""
        valid_results = {name: result for name, result in results.items() 
                        if result.error is None and result.score > 0}
        
        if not valid_results:
            return 0.0
        
        # Weight different metrics
        weights = {
            'mosnet': 0.4,      # High weight for perceptual quality
            'asr_wer': 0.3,     # High weight for intelligibility (inverted)
            'cmvn': 0.15,       # Medium weight for cepstral quality
            'spectral': 0.15    # Medium weight for spectral quality
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for name, result in valid_results.items():
            weight = weights.get(name, 0.1)  # Default weight for unknown metrics
            
            # Handle score scaling and inversion
            if name == 'asr_wer':
                # WER: lower is better, so invert (1 - WER)
                normalized_score = 1.0 - min(result.score, 1.0)
            elif name == 'mosnet':
                # MOS: scale from 1-5 to 0-1
                normalized_score = (result.score - 1.0) / 4.0
            else:
                # Other metrics: assume 0-1 scale
                normalized_score = min(max(result.score, 0.0), 1.0)
            
            weighted_sum += weight * normalized_score
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _calculate_batch_average(self, reports: List[TTSEvaluationReport]) -> float:
        """Calculate average score across batch."""
        valid_scores = [r.overall_score for r in reports if r.overall_score > 0]
        return sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    
    def save_reports(self, reports: List[TTSEvaluationReport], output_file: str):
        """Save evaluation reports to JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert reports to dictionaries
        data = {
            'evaluation_summary': {
                'total_files': len(reports),
                'average_score': self._calculate_batch_average(reports),
                'metric_names': list(self.evaluators.keys())
            },
            'reports': [report.to_dict() for report in reports]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Evaluation reports saved to {output_file}")
    
    def generate_summary_report(self, reports: List[TTSEvaluationReport]) -> Dict:
        """Generate summary statistics from evaluation reports."""
        if not reports:
            return {'error': 'No reports to summarize'}
        
        # Collect scores by metric
        metric_scores = {}
        for report in reports:
            for metric_name, result in report.results.items():
                if result.error is None:
                    if metric_name not in metric_scores:
                        metric_scores[metric_name] = []
                    metric_scores[metric_name].append(result.score)
        
        # Calculate statistics
        summary = {
            'total_files': len(reports),
            'overall_average': self._calculate_batch_average(reports),
            'metric_statistics': {}
        }
        
        for metric_name, scores in metric_scores.items():
            if scores:
                import numpy as np
                summary['metric_statistics'][metric_name] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores)),
                    'count': len(scores)
                }
        
        return summary
    
    def print_summary(self, reports: List[TTSEvaluationReport]):
        """Print a formatted summary of evaluation results."""
        summary = self.generate_summary_report(reports)
        
        print(f"\n{'='*60}")
        print(f"TTS EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total files evaluated: {summary['total_files']}")
        print(f"Overall average score: {summary['overall_average']:.3f}")
        print(f"\nMetric Statistics:")
        print(f"{'='*60}")
        
        for metric_name, stats in summary.get('metric_statistics', {}).items():
            print(f"\n{metric_name.upper()}:")
            print(f"  Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
            print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
            print(f"  Count: {stats['count']}")
        
        print(f"\n{'='*60}")