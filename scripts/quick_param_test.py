#!/usr/bin/env python3
"""
🚀 Quick MyXTTS Parameter Test

A simple script to quickly test different parameter combinations
and find what works best for your setup.

Usage:
    python3 quick_param_test.py                    # Test common scenarios
    python3 quick_param_test.py --plateau-fix      # Focus on plateau issues
    python3 quick_param_test.py --gpu-optimize     # Focus on GPU utilization
    python3 quick_param_test.py --memory-safe      # For limited memory
"""

import os
import sys
import time
import subprocess
import argparse
import json
from datetime import datetime
from typing import List, Dict, Tuple

# Test scenarios
SCENARIOS = {
    'basic_test': [
        {'name': 'Tiny Basic', 'args': ['--model-size', 'tiny', '--optimization-level', 'basic', '--batch-size', '4', '--epochs', '3']},
        {'name': 'Tiny Enhanced', 'args': ['--model-size', 'tiny', '--optimization-level', 'enhanced', '--batch-size', '8', '--epochs', '3']},
        {'name': 'Small Enhanced', 'args': ['--model-size', 'small', '--optimization-level', 'enhanced', '--batch-size', '16', '--epochs', '3']},
    ],
    
    'plateau_fix': [
        {'name': 'Plateau Breaker Low LR', 'args': ['--model-size', 'tiny', '--optimization-level', 'plateau_breaker', '--batch-size', '8', '--lr', '1.5e-5', '--epochs', '5']},
        {'name': 'Plateau Breaker Med LR', 'args': ['--model-size', 'tiny', '--optimization-level', 'plateau_breaker', '--batch-size', '16', '--lr', '2e-5', '--epochs', '5']},
        {'name': 'Enhanced vs Plateau', 'args': ['--model-size', 'small', '--optimization-level', 'enhanced', '--batch-size', '24', '--lr', '3e-5', '--epochs', '5']},
    ],
    
    'gpu_optimize': [
        {'name': 'Standard Training', 'args': ['--model-size', 'small', '--optimization-level', 'enhanced', '--batch-size', '32', '--epochs', '4']},
        {'name': 'Large Batch', 'args': ['--model-size', 'normal', '--optimization-level', 'enhanced', '--batch-size', '48', '--epochs', '3']},
    ],
    
    'memory_safe': [
        {'name': 'Ultra Safe', 'args': ['--model-size', 'tiny', '--optimization-level', 'basic', '--batch-size', '2', '--epochs', '3']},
        {'name': 'Safe Enhanced', 'args': ['--model-size', 'tiny', '--optimization-level', 'enhanced', '--batch-size', '4', '--epochs', '3']},
        {'name': 'Small Safe', 'args': ['--model-size', 'small', '--optimization-level', 'basic', '--batch-size', '8', '--epochs', '3']},
    ],
    
    'learning_rate_sweep': [
        {'name': 'Very Low LR', 'args': ['--model-size', 'tiny', '--optimization-level', 'enhanced', '--lr', '5e-6', '--batch-size', '8', '--epochs', '4']},
        {'name': 'Low LR', 'args': ['--model-size', 'tiny', '--optimization-level', 'enhanced', '--lr', '1e-5', '--batch-size', '8', '--epochs', '4']},
        {'name': 'Medium LR', 'args': ['--model-size', 'tiny', '--optimization-level', 'enhanced', '--lr', '2e-5', '--batch-size', '8', '--epochs', '4']},
        {'name': 'High LR', 'args': ['--model-size', 'tiny', '--optimization-level', 'enhanced', '--lr', '5e-5', '--batch-size', '8', '--epochs', '4']},
        {'name': 'Very High LR', 'args': ['--model-size', 'tiny', '--optimization-level', 'enhanced', '--lr', '1e-4', '--batch-size', '8', '--epochs', '4']},
    ]
}

def run_single_test(test_config: Dict, timeout: int = 300) -> Dict:
    """Run a single parameter test"""
    
    print(f"\n🔬 Testing: {test_config['name']}")
    print(f"   Command: python3 train_main.py {' '.join(test_config['args'])}")
    
    # Add common arguments
    cmd = ['python3', 'train_main.py'] + test_config['args']
    cmd.extend(['--checkpoint-dir', f'./temp_test_{int(time.time())}'])
    
    start_time = time.time()
    final_loss = None
    gpu_usage = "N/A"
    error = None
    completed = False
    
    try:
        # Run training
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            universal_newlines=True
        )
        
        # Monitor output
        output_lines = []
        while True:
            line = process.stdout.readline()
            if not line:
                break
            
            output_lines.append(line)
            print(f"   {line.strip()}")
            
            # Extract metrics
            if "Train Loss:" in line:
                try:
                    final_loss = float(line.split("Train Loss:")[1].split()[0])
                except:
                    pass
            
            if "GPU" in line and "%" in line:
                try:
                    # Extract GPU usage if available
                    if "GPU=" in line:
                        gpu_usage = line.split("GPU=")[1].split("%")[0] + "%"
                except:
                    pass
            
            # Check for timeout
            if time.time() - start_time > timeout:
                process.terminate()
                print(f"   ⏰ Test timed out after {timeout} seconds")
                break
        
        # Wait for completion
        process.wait(timeout=10)
        completed = process.returncode == 0
        
    except Exception as e:
        error = str(e)
        print(f"   ❌ Test failed: {error}")
    
    duration = time.time() - start_time
    
    result = {
        'name': test_config['name'],
        'command': ' '.join(cmd),
        'duration': duration,
        'final_loss': final_loss,
        'gpu_usage': gpu_usage,
        'completed': completed,
        'error': error,
        'timestamp': datetime.now().isoformat()
    }
    
    # Print result summary
    status = "✅ COMPLETED" if completed else "❌ FAILED"
    print(f"   {status} in {duration:.1f}s")
    if final_loss is not None:
        print(f"   📊 Final Loss: {final_loss:.3f}")
    if gpu_usage != "N/A":
        print(f"   🎮 GPU Usage: {gpu_usage}")
    
    return result

def run_scenario(scenario_name: str, scenarios: Dict) -> List[Dict]:
    """Run all tests in a scenario"""
    
    if scenario_name not in scenarios:
        print(f"❌ Unknown scenario: {scenario_name}")
        print(f"Available scenarios: {list(scenarios.keys())}")
        return []
    
    tests = scenarios[scenario_name]
    print(f"\n🚀 Running scenario: {scenario_name} ({len(tests)} tests)")
    print("=" * 60)
    
    results = []
    for i, test in enumerate(tests, 1):
        print(f"\n[{i}/{len(tests)}] Running: {test['name']}")
        result = run_single_test(test)
        results.append(result)
        
        # Brief pause between tests
        if i < len(tests):
            time.sleep(2)
    
    return results

def analyze_results(results: List[Dict]):
    """Analyze and report results"""
    
    if not results:
        print("No results to analyze!")
        return
    
    print("\n" + "=" * 60)
    print("📋 RESULTS SUMMARY")
    print("=" * 60)
    
    # Sort by final loss (lower is better)
    successful_results = [r for r in results if r['completed'] and r['final_loss'] is not None]
    if successful_results:
        successful_results.sort(key=lambda x: x['final_loss'])
        
        print(f"\n🏆 BEST RESULTS (by final loss):")
        for i, result in enumerate(successful_results[:3], 1):
            print(f"{i}. {result['name']}: Loss={result['final_loss']:.3f}, Time={result['duration']:.1f}s")
        
        print(f"\n📊 PERFORMANCE COMPARISON:")
        for result in successful_results:
            stars = "⭐" * min(5, int((5 - result['final_loss']) + 1))
            print(f"{result['name']:25} | Loss: {result['final_loss']:.3f} | Time: {result['duration']:5.1f}s | {stars}")
    
    # Check for failures
    failed_results = [r for r in results if not r['completed']]
    if failed_results:
        print(f"\n❌ FAILED TESTS ({len(failed_results)}):")
        for result in failed_results:
            print(f"   - {result['name']}: {result['error'] or 'Unknown error'}")
    
    # Generate recommendation
    if successful_results:
        best = successful_results[0]
        print(f"\n🎯 RECOMMENDATION:")
        print(f"Best configuration: {best['name']}")
        print(f"Command to use: {best['command'].replace('temp_test_', 'checkpointsmain')}")
        
        # Extract key parameters
        cmd_parts = best['command'].split()
        recommendations = []
        
        for i, part in enumerate(cmd_parts):
            if part in ['--model-size', '--optimization-level', '--batch-size', '--lr']:
                if i + 1 < len(cmd_parts):
                    recommendations.append(f"{part} {cmd_parts[i + 1]}")
        
        if recommendations:
            print(f"Key parameters: {', '.join(recommendations)}")

def save_results(results: List[Dict], scenario_name: str):
    """Save results to file"""
    
    filename = f"param_test_results_{scenario_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w') as f:
        json.dump({
            'scenario': scenario_name,
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Quick MyXTTS Parameter Testing")
    parser.add_argument('--scenario', choices=list(SCENARIOS.keys()), 
                       help='Test scenario to run')
    parser.add_argument('--plateau-fix', action='store_true', 
                       help='Focus on plateau breakthrough testing')
    parser.add_argument('--gpu-optimize', action='store_true', 
                       help='Focus on GPU optimization testing')
    parser.add_argument('--memory-safe', action='store_true', 
                       help='Focus on memory-safe configurations')
    parser.add_argument('--learning-rate-sweep', action='store_true', 
                       help='Test different learning rates')
    parser.add_argument('--all', action='store_true', 
                       help='Run all scenarios (will take a while!)')
    parser.add_argument('--timeout', type=int, default=300, 
                       help='Timeout per test in seconds (default: 300)')
    
    args = parser.parse_args()
    
    # Determine which scenario to run
    if args.scenario:
        scenario_name = args.scenario
    elif args.plateau_fix:
        scenario_name = 'plateau_fix'
    elif args.gpu_optimize:
        scenario_name = 'gpu_optimize'
    elif args.memory_safe:
        scenario_name = 'memory_safe'
    elif args.learning_rate_sweep:
        scenario_name = 'learning_rate_sweep'
    elif args.all:
        scenario_name = 'all'
    else:
        scenario_name = 'basic_test'
    
    print(f"🔬 MyXTTS Quick Parameter Test")
    print(f"Scenario: {scenario_name}")
    print(f"Timeout per test: {args.timeout} seconds")
    
    if scenario_name == 'all':
        # Run all scenarios
        all_results = []
        for name in SCENARIOS.keys():
            print(f"\n{'='*20} {name.upper()} {'='*20}")
            results = run_scenario(name, SCENARIOS)
            all_results.extend(results)
            analyze_results(results)
            save_results(results, name)
        
        print(f"\n{'='*20} OVERALL ANALYSIS {'='*20}")
        analyze_results(all_results)
        save_results(all_results, 'all_scenarios')
    else:
        # Run specific scenario
        results = run_scenario(scenario_name, SCENARIOS)
        analyze_results(results)
        save_results(results, scenario_name)
    
    print(f"\n🎉 Testing complete!")
    print(f"💡 Use the recommended parameters for your full training run.")

if __name__ == "__main__":
    main()