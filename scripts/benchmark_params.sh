#!/bin/bash
# 🔬 MyXTTS Parameter Benchmark Script

echo "🚀 MyXTTS Parameter Benchmark Suite"
echo "===================================="

# Check if Python script exists
if [ ! -f "scripts/quick_param_test.py" ]; then
    echo "❌ Error: quick_param_test.py not found!"
    echo "Please run this script from the MyXTTS project root directory."
    exit 1
fi

# Make scripts executable
chmod +x scripts/quick_param_test.py
chmod +x utilities/benchmark_hyperparameters.py

# Show menu
echo ""
echo "Select benchmark type:"
echo "1. Quick Test (5 minutes) - Basic parameter testing"
echo "2. Plateau Fix (10 minutes) - Focus on loss plateau issues" 
echo "3. GPU Optimization (15 minutes) - Focus on GPU utilization"
echo "4. Memory Safe (8 minutes) - For limited VRAM setups"
echo "5. Learning Rate Sweep (20 minutes) - Test different learning rates"
echo "6. Full Benchmark (2+ hours) - Comprehensive hyperparameter search"
echo "7. Custom Quick Tests - Run specific scenarios"
echo ""

read -p "Enter your choice (1-7): " choice

case $choice in
    1)
        echo "🔬 Starting Quick Parameter Test..."
        python3 scripts/quick_param_test.py
        ;;
    2)
        echo "🎯 Starting Plateau Fix Testing..."
        python3 scripts/quick_param_test.py --plateau-fix
        ;;
    3)
        echo "🎮 Starting GPU Optimization Testing..."
        python3 scripts/quick_param_test.py --gpu-optimize
        ;;
    4)
        echo "💾 Starting Memory Safe Testing..."
        python3 scripts/quick_param_test.py --memory-safe
        ;;
    5)
        echo "📈 Starting Learning Rate Sweep..."
        python3 scripts/quick_param_test.py --learning-rate-sweep
        ;;
    6)
        echo "🔬 Starting Full Hyperparameter Benchmark..."
        echo "⚠️  This will take 2+ hours and test many combinations!"
        read -p "Are you sure? (y/N): " confirm
        if [[ $confirm =~ ^[Yy]$ ]]; then
            python3 utilities/benchmark_hyperparameters.py --full-sweep
        else
            echo "Cancelled."
        fi
        ;;
    7)
        echo "🛠️ Custom Quick Tests Available:"
        echo ""
        echo "Basic scenarios:"
        echo "  python3 scripts/quick_param_test.py --scenario basic_test"
        echo ""
        echo "Problem-specific:"
        echo "  python3 scripts/quick_param_test.py --plateau-fix"
        echo "  python3 scripts/quick_param_test.py --gpu-optimize"
        echo "  python3 scripts/quick_param_test.py --memory-safe"
        echo ""
        echo "Advanced:"
        echo "  python3 utilities/benchmark_hyperparameters.py --quick-test"
        echo "  python3 utilities/benchmark_hyperparameters.py --config configs/benchmark_config.yaml"
        ;;
    *)
        echo "Invalid choice. Running basic test..."
        python3 scripts/quick_param_test.py
        ;;
esac

echo ""
echo "✅ Benchmark completed!"
echo ""
echo "📋 Next steps:"
echo "1. Check the results and recommendations above"
echo "2. Use the recommended parameters for your training"
echo "3. Check generated JSON files for detailed analysis"
echo ""
echo "💡 Quick commands based on common results:"
echo ""
echo "For stuck loss around 2.5:"
echo "  python3 train_main.py --optimization-level plateau_breaker --batch-size 24"
echo ""
echo "For optimized training:"
echo "  python3 train_main.py --optimization-level enhanced"
echo ""
echo "For memory-constrained systems:"
echo "  python3 train_main.py --model-size tiny --batch-size 4 --optimization-level basic"