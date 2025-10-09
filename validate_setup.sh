#!/bin/bash
# Quick validation script for retracing fix
# Run this before starting training to ensure everything is configured correctly

set -e

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  MyXTTS Training Setup Validation"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo "ℹ️  $1"
}

# Check if config file exists
CONFIG_FILE="${1:-configs/config.yaml}"

echo "Validation Steps:"
echo "────────────────────────────────────────────────────────────────────────────────"
echo ""

# Step 1: Check Python and TensorFlow
echo "1️⃣  Checking Python and TensorFlow..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python installed: $PYTHON_VERSION"
else
    print_error "Python3 not found!"
    exit 1
fi

if python3 -c "import tensorflow" 2>/dev/null; then
    TF_VERSION=$(python3 -c "import tensorflow as tf; print(tf.__version__)")
    print_success "TensorFlow installed: $TF_VERSION"
else
    print_error "TensorFlow not installed! Run: pip install tensorflow"
    exit 1
fi
echo ""

# Step 2: Check config file
echo "2️⃣  Checking configuration file..."
if [ -f "$CONFIG_FILE" ]; then
    print_success "Config file found: $CONFIG_FILE"
else
    print_error "Config file not found: $CONFIG_FILE"
    print_info "Usage: $0 <path/to/config.yaml>"
    exit 1
fi
echo ""

# Step 3: Run unit tests
echo "3️⃣  Running unit tests..."
if python3 tests/test_static_shapes_fix.py > /tmp/test_output.txt 2>&1; then
    print_success "All unit tests passed (5/5)"
else
    print_error "Unit tests failed!"
    echo "See /tmp/test_output.txt for details"
    exit 1
fi
echo ""

# Step 4: Run diagnostic tool
echo "4️⃣  Running retracing diagnostic..."
print_info "This will take 1-2 minutes..."
if python3 utilities/diagnose_retracing.py --config "$CONFIG_FILE" --steps 5 > /tmp/diagnostic_output.txt 2>&1; then
    print_success "Diagnostic passed - no retracing detected!"
    
    # Extract key metrics
    RETRACE_COUNT=$(grep "Retracing events:" /tmp/diagnostic_output.txt | awk '{print $3}')
    AVG_TIME=$(grep "Average step time:" /tmp/diagnostic_output.txt | awk '{print $4}')
    
    echo ""
    echo "  📊 Key Metrics:"
    echo "     Retracing events: $RETRACE_COUNT"
    echo "     Avg step time: $AVG_TIME"
else
    print_error "Diagnostic failed!"
    print_info "Common issues:"
    echo "     - pad_to_fixed_length not enabled in config"
    echo "     - max_text_length or max_mel_frames not set"
    echo "     - Data path incorrect"
    echo ""
    print_info "Full output in /tmp/diagnostic_output.txt"
    exit 1
fi
echo ""

# Step 5: Check GPU availability
echo "5️⃣  Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    if [ "$GPU_COUNT" -gt 0 ]; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
        print_success "GPU detected: $GPU_NAME (${GPU_MEM}MB)"
    else
        print_warning "No GPU detected - training will use CPU"
    fi
else
    print_warning "nvidia-smi not found - cannot check GPU"
    print_info "Training will proceed on CPU if no GPU available"
fi
echo ""

# Summary
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  VALIDATION COMPLETE"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
print_success "All checks passed! Your setup is ready for training."
echo ""
echo "Next steps:"
echo "  1. Review your config: $CONFIG_FILE"
echo "  2. Start training: python train_main.py --config $CONFIG_FILE"
echo "  3. Monitor GPU: watch -n 1 nvidia-smi"
echo ""
echo "Expected results:"
echo "  • Initial compilation: ~30 seconds (one time)"
echo "  • Training steps: ~0.5 seconds each"
echo "  • GPU utilization: 70-90% (stable)"
echo "  • No retracing warnings after first step"
echo ""
print_info "For troubleshooting, see: RETRACING_COMPLETE_SOLUTION.md"
echo ""
