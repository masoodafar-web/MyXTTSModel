#!/bin/bash
# Complete GPU Analysis Workflow
# This script runs all diagnostic tools in sequence

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DATA_PATH="${1:-./data}"
CONFIG_PATH="${2:-configs/config.yaml}"
OUTPUT_DIR="${3:-./gpu_analysis_results}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Complete GPU Analysis Workflow${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Data path: $DATA_PATH"
echo "Config path: $CONFIG_PATH"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo -e "${GREEN}✅ Created output directory: $OUTPUT_DIR${NC}"
echo ""

# Check if data path exists
if [ ! -d "$DATA_PATH" ]; then
    echo -e "${RED}❌ Error: Data path not found: $DATA_PATH${NC}"
    echo "Usage: $0 [data_path] [config_path] [output_dir]"
    echo "Example: $0 ./data configs/config.yaml ./results"
    exit 1
fi

# Check Python and TensorFlow
echo -e "${BLUE}[Step 0/5] Checking dependencies...${NC}"
if ! python -c "import tensorflow" 2>/dev/null; then
    echo -e "${RED}❌ TensorFlow not installed${NC}"
    echo "Please install: pip install tensorflow>=2.12.0"
    exit 1
fi
echo -e "${GREEN}✅ TensorFlow is installed${NC}"
echo ""

# Step 1: Comprehensive Diagnostic
echo -e "${BLUE}[Step 1/5] Running Comprehensive Diagnostic...${NC}"
echo "This will check all configuration and identify issues"
echo ""

python utilities/comprehensive_gpu_diagnostic.py \
    --config "$CONFIG_PATH" \
    --data-path "$DATA_PATH" \
    --output "$OUTPUT_DIR/diagnostic_report.txt"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Diagnostic complete${NC}"
    echo "Report saved to: $OUTPUT_DIR/diagnostic_report.txt"
else
    echo -e "${RED}❌ Diagnostic failed${NC}"
    exit 1
fi
echo ""

# Step 2: Quick Data Pipeline Profile
echo -e "${BLUE}[Step 2/5] Profiling Data Pipeline (Quick Test)...${NC}"
echo "Testing 50 batches with current settings"
echo ""

python utilities/enhanced_gpu_profiler.py \
    --config "$CONFIG_PATH" \
    --data-path "$DATA_PATH" \
    --batch-size 16 \
    --num-batches 50 \
    --output "$OUTPUT_DIR/quick_profile.txt"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Quick profile complete${NC}"
    echo "Report saved to: $OUTPUT_DIR/quick_profile.txt"
else
    echo -e "${YELLOW}⚠️  Quick profile had issues (may be normal)${NC}"
fi
echo ""

# Step 3: Benchmark (Optional - takes longer)
echo -e "${BLUE}[Step 3/5] Running Benchmark (Optional)...${NC}"
echo "This will test multiple configurations to find optimal settings"
read -p "Run full benchmark? This may take 5-10 minutes [y/N]: " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    python utilities/enhanced_gpu_profiler.py \
        --config "$CONFIG_PATH" \
        --data-path "$DATA_PATH" \
        --benchmark \
        --output "$OUTPUT_DIR/benchmark_results.txt"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Benchmark complete${NC}"
        echo "Report saved to: $OUTPUT_DIR/benchmark_results.txt"
    else
        echo -e "${YELLOW}⚠️  Benchmark had issues${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Benchmark skipped${NC}"
fi
echo ""

# Step 4: Training Step Profile
echo -e "${BLUE}[Step 4/5] Profiling Training Steps...${NC}"
echo "This profiles the complete training loop"
echo ""

python utilities/training_step_profiler.py \
    --config "$CONFIG_PATH" \
    --data-path "$DATA_PATH" \
    --num-steps 50 \
    --batch-size 16

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Training step profile complete${NC}"
else
    echo -e "${YELLOW}⚠️  Training step profile had issues${NC}"
fi
echo ""

# Step 5: Generate Summary
echo -e "${BLUE}[Step 5/5] Generating Summary...${NC}"
echo ""

SUMMARY_FILE="$OUTPUT_DIR/ANALYSIS_SUMMARY.txt"

cat > "$SUMMARY_FILE" << EOF
========================================
GPU ANALYSIS SUMMARY
========================================
Generated: $(date)
Data path: $DATA_PATH
Config path: $CONFIG_PATH

========================================
FILES GENERATED
========================================
1. diagnostic_report.txt - Comprehensive diagnostic
2. quick_profile.txt - Quick data pipeline profile
3. benchmark_results.txt - Benchmark results (if run)
4. ANALYSIS_SUMMARY.txt - This file

========================================
NEXT STEPS
========================================

1. Review diagnostic_report.txt for issues and recommendations

2. Apply recommended configuration changes to $CONFIG_PATH

3. If needed, review benchmark_results.txt for optimal settings

4. Re-run diagnostic to verify fixes:
   python utilities/comprehensive_gpu_diagnostic.py --data-path $DATA_PATH

5. Start training:
   python train_main.py --train-data $DATA_PATH --batch-size <optimal>

6. Monitor GPU during training:
   watch -n 0.5 nvidia-smi

========================================
DOCUMENTATION
========================================
- docs/COMPREHENSIVE_GPU_BOTTLENECK_ANALYSIS.md - Full guide
- USAGE_GUIDE_GPU_PROFILING.md - Usage examples
- GPU_OSCILLATION_SOLUTION_SUMMARY.md - Solution summary

========================================
EOF

echo -e "${GREEN}✅ Summary generated: $SUMMARY_FILE${NC}"
echo ""

# Print summary
cat "$SUMMARY_FILE"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Analysis Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "All results saved to: $OUTPUT_DIR"
echo ""
echo "Next: Review $OUTPUT_DIR/diagnostic_report.txt for recommendations"
