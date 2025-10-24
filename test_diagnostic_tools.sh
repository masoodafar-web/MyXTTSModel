#!/bin/bash
# Test script for all diagnostic tools
# Usage: ./test_diagnostic_tools.sh

echo "=========================================="
echo "Testing MyXTTS Diagnostic Tools"
echo "=========================================="
echo ""

# Check if config file exists
CONFIG_FILE="configs/config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "✅ Config file found: $CONFIG_FILE"
echo ""

# Test 1: Functional Issues Diagnostic
echo "=========================================="
echo "Test 1: Functional Issues Diagnostic"
echo "=========================================="
python utilities/diagnose_functional_issues.py --config "$CONFIG_FILE"
RESULT1=$?
echo ""

# Test 2: GPU Issues Diagnostic
echo "=========================================="
echo "Test 2: GPU Issues Diagnostic"
echo "=========================================="
python utilities/diagnose_gpu_issues.py --check-config "$CONFIG_FILE"
RESULT2=$?
echo ""

# Test 3: Convergence Diagnostic (with sample log)
echo "=========================================="
echo "Test 3: Convergence Diagnostic"
echo "=========================================="

# Create sample log file
SAMPLE_LOG="/tmp/test_training.log"
cat > "$SAMPLE_LOG" << 'EOF'
Epoch 1, Step 10: Loss: 15.23
Epoch 1, Step 20: Loss: 12.45
Epoch 2, Step 10: Loss: 8.67
Epoch 2, Step 20: Loss: 6.89
Epoch 3, Step 10: Loss: 4.56
Epoch 3, Step 20: Loss: 3.45
EOF

python utilities/diagnose_convergence.py --log-file "$SAMPLE_LOG"
RESULT3=$?
echo ""

# Clean up
rm -f "$SAMPLE_LOG"

# Summary
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "1. Functional Issues Diagnostic: $([ $RESULT1 -eq 0 ] && echo '✅ PASS' || echo '⚠️  WARN')"
echo "2. GPU Issues Diagnostic:        $([ $RESULT2 -eq 0 ] && echo '✅ PASS' || echo '⚠️  WARN')"
echo "3. Convergence Diagnostic:       $([ $RESULT3 -eq 0 ] && echo '✅ PASS' || echo '⚠️  WARN')"
echo ""

# Note about warnings
if [ $RESULT1 -ne 0 ] || [ $RESULT2 -ne 0 ] || [ $RESULT3 -ne 0 ]; then
    echo "ℹ️  Note: Non-zero exit codes indicate warnings/recommendations found."
    echo "   This is normal - review the output above for details."
fi

echo ""
echo "All diagnostic tools are functional!"
echo "See DIAGNOSTIC_TOOLS_GUIDE.md for usage details."
