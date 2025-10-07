#!/bin/bash
# MyXTTS Project Management Script
# مدیریت آسان پروژه MyXTTS

echo "🎯 MyXTTS Project Manager"
echo "========================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

function show_status() {
    echo -e "${BLUE}📊 Project Status:${NC}"
    echo "==================="
    
    # Check if training is running
    if pgrep -f "train_main.py" > /dev/null; then
        echo -e "${GREEN}✅ Training: RUNNING${NC}"
        # Get latest training progress
        tail -n 5 logs/run_*.log 2>/dev/null | grep -E "(loss=|INFO)" | tail -3
    else
        echo -e "${RED}❌ Training: STOPPED${NC}"
    fi
    
    # Check checkpoints
    if [ -d "checkpointsmain" ] && [ "$(ls -A checkpointsmain/checkpoint_* 2>/dev/null)" ]; then
        LATEST_CHECKPOINT=$(ls -t checkpointsmain/checkpoint_*_metadata.json 2>/dev/null | head -1 | sed 's/_metadata.json//')
        if [ -n "$LATEST_CHECKPOINT" ]; then
            echo -e "${GREEN}✅ Latest checkpoint: $(basename $LATEST_CHECKPOINT)${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️ No checkpoints found${NC}"
    fi
    
    # Check GPU usage
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${BLUE}🖥️ GPU Status:${NC}"
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -2
    fi
    
    echo ""
}

function start_training() {
    echo -e "${BLUE}🚀 Starting Training...${NC}"
    
    if pgrep -f "train_main.py" > /dev/null; then
        echo -e "${RED}❌ Training already running!${NC}"
        return 1
    fi
    
    # Default training command
    CMD="python3 train_main.py --model-size normal --optimization-level enhanced --batch-size 32"
    
    echo "Command: $CMD"
    echo "Starting in 3 seconds... (Ctrl+C to cancel)"
    sleep 3
    
    nohup $CMD > training_output.log 2>&1 &
    echo -e "${GREEN}✅ Training started! Check training_output.log for progress${NC}"
}

function stop_training() {
    echo -e "${YELLOW}⏹️ Stopping Training...${NC}"
    
    if pgrep -f "train_main.py" > /dev/null; then
        pkill -f "train_main.py"
        echo -e "${GREEN}✅ Training stopped${NC}"
    else
        echo -e "${RED}❌ No training process found${NC}"
    fi
}

function test_model() {
    echo -e "${BLUE}🧪 Testing Model...${NC}"
    
    if [ ! -d "checkpointsmain" ] || [ ! "$(ls -A checkpointsmain/checkpoint_* 2>/dev/null)" ]; then
        echo -e "${RED}❌ No checkpoints found for testing${NC}"
        return 1
    fi
    
    echo "Running inference test..."
    python3 fixed_inference.py --text "Hello world, this is a test" --output test_output.wav --speaker-audio speaker.wav
    
    if [ -f "test_output.wav" ]; then
        echo -e "${GREEN}✅ Test completed! Output: test_output.wav${NC}"
        mv test_output.wav outputs/audio_samples/
    else
        echo -e "${RED}❌ Test failed${NC}"
    fi
}

function analyze_quality() {
    echo -e "${BLUE}📊 Analyzing Model Quality...${NC}"
    echo -e "${YELLOW}⚠️  Quality analysis tools have been removed for project simplification${NC}"
    echo -e "${BLUE}💡 You can use utilities/evaluate_tts.py for basic quality evaluation${NC}"
}

function clean_project() {
    echo -e "${YELLOW}🧹 Cleaning Project...${NC}"
    
    # Clean up temporary files
    find . -name "*.pyc" -delete 2>/dev/null
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
    
    # Clean old logs (keep last 5)
    if [ -d "logs" ]; then
        cd logs
        ls -t run_*.log 2>/dev/null | tail -n +6 | xargs rm -f 2>/dev/null
        cd ..
    fi
    
    # Clean temporary audio files in root
    rm -f *.wav 2>/dev/null || true
    rm -f *.png 2>/dev/null || true
    rm -f *.npy 2>/dev/null || true
    
    echo -e "${GREEN}✅ Project cleaned${NC}"
}

function backup_checkpoints() {
    echo -e "${BLUE}💾 Backing up Checkpoints...${NC}"
    
    if [ ! -d "checkpointsmain" ]; then
        echo -e "${RED}❌ No checkpoints to backup${NC}"
        return 1
    fi
    
    BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    cp -r checkpointsmain "$BACKUP_DIR/"
    
    echo -e "${GREEN}✅ Checkpoints backed up to: $BACKUP_DIR${NC}"
}

function show_menu() {
    echo -e "${BLUE}Available Commands:${NC}"
    echo "1. status    - Show project status"
    echo "2. train     - Start training"
    echo "3. stop      - Stop training"
    echo "4. test      - Test model inference"
    echo "5. analyze   - Analyze model quality"
    echo "6. clean     - Clean temporary files"
    echo "7. backup    - Backup checkpoints"
    echo "8. help      - Show this menu"
    echo ""
}

# Main logic
case "$1" in
    "status"|"s")
        show_status
        ;;
    "train"|"t")
        start_training
        ;;
    "stop")
        stop_training
        ;;
    "test")
        test_model
        ;;
    "analyze"|"a")
        analyze_quality
        ;;
    "clean"|"c")
        clean_project
        ;;
    "backup"|"b")
        backup_checkpoints
        ;;
    "help"|"h"|"")
        show_menu
        ;;
    *)
        echo -e "${RED}❌ Unknown command: $1${NC}"
        show_menu
        ;;
esac