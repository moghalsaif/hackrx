#!/bin/bash

# RAGFlow Insurance Policy Assistant - Unix/Linux Launcher

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                  RAGFlow Insurance Policy Assistant             ║"
echo "║                        Unix/Linux Launcher                     ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo "🏥 Starting RAGFlow Web Interface..."
echo "🌐 Powered by Streamlit and Local Llama 3"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo -e "${RED}❌ Python not found. Please install Python and try again.${NC}"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo -e "${GREEN}✅ Python found: $(${PYTHON_CMD} --version)${NC}"

# Make scripts executable
chmod +x app_launcher.py 2>/dev/null

# Launch the application
echo "🚀 Starting application..."
echo ""

${PYTHON_CMD} app_launcher.py

exit_code=$?

if [ $exit_code -ne 0 ]; then
    echo ""
    echo -e "${RED}❌ Application failed to start.${NC}"
    echo -e "${YELLOW}💡 Try running: ${PYTHON_CMD} setup_llama3.py${NC}"
    echo ""
    exit $exit_code
fi

echo ""
echo -e "${GREEN}✅ RAGFlow session completed.${NC}" 