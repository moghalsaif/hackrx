# 🏥 RAGFlow Insurance Policy Assistant

<div align="center">

![RAGFlow Logo](https://img.shields.io/badge/RAGFlow-Insurance%20Assistant-blue?style=for-the-badge&logo=robot)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Interface-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Llama 3](https://img.shields.io/badge/Llama%203-Local%20AI-00D2FF?style=for-the-badge&logo=meta&logoColor=white)](https://llama.meta.com)

**Advanced Retrieval-Augmented Generation System for Insurance Policy Analysis**

*Ask natural language questions about insurance policies and get precise, well-referenced answers with exact clause citations*

[🚀 Quick Start](#-quick-start) • [📖 Documentation](#-documentation) • [🎯 Features](#-features) • [🛠️ Setup](#%EF%B8%8F-installation--setup)

</div>

---

## 🎯 **What is RAGFlow?**

RAGFlow is a sophisticated **Retrieval-Augmented Generation (RAG)** system specifically designed for insurance policy document analysis. It processes complex policy documents and enables users to ask natural language questions, receiving precise answers with exact clause references and page numbers for full auditability.

### **🌟 Key Highlights**
- 🦙 **100% Local AI** - Powered by Llama 3, no data sent to external services
- 🔍 **Semantic Search** - Advanced NLP understanding of insurance terminology
- 📋 **Exact Citations** - Every answer includes clause numbers, sections, and page references
- 🌐 **Beautiful Web Interface** - Professional Streamlit UI with real-time monitoring
- ⚡ **High Performance** - Fast query processing (< 1 second response time)
- 🔒 **Enterprise Ready** - Complete audit trails and compliance features

---

## 🎯 **Features**

### 📄 **Intelligent Document Processing**
- **Multi-format Support**: PDF processing with OCR fallback
- **Smart Chunking**: Preserves clause boundaries and insurance policy structure
- **Metadata Extraction**: Automatic section, clause, and page number detection
- **Content Validation**: Ensures complete text extraction and processing

### 🧠 **Advanced AI & NLP**
- **Local Llama 3**: Complete privacy with Meta's latest language model
- **Semantic Embeddings**: 768-dimensional vectors using Sentence Transformers
- **Entity Recognition**: Extracts ages, procedures, locations, policy terms
- **Query Classification**: Understands coverage, exclusion, waiting period, and limit inquiries

### 🔍 **Powerful Search & Retrieval**
- **Vector Similarity Search**: ChromaDB with cosine similarity ranking
- **Metadata Filtering**: Filter by page, section, clause, or procedure type
- **Configurable Thresholds**: Adjustable similarity thresholds for precision/recall tuning
- **Fast Performance**: Sub-second query processing for real-time interaction

### 🌐 **Professional Web Interface**
- **Modern UI**: Beautiful, responsive Streamlit interface
- **Real-time Monitoring**: Live system metrics (RAM, CPU, GPU usage)
- **Performance Analytics**: Response time tracking and optimization insights
- **Example Queries**: Categorized examples for different query types
- **Debug Mode**: Complete transparency into retrieval and reasoning process

### 📊 **Enterprise Features**
- **Audit Logging**: Complete interaction logs for compliance
- **Structured Output**: JSON responses with confidence scores
- **Reference Tracking**: Exact clause citations for every answer
- **System Health**: Built-in diagnostics and performance monitoring

---

## 🛠️ **Installation & Setup**

### **📋 Prerequisites**

#### **System Requirements**
- **Operating System**: macOS, Linux, or Windows 10/11
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended for optimal performance)
- **Storage**: 10GB free space (for models and data)
- **Optional**: NVIDIA GPU with CUDA support for faster processing

#### **Dependencies Check**
```bash
# Check Python version
python3 --version  # Should be 3.8+

# Check available RAM
# macOS/Linux
free -h
# macOS specific
system_profiler SPHardwareDataType | grep Memory

# Windows
systeminfo | findstr "Total Physical Memory"
```

### **🚀 Quick Start**

#### **Option 1: Automated Setup (Recommended)**

1. **Clone the Repository**
```bash
git clone https://github.com/moghalsaif/hackrx.git
cd hackrx
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Setup Llama 3 (Interactive)**
```bash
python3 setup_llama3.py
```

4. **Launch the Application**
```bash
python3 app_launcher.py
```

5. **Open in Browser**
- The interface will automatically open at: `http://localhost:8501`
- If not, manually navigate to the URL shown in terminal

#### **Option 2: Manual Setup**

1. **Clone and Install**
```bash
git clone https://github.com/moghalsaif/hackrx.git
cd hackrx
pip install -r requirements.txt
```

2. **Install Ollama (for macOS/Linux)**
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: Download from https://ollama.ai/download
```

3. **Download Llama 3**
```bash
ollama pull llama3
```

4. **Configure Environment**
```bash
cp env_template.txt .env
# Edit .env file with your preferred settings
```

5. **Launch Application**
```bash
python3 app_launcher.py
```

### **🎛️ Configuration Options**

#### **Environment Variables (.env file)**
```bash
# LLM Configuration
LLM_PROVIDER=ollama                    # Options: ollama, openai, anthropic
OLLAMA_MODEL_NAME=llama3:latest        # Ollama model name
LLM_MODEL=llama3                       # Model identifier
LOCAL_MODEL_TYPE=ollama                # Local model backend

# Performance Tuning
SIMILARITY_THRESHOLD=0.4               # Retrieval similarity threshold (0.0-1.0)
CHUNK_SIZE=512                         # Document chunk size in characters
CHUNK_OVERLAP=50                       # Overlap between chunks
MAX_TOKENS=1000                        # Maximum LLM response tokens

# Hardware Settings
USE_GPU=true                           # Enable GPU acceleration
LOCAL_MODEL_DEVICE=auto                # Device: auto, cpu, cuda
LOCAL_MODEL_PRECISION=fp16             # Precision: fp32, fp16, int8, int4

# Optional: Cloud LLM APIs (alternative to local)
# OPENAI_API_KEY=your_openai_key_here
# ANTHROPIC_API_KEY=your_anthropic_key_here
```

### **🔧 Platform-Specific Instructions**

#### **macOS Setup**
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and dependencies
brew install python@3.9
pip3 install -r requirements.txt

# Install Ollama
brew install ollama

# Launch Ollama service
brew services start ollama

# Download Llama 3
ollama pull llama3

# Quick launch
./start_ragflow.sh
```

#### **Linux (Ubuntu/Debian) Setup**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3 python3-pip python3-venv -y

# Create virtual environment (recommended)
python3 -m venv ragflow-env
source ragflow-env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Download Llama 3
ollama pull llama3

# Launch application
./start_ragflow.sh
```

#### **Windows Setup**
```cmd
REM Install Python from python.org (3.8+ required)
REM Download and install Git from git-scm.com

REM Clone repository
git clone https://github.com/moghalsaif/hackrx.git
cd hackrx

REM Install dependencies
pip install -r requirements.txt

REM Download Ollama from https://ollama.ai/download
REM Install and start Ollama service

REM Download Llama 3
ollama pull llama3

REM Launch application
start_ragflow.bat
```

---

## 📖 **Usage Guide**

### **🌐 Web Interface**

#### **Getting Started**
1. Launch the application using `python3 app_launcher.py`
2. Open your browser to `http://localhost:8501`
3. Wait for system initialization (first run takes longer)
4. Start asking questions about your insurance policy!

#### **Example Queries**

**Coverage Questions:**
- "Is orthopedic surgery covered for a 45-year-old patient?"
- "What knee surgery procedures are covered under this policy?"
- "Is cardiac bypass surgery covered for emergency cases?"
- "Are maternity benefits available for 28-year-old members?"

**Exclusions & Limitations:**
- "Are cosmetic surgeries excluded from coverage?"
- "What dental treatments are not covered?"
- "Are pre-existing conditions excluded?"
- "What are the age-related coverage restrictions?"

**Waiting Periods:**
- "What is the waiting period for orthopedic surgery?"
- "Can I get treatment immediately after buying the policy?"
- "When does surgery coverage become active?"

**Complex Scenarios:**
- "I'm 46, bought policy 3 months ago, need ACL surgery in Mumbai. What's covered?"
- "Pre-existing diabetes patient needs cardiac surgery - what are the options?"
- "Emergency appendectomy during vacation - will it be covered?"

#### **Interface Features**

**🎛️ Control Panel:**
- **Model Status**: Real-time LLM configuration and health
- **System Metrics**: Live RAM, CPU, GPU monitoring
- **Performance Charts**: Response time analytics
- **Setup Validation**: Automatic issue detection and fixes

**💡 Smart Features:**
- **Categorized Examples**: Pre-built queries by type
- **Random Examples**: Quick exploration of capabilities
- **Debug Mode**: Complete query analysis transparency
- **Performance Tracking**: Real-time response metrics

### **🖥️ Command Line Interface**

#### **Basic Usage**
```python
from ragflow_system import RAGFlowSystem

# Initialize system
rag_system = RAGFlowSystem()
rag_system.initialize_system()

# Query the system
response = rag_system.query_system(
    "Is ACL reconstruction surgery covered for a 46-year-old under a 3-month-old policy?"
)

# Access results
print(f"Decision: {response['decision']}")
print(f"Justification: {response['justification']}")
print(f"References: {response['references']}")
```

#### **Advanced Usage**
```python
# Query with debug information
response = rag_system.query_system(
    query="What is the waiting period for cardiac surgery?",
    include_debug_info=True
)

# Access debug information
debug_info = response['debug_info']
print(f"Retrieved chunks: {len(debug_info['retrieved_chunks'])}")
print(f"Query type: {debug_info['parsed_query']['query_type']}")
```

### **📊 Response Format**

All responses follow this structured JSON format:

```json
{
  "decision": "approved|rejected|conditional|uncertain|information_only",
  "justification": "Detailed explanation with policy references",
  "references": [
    {
      "clause_number": "4.2.1",
      "section": "Surgical Procedures Coverage",
      "page": 15
    }
  ],
  "confidence_score": 0.85,
  "processing_time_seconds": 1.23,
  "system_metadata": {
    "query_type": "coverage_inquiry",
    "retrieved_chunks_count": 3,
    "system_version": "1.0.0",
    "policy_document": "BAJHLIP23020V012223.pdf"
  }
}
```

---

## 🧪 **Testing & Validation**

### **Run Test Suite**
```bash
# Comprehensive system tests
python3 test_system.py

# Quick demo with sample queries
python3 run_demo.py

# Performance benchmarking
python3 -c "from ragflow_system import RAGFlowSystem; rag = RAGFlowSystem(); rag.initialize_system(); print('System ready for testing!')"
```

### **Validate Setup**
```bash
# Check system configuration
python3 -c "from config import RAGFlowConfig; print('✅ Configuration loaded successfully')"

# Test Ollama connection
ollama list

# Verify model availability
ollama run llama3 "Hello, are you working?"
```

---

## 🏗️ **System Architecture**

### **Components Overview**

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAGFlow Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│  📄 PDF Document → 🔧 Document Processor → 📋 Intelligent Chunks │
│                                                                 │
│  🔢 Embedding Generator → 🗄️ Vector Database (ChromaDB)          │
│                                                                 │
│  💬 User Query → 🔍 Query Processor → 🎯 Semantic Search        │
│                                                                 │
│  🦙 Llama 3 LLM → 📋 Structured Response → 🌐 Web Interface     │
└─────────────────────────────────────────────────────────────────┘
```

### **Data Flow**
1. **Document Ingestion**: PDF → Text extraction → Intelligent chunking
2. **Embedding Generation**: Text chunks → 768D vectors → ChromaDB storage
3. **Query Processing**: Natural language → Entity extraction → Search terms
4. **Semantic Retrieval**: Query vectors → Similarity search → Relevant chunks
5. **LLM Reasoning**: Retrieved context → Llama 3 → Structured response
6. **Response Delivery**: JSON output → Web interface → User display

### **Core Files**
- `config.py` - System configuration and settings
- `document_processor.py` - PDF processing and intelligent chunking
- `embedding_manager.py` - Vector generation and ChromaDB management
- `query_processor.py` - Natural language query analysis
- `llm_reasoner.py` - Llama 3 integration and response generation
- `ragflow_system.py` - Main system orchestrator
- `web_interface.py` - Streamlit web application
- `app_launcher.py` - Application launcher with setup validation

---

## 🚨 **Troubleshooting**

### **Common Issues & Solutions**

#### **🔧 Installation Issues**

**"Python not found"**
```bash
# macOS: Install via Homebrew
brew install python@3.9

# Linux: Install via package manager
sudo apt install python3 python3-pip

# Windows: Download from python.org
```

**"pip install fails"**
```bash
# Upgrade pip
python3 -m pip install --upgrade pip

# Use virtual environment
python3 -m venv ragflow-env
source ragflow-env/bin/activate  # Linux/macOS
# ragflow-env\Scripts\activate  # Windows
pip install -r requirements.txt
```

#### **🦙 Llama 3 Issues**

**"Ollama not found"**
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: Download from https://ollama.ai/download
```

**"Model not found"**
```bash
# Download Llama 3
ollama pull llama3

# Verify installation
ollama list

# Test model
ollama run llama3 "Hello"
```

**"Ollama service not running"**
```bash
# Start Ollama service
ollama serve

# Check if running
curl http://localhost:11434/api/tags
```

#### **⚡ Performance Issues**

**"Slow response times"**
- Enable GPU acceleration: Set `USE_GPU=true` in `.env`
- Lower precision: Set `LOCAL_MODEL_PRECISION=int8` or `int4`
- Reduce similarity threshold: Set `SIMILARITY_THRESHOLD=0.3`

**"High memory usage"**
- Use CPU inference: Set `LOCAL_MODEL_DEVICE=cpu`
- Enable model quantization: Set `LOCAL_MODEL_PRECISION=int8`
- Reduce chunk size: Set `CHUNK_SIZE=256`

**"Port already in use"**
```bash
# Find and kill process using port 8501
lsof -ti:8501 | xargs kill -9

# Or use different port
streamlit run web_interface.py --server.port=8502
```

#### **🔍 Search Issues**

**"No results found"**
- Lower similarity threshold in `.env`: `SIMILARITY_THRESHOLD=0.3`
- Check document processing: Ensure PDF text was extracted correctly
- Verify embeddings: Check if vector database has content

**"Irrelevant results"**
- Increase similarity threshold: `SIMILARITY_THRESHOLD=0.6`
- Enable debug mode to inspect retrieved chunks
- Refine query phrasing for better semantic matching

#### **🌐 Web Interface Issues**

**"Interface won't load"**
- Check if all dependencies are installed: `pip install -r requirements.txt`
- Verify port availability: `lsof -i:8501`
- Check terminal for error messages
- Try launching with: `python3 app_launcher.py`

**"System configuration errors"**
- Verify `.env` file exists and has correct values
- Run setup script: `python3 setup_llama3.py`
- Check file permissions and paths

### **🩺 System Diagnostics**

```bash
# Complete system check
python3 app_launcher.py --test-only

# Component-wise testing
python3 -c "
from config import RAGFlowConfig
from embedding_manager import EmbeddingManager
from llm_reasoner import LLMReasoner

config = RAGFlowConfig()
print('✅ Configuration: OK')

embedding_mgr = EmbeddingManager(config)
print('✅ Embedding Manager: OK')

llm_reasoner = LLMReasoner(config)
print('✅ LLM Reasoner: OK')

print('🎉 All components working!')
"
```

---

## 🤝 **Contributing**

We welcome contributions to RAGFlow! Here's how you can help:

### **Development Setup**
```bash
# Fork and clone the repository
git clone https://github.com/yourusername/hackrx.git
cd hackrx

# Create development environment
python3 -m venv dev-env
source dev-env/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy

# Run tests
python3 test_system.py
pytest
```

### **Code Standards**
- Follow PEP 8 style guidelines
- Add type hints for new functions
- Include docstrings for all public methods
- Write tests for new features
- Update documentation as needed

### **Submitting Changes**
1. Create a feature branch: `git checkout -b feature-name`
2. Make your changes and test thoroughly
3. Run code quality checks: `black . && flake8 . && mypy .`
4. Commit with clear messages: `git commit -m "Add feature description"`
5. Push and create a pull request

---

## 📜 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- **Meta AI** - For the Llama 3 language model
- **Ollama** - For local model management and serving
- **Streamlit** - For the beautiful web interface framework
- **Hugging Face** - For Sentence Transformers and model hosting
- **ChromaDB** - For vector database functionality
- **The Open Source Community** - For the amazing tools and libraries

---

## 📞 **Support & Community**

### **Getting Help**
- 📖 **Documentation**: Check this README and inline code comments
- 🐛 **Bug Reports**: Open an issue on GitHub with detailed reproduction steps
- 💡 **Feature Requests**: Suggest improvements via GitHub issues
- 💬 **Discussions**: Join community discussions in GitHub Discussions

### **Quick Links**
- 🔗 **Repository**: [https://github.com/moghalsaif/hackrx.git](https://github.com/moghalsaif/hackrx.git)
- 📚 **Documentation**: See inline code documentation and examples
- 🎥 **Video Tutorials**: Coming soon!
- 📧 **Contact**: Open an issue for support requests

---

<div align="center">

**🎉 Ready to revolutionize insurance policy analysis with AI?**

[🚀 Get Started Now](#-quick-start) • [⭐ Star this Repository](https://github.com/moghalsaif/hackrx) • [🍴 Fork & Contribute](https://github.com/moghalsaif/hackrx/fork)

---

*Built with ❤️ for better insurance policy understanding using local AI*

**RAGFlow v1.0.0** | **Powered by Llama 3** | **100% Private & Local**

</div> 