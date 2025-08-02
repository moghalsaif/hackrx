import streamlit as st
import json
import time
import os
from typing import Dict, Any, List
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import psutil
import subprocess

from config import RAGFlowConfig
from ragflow_system import RAGFlowSystem

# Page configuration
st.set_page_config(
    page_title="RAGFlow Insurance Policy Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .decision-approved {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid #28a745;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .decision-rejected {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 2px solid #dc3545;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .decision-conditional {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 2px solid #ffc107;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .decision-uncertain {
        background: linear-gradient(135deg, #e2e3e5 0%, #d6d8db 100%);
        border: 2px solid #6c757d;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .reference-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 6px solid #007bff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid #dee2e6;
    }
    .setup-warning {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 2px solid #ffc107;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .model-status {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
        margin: 0.5rem 0;
    }
    .status-active {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .status-inactive {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .query-example {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.25rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .query-example:hover {
        background-color: #e9ecef;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_system():
    """Initialize and cache the RAG system"""
    with st.spinner("🚀 Initializing RAGFlow system... This may take a few minutes on first run."):
        try:
            config = RAGFlowConfig()
            rag_system = RAGFlowSystem(config)
            
            if rag_system.initialize_system():
                return rag_system, None
            else:
                return None, "Failed to initialize RAG system"
        except Exception as e:
            return None, str(e)

def check_system_setup():
    """Check if the system is properly set up"""
    config = RAGFlowConfig()
    issues = []
    
    # Check LLM configuration
    if config.LLM_PROVIDER == "openai" and not config.get_openai_api_key():
        issues.append("OpenAI API key not configured")
    elif config.LLM_PROVIDER == "anthropic" and not config.get_anthropic_api_key():
        issues.append("Anthropic API key not configured")
    elif config.LLM_PROVIDER in ["local", "ollama"]:
        if config.LOCAL_MODEL_TYPE == "ollama":
            # Check if Ollama is running
            try:
                result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
                if result.returncode != 0:
                    issues.append("Ollama service not running")
                elif config.OLLAMA_MODEL_NAME not in result.stdout:
                    issues.append(f"Ollama model '{config.OLLAMA_MODEL_NAME}' not found")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                issues.append("Ollama not installed or not accessible")
        elif config.LOCAL_MODEL_TYPE == "transformers":
            if not Path(config.LOCAL_MODEL_PATH).exists():
                issues.append(f"Model path not found: {config.LOCAL_MODEL_PATH}")
    
    # Check PDF file
    if not Path(config.PDF_FILE).exists():
        issues.append(f"PDF file not found: {config.PDF_FILE}")
    
    return issues

def display_setup_guidance(issues: List[str]):
    """Display setup guidance for any issues found"""
    if not issues:
        st.success("🎉 System is properly configured!")
        return
    
    st.markdown("""
    <div class="setup-warning">
        <h3>⚠️ Setup Required</h3>
        <p>Some configuration issues were detected. Please follow the guidance below:</p>
    </div>
    """, unsafe_allow_html=True)
    
    for issue in issues:
        if "Ollama" in issue:
            st.error(f"❌ {issue}")
            st.info("**Fix:** Install and start Ollama:")
            st.code("""
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Download Llama 3
ollama pull llama3

# Start Ollama service (if needed)
ollama serve
            """)
        elif "API key" in issue:
            st.error(f"❌ {issue}")
            st.info("**Fix:** Set your API key in the environment or .env file")
        elif "Model path" in issue:
            st.error(f"❌ {issue}")
            st.info("**Fix:** Download the model files or run the setup script:")
            st.code("python setup_llama3.py")
        elif "PDF file" in issue:
            st.error(f"❌ {issue}")
            st.info("**Fix:** Ensure the PDF file is in the correct location")
        else:
            st.error(f"❌ {issue}")

def get_system_info():
    """Get system information for display"""
    info = {}
    
    # System resources
    memory = psutil.virtual_memory()
    info['ram_total'] = memory.total / (1024**3)
    info['ram_used'] = memory.used / (1024**3)
    info['ram_percent'] = memory.percent
    
    # CPU info
    info['cpu_percent'] = psutil.cpu_percent(interval=1)
    info['cpu_count'] = psutil.cpu_count()
    
    # GPU info
    try:
        import torch
        info['gpu_available'] = torch.cuda.is_available()
        if info['gpu_available']:
            info['gpu_count'] = torch.cuda.device_count()
            info['gpu_name'] = torch.cuda.get_device_name(0) if info['gpu_count'] > 0 else "Unknown"
        else:
            info['gpu_count'] = 0
            info['gpu_name'] = "No GPU"
    except ImportError:
        info['gpu_available'] = False
        info['gpu_count'] = 0
        info['gpu_name'] = "PyTorch not available"
    
    return info

def display_model_status():
    """Display current model configuration and status"""
    config = RAGFlowConfig()
    
    st.subheader("🤖 Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Provider:** {config.LLM_PROVIDER}")
        st.write(f"**Model:** {config.LLM_MODEL}")
        if config.LLM_PROVIDER in ["local", "ollama"]:
            st.write(f"**Backend:** {config.LOCAL_MODEL_TYPE}")
            if config.LOCAL_MODEL_TYPE == "ollama":
                st.write(f"**Ollama Model:** {config.OLLAMA_MODEL_NAME}")
            else:
                st.write(f"**Model Path:** {config.LOCAL_MODEL_PATH}")
    
    with col2:
        st.write(f"**Temperature:** {config.LLM_TEMPERATURE}")
        st.write(f"**Max Tokens:** {config.MAX_TOKENS}")
        if config.LLM_PROVIDER in ["local", "ollama"]:
            st.write(f"**Device:** {config.LOCAL_MODEL_DEVICE}")
            st.write(f"**Precision:** {config.LOCAL_MODEL_PRECISION}")
    
    # Model status indicator
    issues = check_system_setup()
    if not issues:
        st.markdown('<div class="model-status status-active">✅ Model Ready</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="model-status status-inactive">❌ Configuration Issues</div>', unsafe_allow_html=True)

def display_system_metrics():
    """Display real-time system metrics"""
    info = get_system_info()
    
    st.subheader("📊 System Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="RAM Usage",
            value=f"{info['ram_used']:.1f} GB",
            delta=f"{info['ram_percent']:.1f}%"
        )
    
    with col2:
        st.metric(
            label="CPU Usage",
            value=f"{info['cpu_percent']:.1f}%",
            delta=f"{info['cpu_count']} cores"
        )
    
    with col3:
        gpu_status = "Available" if info['gpu_available'] else "Not Available"
        st.metric(
            label="GPU Status",
            value=gpu_status,
            delta=f"{info['gpu_count']} devices"
        )
    
    with col4:
        st.metric(
            label="GPU Device",
            value=info['gpu_name'][:15] + "..." if len(info['gpu_name']) > 15 else info['gpu_name'],
            delta="CUDA" if info['gpu_available'] else "CPU Only"
        )

def display_decision_card(response: Dict[str, Any]):
    """Display the decision in a styled card"""
    decision = response.get("decision", "unknown")
    confidence = response.get("confidence_score", 0)
    
    # Decision styling
    decision_classes = {
        "approved": "decision-approved",
        "rejected": "decision-rejected", 
        "conditional": "decision-conditional",
        "uncertain": "decision-uncertain",
        "information_only": "decision-conditional"
    }
    
    decision_icons = {
        "approved": "✅",
        "rejected": "❌",
        "conditional": "⚠️",
        "uncertain": "❓",
        "information_only": "ℹ️",
        "error": "🚫"
    }
    
    css_class = decision_classes.get(decision, "decision-uncertain")
    icon = decision_icons.get(decision, "❓")
    
    # Confidence color
    conf_color = "#28a745" if confidence > 0.8 else "#ffc107" if confidence > 0.5 else "#dc3545"
    
    st.markdown(f"""
    <div class="{css_class}">
        <h3>{icon} Decision: {decision.upper()}</h3>
        <div style="display: flex; align-items: center; margin: 1rem 0;">
            <span style="margin-right: 1rem;"><strong>Confidence Score:</strong></span>
            <div style="background-color: #f0f0f0; border-radius: 10px; padding: 5px; width: 200px;">
                <div style="background-color: {conf_color}; height: 20px; width: {confidence*100}%; border-radius: 5px;"></div>
            </div>
            <span style="margin-left: 1rem; font-weight: bold; color: {conf_color};">{confidence:.2f} ({confidence:.1%})</span>
        </div>
        <p><strong>Justification:</strong></p>
        <p style="background-color: rgba(255,255,255,0.7); padding: 1rem; border-radius: 8px; margin-top: 1rem;">
            {response.get('justification', 'No justification provided.')}
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_references(references: list):
    """Display policy references in styled boxes"""
    if not references:
        st.info("No specific policy references found.")
        return
    
    st.subheader("📋 Policy References")
    
    for i, ref in enumerate(references, 1):
        clause = ref.get("clause_number", "Unknown")
        section = ref.get("section", "Unknown Section")
        page = ref.get("page", "Unknown")
        
        st.markdown(f"""
        <div class="reference-box">
            <h4 style="margin: 0 0 0.5rem 0; color: #007bff;">Reference {i}</h4>
            <p style="margin: 0.25rem 0;"><strong>Clause:</strong> {clause}</p>
            <p style="margin: 0.25rem 0;"><strong>Section:</strong> {section}</p>
            <p style="margin: 0.25rem 0;"><strong>Page:</strong> {page}</p>
        </div>
        """, unsafe_allow_html=True)

def create_query_examples():
    """Create categorized example queries"""
    examples = {
        "Coverage Questions": [
            "Is ACL reconstruction surgery covered for a 46-year-old under a 3-month-old policy?",
            "What knee surgery procedures are covered under this policy?",
            "Is cardiac bypass surgery covered for emergency cases?",
            "Are orthopedic treatments covered for sports injuries?",
            "What maternity benefits are available for 28-year-old members?"
        ],
        "Exclusions & Limitations": [
            "Are cosmetic surgeries excluded from coverage?",
            "What dental treatments are not covered?",
            "Are pre-existing conditions excluded?",
            "Is alternative medicine covered under this policy?",
            "What are the age-related coverage restrictions?"
        ],
        "Waiting Periods": [
            "What is the waiting period for orthopedic surgery?",
            "Can I get treatment immediately after buying the policy?",
            "When does surgery coverage become active?",
            "What's the waiting period for maternity benefits?",
            "Are there different waiting periods for different procedures?"
        ],
        "Benefits & Limits": [
            "What are the coverage limits for orthopedic procedures?",
            "Is there a sub-limit for physiotherapy treatments?",
            "What's the maximum annual benefit?",
            "What is the room rent limit for hospitalization?",
            "Are there any day-care procedure benefits?"
        ],
        "Complex Scenarios": [
            "I'm 45, bought policy 6 months ago, need ACL surgery in Mumbai. What's covered?",
            "Pre-existing diabetes patient needs cardiac surgery - what are the options?",
            "Emergency appendectomy during vacation - will it be covered?",
            "Multiple surgeries needed - what are the cumulative limits?",
            "Treatment required abroad - what's the coverage?"
        ]
    }
    
    return examples

def display_performance_chart(response_times: List[float]):
    """Display performance chart"""
    if not response_times:
        return
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=response_times,
        mode='lines+markers',
        name='Response Time',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Response Time Performance",
        xaxis_title="Query Number",
        yaxis_title="Response Time (seconds)",
        height=300,
        showlegend=False
    )
    
    return fig

def main():
    """Main Streamlit application"""
    # Header
    st.markdown('<h1 class="main-header">🏥 RAGFlow Insurance Policy Assistant</h1>', unsafe_allow_html=True)
    st.markdown("**Powered by Local Llama 3** - Ask questions about your insurance policy and get precise, well-referenced answers.")
    
    # Check system setup
    issues = check_system_setup()
    
    if issues:
        display_setup_guidance(issues)
        st.stop()
    
    # Initialize system
    rag_system, error = initialize_rag_system()
    
    if not rag_system:
        st.error(f"❌ Failed to initialize the system: {error}")
        st.info("Please check your configuration and try again.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ System Control Panel")
        
        # Model status
        display_model_status()
        
        st.markdown("---")
        
        # System metrics
        display_system_metrics()
        
        st.markdown("---")
        
        st.header("💡 Example Queries")
        examples = create_query_examples()
        
        selected_category = st.selectbox("Choose category:", list(examples.keys()))
        
        for i, example in enumerate(examples[selected_category][:3]):  # Show top 3
            if st.button(f"📝 {example[:35]}...", key=f"example_{selected_category}_{i}", use_container_width=True):
                st.session_state.query_input = example
        
        if st.button("🔄 Show More Examples", use_container_width=True):
            st.session_state.show_all_examples = not st.session_state.get('show_all_examples', False)
        
        if st.session_state.get('show_all_examples', False):
            for example in examples[selected_category][3:]:
                if st.button(f"📝 {example[:35]}...", key=f"example_extra_{hash(example)}", use_container_width=True):
                    st.session_state.query_input = example
        
        st.markdown("---")
        
        st.header("⚙️ Settings")
        include_debug = st.checkbox("Include debug information", value=False)
        show_performance = st.checkbox("Show performance metrics", value=True)
        
        if st.button("🔄 Reset System", type="secondary", use_container_width=True):
            if rag_system.reset_system():
                st.success("System reset successfully!")
                st.rerun()
            else:
                st.error("Failed to reset system")
        
        if st.button("🧪 Run System Test", use_container_width=True):
            with st.spinner("Running system test..."):
                try:
                    test_response = rag_system.query_system("Is surgery covered?", include_debug_info=False)
                    if test_response.get('decision') != 'error':
                        st.success("✅ System test passed!")
                    else:
                        st.error("❌ System test failed")
                except Exception as e:
                    st.error(f"❌ System test failed: {e}")
    
    # Main content
    st.header("💬 Ask Your Insurance Policy Question")
    
    # Query input with enhanced styling
    query = st.text_area(
        "Enter your insurance policy question:",
        value=st.session_state.get('query_input', ''),
        height=100,
        placeholder="e.g., Is knee surgery covered for a 45-year-old patient who bought the policy 6 months ago?",
        help="Ask any question about coverage, exclusions, waiting periods, benefits, or complex scenarios."
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_button = st.button("🔍 Analyze Query", type="primary", use_container_width=True)
    
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    with col3:
        if st.button("🎲 Random Example", use_container_width=True):
            import random
            all_examples = []
            for category_examples in examples.values():
                all_examples.extend(category_examples)
            st.session_state.query_input = random.choice(all_examples)
            st.rerun()
    
    if clear_button:
        st.session_state.query_input = ""
        st.rerun()
    
    # Initialize session state for performance tracking
    if 'response_times' not in st.session_state:
        st.session_state.response_times = []
    
    # Process query
    if search_button and query.strip():
        with st.spinner("🤔 Analyzing your question with Llama 3..."):
            start_time = time.time()
            
            try:
                response = rag_system.query_system(query, include_debug_info=include_debug)
                processing_time = time.time() - start_time
                
                # Track performance
                if show_performance:
                    st.session_state.response_times.append(processing_time)
                    if len(st.session_state.response_times) > 20:  # Keep last 20
                        st.session_state.response_times = st.session_state.response_times[-20:]
                
                # Display results
                st.header("📋 Analysis Results")
                
                # Decision card
                display_decision_card(response)
                
                # References
                references = response.get("references", [])
                display_references(references)
                
                # Performance metrics
                if show_performance:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Processing Time", f"{processing_time:.2f}s")
                    
                    with col2:
                        st.metric("Confidence", f"{response.get('confidence_score', 0):.2f}")
                    
                    with col3:
                        st.metric("References Found", len(references))
                    
                    with col4:
                        chunks_count = response.get('system_metadata', {}).get('retrieved_chunks_count', 0)
                        st.metric("Chunks Retrieved", chunks_count)
                    
                    # Performance chart
                    if len(st.session_state.response_times) > 1:
                        st.plotly_chart(
                            display_performance_chart(st.session_state.response_times),
                            use_container_width=True
                        )
                
                # Metadata
                with st.expander("📊 Response Metadata", expanded=False):
                    metadata = response.get("system_metadata", {})
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Query Type:** {metadata.get('query_type', 'N/A')}")
                        st.write(f"**Retrieved Chunks:** {metadata.get('retrieved_chunks_count', 0)}")
                        st.write(f"**Processing Time:** {response.get('processing_time_seconds', 0):.2f}s")
                    
                    with col2:
                        st.write(f"**System Version:** {metadata.get('system_version', 'N/A')}")
                        st.write(f"**Policy Document:** {metadata.get('policy_document', 'N/A')}")
                        st.write(f"**LLM Provider:** {rag_system.config.LLM_PROVIDER}")
                
                # Debug information
                if include_debug:
                    debug_info = response.get("debug_info", {})
                    if debug_info:
                        with st.expander("🔍 Debug Information", expanded=False):
                            parsed_query = debug_info.get("parsed_query", {})
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("Query Analysis")
                                st.write(f"**Original Query:** {parsed_query.get('original', 'N/A')}")
                                st.write(f"**Processed Query:** {parsed_query.get('processed', 'N/A')}")
                                st.write(f"**Parsing Confidence:** {parsed_query.get('parsing_confidence', 0):.2f}")
                                
                                entities = parsed_query.get("entities", {})
                                if entities:
                                    st.subheader("Extracted Entities")
                                    for key, value in entities.items():
                                        if value:
                                            st.write(f"**{key.title()}:** {value}")
                            
                            with col2:
                                st.subheader("Retrieved Chunks")
                                retrieved_chunks = debug_info.get("retrieved_chunks", [])
                                
                                if retrieved_chunks:
                                    for i, chunk in enumerate(retrieved_chunks, 1):
                                        with st.container():
                                            st.write(f"**Chunk {i}** (Score: {chunk.get('similarity_score', 0):.3f})")
                                            st.write(f"Page {chunk.get('page_number', 'N/A')}, Section: {chunk.get('section_title', 'N/A')}")
                                            st.write(f"Preview: {chunk.get('text_preview', 'N/A')}")
                                            st.write("---")
                
                # Success message
                st.success(f"✅ Query processed successfully in {processing_time:.2f} seconds using {rag_system.config.LLM_PROVIDER} ({rag_system.config.LLM_MODEL})!")
                
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
                st.info("Please check your system configuration and try again.")
    
    elif search_button:
        st.warning("⚠️ Please enter a question to analyze.")
    
    # Footer
    st.markdown("---")
    
    # System status footer
    status = rag_system.get_system_status()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Chunks Loaded", status.get('chunks_loaded', 0))
    
    with col2:
        system_stats = status.get('system_stats', {})
        st.metric("Unique Pages", system_stats.get('unique_pages', 0))
    
    with col3:
        st.metric("Model Provider", rag_system.config.LLM_PROVIDER.upper())
    
    with col4:
        st.metric("Model Type", rag_system.config.LLM_MODEL)
    
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🏥 <strong>RAGFlow Insurance Policy Assistant v1.0.0</strong></p>
        <p>Powered by <strong>Local Llama 3</strong> • Advanced NLP • Retrieval-Augmented Generation</p>
        <p>🔒 <em>100% Private • No Data Sent to External Services</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 