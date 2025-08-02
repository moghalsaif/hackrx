#!/usr/bin/env python3
"""
RAGFlow Insurance Policy System - Demo Script

This script demonstrates the complete functionality of the RAGFlow system
with sample queries and showcases all key features.
"""

import json
import time
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from config import RAGFlowConfig
from ragflow_system import RAGFlowSystem

def print_banner():
    """Print a welcome banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                  RAGFlow Insurance Policy System                 ║
    ║                           Demo Script                           ║
    ╚══════════════════════════════════════════════════════════════════╝
    
    🏥 A comprehensive RAG system for insurance policy analysis
    🔍 Providing traceable, clause-aware responses to policy queries
    📋 Complete with exact clause references and page numbers
    """
    print(banner)

def print_section_header(title: str):
    """Print a section header"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

def print_query_result(query: str, response: dict, query_num: int = None):
    """Print formatted query results"""
    prefix = f"Query {query_num}: " if query_num else "Query: "
    
    print(f"\n{prefix}{query}")
    print("-" * 80)
    
    # Decision with emoji
    decision = response.get('decision', 'unknown')
    decision_emojis = {
        'approved': '✅',
        'rejected': '❌',
        'conditional': '⚠️',
        'uncertain': '❓',
        'information_only': 'ℹ️',
        'error': '🚫'
    }
    
    emoji = decision_emojis.get(decision, '❓')
    print(f"🎯 Decision: {emoji} {decision.upper()}")
    print(f"🔒 Confidence: {response.get('confidence_score', 0):.2f} ({response.get('confidence_score', 0):.1%})")
    print(f"⏱️ Processing Time: {response.get('processing_time_seconds', 0):.2f}s")
    
    # Justification
    justification = response.get('justification', 'No justification provided.')
    print(f"\n📝 Justification:")
    print(f"   {justification}")
    
    # References
    references = response.get('references', [])
    if references:
        print(f"\n📋 Policy References:")
        for i, ref in enumerate(references, 1):
            clause = ref.get('clause_number', 'Unknown')
            section = ref.get('section', 'Unknown Section')
            page = ref.get('page', 'Unknown')
            print(f"   {i}. Clause {clause} - {section} (Page {page})")
    else:
        print(f"\n📋 Policy References: None found")
    
    # Metadata
    metadata = response.get('system_metadata', {})
    print(f"\n📊 Metadata: {metadata.get('query_type', 'N/A')} | "
          f"Chunks: {metadata.get('retrieved_chunks_count', 0)} | "
          f"System: v{metadata.get('system_version', 'N/A')}")

def run_demo_queries(rag_system: RAGFlowSystem):
    """Run a series of demo queries showcasing different capabilities"""
    
    demo_queries = [
        {
            "category": "Coverage Inquiries",
            "queries": [
                "Is ACL reconstruction surgery covered for a 46-year-old under a 3-month-old policy?",
                "What knee surgery procedures are covered under this policy?",
                "Is cardiac bypass surgery covered for emergency cases?"
            ]
        },
        {
            "category": "Exclusion Inquiries", 
            "queries": [
                "Are cosmetic surgeries excluded from coverage?",
                "What dental treatments are not covered?",
                "Are pre-existing conditions excluded?"
            ]
        },
        {
            "category": "Waiting Period Questions",
            "queries": [
                "What is the waiting period for orthopedic surgery?",
                "Can I get treatment immediately after buying the policy?",
                "When does surgery coverage become active?"
            ]
        },
        {
            "category": "Limit and Benefit Questions",
            "queries": [
                "What are the coverage limits for orthopedic procedures?",
                "Is there a sub-limit for physiotherapy treatments?",
                "What's the maximum annual benefit?"
            ]
        },
        {
            "category": "Complex Scenarios",
            "queries": [
                "I'm 45, bought policy 6 months ago, need ACL surgery in Mumbai. What's covered?",
                "Pre-existing diabetes patient needs cardiac surgery - what are the options?"
            ]
        }
    ]
    
    total_queries = 0
    successful_queries = 0
    total_time = 0
    
    for category_data in demo_queries:
        category = category_data["category"]
        queries = category_data["queries"]
        
        print_section_header(f"🔍 {category}")
        
        for i, query in enumerate(queries, 1):
            try:
                start_time = time.time()
                response = rag_system.query_system(query, include_debug_info=False)
                query_time = time.time() - start_time
                
                print_query_result(query, response, i)
                
                total_queries += 1
                if response.get('decision') != 'error':
                    successful_queries += 1
                total_time += query_time
                
                # Add a pause between queries for readability
                time.sleep(0.5)
                
            except Exception as e:
                print(f"\n❌ Error processing query: {str(e)}")
                total_queries += 1
    
    return total_queries, successful_queries, total_time

def show_system_statistics(rag_system: RAGFlowSystem):
    """Display comprehensive system statistics"""
    print_section_header("📊 System Statistics")
    
    status = rag_system.get_system_status()
    system_stats = status.get('system_stats', {})
    config = status.get('config', {})
    
    print(f"📄 Document Information:")
    print(f"   └─ Policy Document: {config.get('pdf_file', 'N/A')}")
    print(f"   └─ Total Chunks: {status.get('chunks_loaded', 0)}")
    print(f"   └─ Unique Pages: {system_stats.get('unique_pages', 0)}")
    print(f"   └─ Unique Sections: {system_stats.get('unique_sections', 0)}")
    print(f"   └─ Unique Clauses: {system_stats.get('unique_clauses', 0)}")
    print(f"   └─ Average Chunk Length: {system_stats.get('avg_chunk_length', 0):.1f} characters")
    
    print(f"\n🤖 Model Configuration:")
    print(f"   └─ Embedding Model: {config.get('embedding_model', 'N/A')}")
    print(f"   └─ LLM Model: {config.get('llm_model', 'N/A')}")
    print(f"   └─ Vector Database: {config.get('vector_db_type', 'N/A')}")
    
    vector_stats = system_stats.get('vector_db_stats', {})
    print(f"\n🗄️ Vector Database:")
    print(f"   └─ Total Vectors: {vector_stats.get('total_chunks', 0)}")
    print(f"   └─ Embedding Dimension: {vector_stats.get('embedding_dimension', 0)}")
    print(f"   └─ Collection Name: {vector_stats.get('collection_name', 'N/A')}")

def run_performance_test(rag_system: RAGFlowSystem):
    """Run a quick performance test"""
    print_section_header("⚡ Performance Test")
    
    test_queries = [
        "Is surgery covered?",
        "What are the exclusions?",
        "What is the waiting period?",
        "Are there coverage limits?",
        "Is emergency treatment covered?"
    ]
    
    times = []
    
    print("Running performance test with 5 queries...")
    
    for i, query in enumerate(test_queries, 1):
        try:
            start_time = time.time()
            response = rag_system.query_system(query, include_debug_info=False)
            query_time = time.time() - start_time
            times.append(query_time)
            
            status = "✅" if response.get('decision') != 'error' else "❌"
            print(f"   Query {i}: {query_time:.2f}s {status}")
            
        except Exception as e:
            print(f"   Query {i}: Error - {str(e)}")
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\n📈 Performance Summary:")
        print(f"   └─ Average Response Time: {avg_time:.2f}s")
        print(f"   └─ Fastest Response: {min_time:.2f}s")
        print(f"   └─ Slowest Response: {max_time:.2f}s")
        print(f"   └─ Total Test Time: {sum(times):.2f}s")

def main():
    """Main demo function"""
    print_banner()
    
    try:
        # Check for API key
        config = RAGFlowConfig()
        if not config.get_openai_api_key() and not config.get_anthropic_api_key():
            print("❌ No LLM API key found!")
            print("Please set either OPENAI_API_KEY or ANTHROPIC_API_KEY in your environment.")
            print("See .env.template for configuration instructions.")
            return
        
        print("🚀 Initializing RAGFlow system...")
        print("   This may take a few minutes on the first run...")
        
        # Initialize system
        rag_system = RAGFlowSystem(config)
        
        initialization_start = time.time()
        if not rag_system.initialize_system():
            print("❌ Failed to initialize RAGFlow system")
            print("Please check your configuration and try again.")
            return
        
        initialization_time = time.time() - initialization_start
        print(f"✅ System initialized successfully in {initialization_time:.2f} seconds!")
        
        # Show system statistics
        show_system_statistics(rag_system)
        
        # Run performance test
        run_performance_test(rag_system)
        
        # Run demo queries
        print_section_header("🎯 Demo Queries")
        print("Running comprehensive demo queries across different categories...")
        
        demo_start = time.time()
        total_queries, successful_queries, total_time = run_demo_queries(rag_system)
        demo_total_time = time.time() - demo_start
        
        # Final summary
        print_section_header("🎉 Demo Summary")
        
        success_rate = (successful_queries / total_queries * 100) if total_queries > 0 else 0
        avg_response_time = total_time / total_queries if total_queries > 0 else 0
        
        print(f"📊 Results:")
        print(f"   └─ Total Queries Processed: {total_queries}")
        print(f"   └─ Successful Responses: {successful_queries}")
        print(f"   └─ Success Rate: {success_rate:.1f}%")
        print(f"   └─ Average Response Time: {avg_response_time:.2f}s")
        print(f"   └─ Total Demo Time: {demo_total_time:.2f}s")
        
        print(f"\n🌐 Next Steps:")
        print(f"   └─ Try the web interface: streamlit run web_interface.py")
        print(f"   └─ Run the test suite: python test_system.py")
        print(f"   └─ Explore the API: python ragflow_system.py")
        
        print(f"\n✨ RAGFlow demo completed successfully!")
        print(f"   The system is ready for your insurance policy questions!")
        
    except KeyboardInterrupt:
        print(f"\n\n👋 Demo interrupted by user. Goodbye!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        print(f"Please check the logs for more details.")

if __name__ == "__main__":
    main() 