import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json

from config import RAGFlowConfig
from document_processor import DocumentProcessor, DocumentChunk
from embedding_manager import EmbeddingManager
from query_processor import QueryProcessor, ParsedQuery
from llm_reasoner import LLMReasoner, InsuranceDecision

# Set up logging
logging.basicConfig(level=getattr(logging, RAGFlowConfig.LOG_LEVEL))
logger = logging.getLogger(__name__)

class RAGFlowSystem:
    """Main RAGFlow system for insurance policy document analysis"""
    
    def __init__(self, config: RAGFlowConfig = None):
        self.config = config or RAGFlowConfig()
        
        # Initialize components
        self.document_processor = DocumentProcessor(self.config)
        self.embedding_manager = EmbeddingManager(self.config)
        self.query_processor = QueryProcessor(self.config)
        self.llm_reasoner = LLMReasoner(self.config)
        
        # System state
        self.is_initialized = False
        self.chunks = []
        self.system_stats = {}
        
        logger.info("RAGFlow system initialized")
    
    def initialize_system(self, force_reprocess: bool = False) -> bool:
        """Initialize the complete RAG system"""
        logger.info("Starting system initialization...")
        start_time = time.time()
        
        try:
            # Validate configuration
            self.config.validate_config()
            
            # Check if we can load from backup
            if not force_reprocess:
                backup_data = self.embedding_manager.load_embeddings_backup()
                if backup_data:
                    logger.info("Loading from backup...")
                    self.chunks = backup_data['chunks']
                    # Restore to vector database
                    self.embedding_manager.embed_and_store_chunks(self.chunks)
                    self.is_initialized = True
                    logger.info("System initialized from backup successfully")
                    return True
            
            # Full initialization process
            logger.info("Starting full document processing...")
            
            # Step 1: Process the PDF document
            logger.info("Step 1: Processing PDF document...")
            self.chunks = self.document_processor.process_document()
            
            if not self.chunks:
                logger.error("No chunks generated from document processing")
                return False
            
            # Step 2: Generate embeddings and store in vector database
            logger.info("Step 2: Generating embeddings and storing in vector database...")
            success = self.embedding_manager.embed_and_store_chunks(self.chunks)
            
            if not success:
                logger.error("Failed to generate embeddings and store in vector database")
                return False
            
            # Step 3: Save backup for faster future loading
            logger.info("Step 3: Saving embeddings backup...")
            self.embedding_manager.save_embeddings_backup(self.chunks)
            
            # Step 4: Generate system statistics
            self.system_stats = self._generate_system_stats()
            
            self.is_initialized = True
            initialization_time = time.time() - start_time
            
            logger.info(f"System initialization complete in {initialization_time:.2f} seconds")
            self._log_system_stats()
            
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False
    
    def query_system(self, query: str, include_debug_info: bool = False) -> Dict[str, Any]:
        """Main method to query the RAG system"""
        if not self.is_initialized:
            raise RuntimeError("System not initialized. Call initialize_system() first.")
        
        logger.info(f"Processing query: '{query}'")
        start_time = time.time()
        
        try:
            # Step 1: Validate and parse the query
            is_valid, validation_msg = self.query_processor.validate_query(query)
            if not is_valid:
                return self._create_error_response(f"Invalid query: {validation_msg}")
            
            parsed_query = self.query_processor.parse_query(query)
            logger.info(f"Query parsed - Type: {parsed_query.query_type}, Confidence: {parsed_query.confidence:.2f}")
            
            # Step 2: Retrieve relevant chunks
            retrieved_chunks = self._retrieve_relevant_chunks(parsed_query)
            logger.info(f"Retrieved {len(retrieved_chunks)} relevant chunks")
            
            if not retrieved_chunks:
                return self._create_no_results_response(parsed_query)
            
            # Step 3: Generate LLM response
            decision = self.llm_reasoner.analyze_insurance_query(parsed_query, retrieved_chunks)
            
            # Step 4: Prepare final response
            processing_time = time.time() - start_time
            response = self._create_final_response(
                decision, parsed_query, retrieved_chunks, processing_time, include_debug_info
            )
            
            # Log the interaction
            self._log_interaction(query, decision, processing_time)
            
            logger.info(f"Query processed successfully in {processing_time:.2f} seconds")
            return response
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return self._create_error_response(f"System error: {str(e)}")
    
    def _retrieve_relevant_chunks(self, parsed_query: ParsedQuery) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks using multiple strategies"""
        all_chunks = []
        
        # Primary search with main query
        primary_chunks = self.embedding_manager.search_similar_chunks(
            parsed_query.processed_query,
            top_k=self.config.TOP_K_RETRIEVAL,
            filters=parsed_query.filters
        )
        all_chunks.extend(primary_chunks)
        
        # Expand search with additional terms if enabled
        if self.config.ENABLE_QUERY_EXPANSION and parsed_query.search_terms:
            for search_term in parsed_query.search_terms[:3]:  # Limit to top 3 terms
                if search_term != parsed_query.processed_query:
                    additional_chunks = self.embedding_manager.search_similar_chunks(
                        search_term,
                        top_k=2,  # Fewer results for additional terms
                        filters=parsed_query.filters
                    )
                    all_chunks.extend(additional_chunks)
        
        # Remove duplicates based on chunk text
        seen_texts = set()
        unique_chunks = []
        for chunk in all_chunks:
            if chunk['text'] not in seen_texts:
                seen_texts.add(chunk['text'])
                unique_chunks.append(chunk)
        
        # Sort by similarity score and limit results
        unique_chunks.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        return unique_chunks[:self.config.TOP_K_RETRIEVAL]
    
    def _create_final_response(self, decision: InsuranceDecision, parsed_query: ParsedQuery,
                              retrieved_chunks: List[Dict[str, Any]], processing_time: float,
                              include_debug_info: bool) -> Dict[str, Any]:
        """Create the final structured response"""
        response = {
            "decision": decision.decision,
            "justification": decision.justification,
            "references": decision.references,
            "confidence_score": decision.confidence_score,
            "processing_time_seconds": round(processing_time, 2),
            "system_metadata": {
                "query_type": decision.query_type,
                "retrieved_chunks_count": decision.retrieved_chunks_count,
                "system_version": "1.0.0",
                "policy_document": self.config.PDF_FILE
            }
        }
        
        if include_debug_info:
            response["debug_info"] = {
                "parsed_query": {
                    "original": parsed_query.original_query,
                    "processed": parsed_query.processed_query,
                    "entities": parsed_query.entities,
                    "search_terms": parsed_query.search_terms,
                    "parsing_confidence": parsed_query.confidence
                },
                "retrieved_chunks": [
                    {
                        "rank": chunk.get("rank", 0),
                        "similarity_score": chunk.get("similarity_score", 0),
                        "page_number": chunk.get("page_number", 0),
                        "section_title": chunk.get("section_title", ""),
                        "clause_number": chunk.get("clause_number", ""),
                        "text_preview": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"]
                    }
                    for chunk in retrieved_chunks
                ],
                "system_stats": self.system_stats
            }
        
        return response
    
    def _create_error_response(self, error_msg: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            "decision": "error",
            "justification": error_msg,
            "references": [],
            "confidence_score": 0.0,
            "processing_time_seconds": 0.0,
            "system_metadata": {
                "query_type": "error",
                "retrieved_chunks_count": 0,
                "system_version": "1.0.0",
                "policy_document": self.config.PDF_FILE
            }
        }
    
    def _create_no_results_response(self, parsed_query: ParsedQuery) -> Dict[str, Any]:
        """Create response when no relevant chunks are found"""
        return {
            "decision": "uncertain",
            "justification": "No relevant policy information found for this query. The query may be outside the scope of this policy document or may require human review.",
            "references": [],
            "confidence_score": 0.1,
            "processing_time_seconds": 0.0,
            "system_metadata": {
                "query_type": parsed_query.query_type,
                "retrieved_chunks_count": 0,
                "system_version": "1.0.0",
                "policy_document": self.config.PDF_FILE
            }
        }
    
    def _generate_system_stats(self) -> Dict[str, Any]:
        """Generate system statistics"""
        try:
            embedding_stats = self.embedding_manager.get_collection_stats()
            
            stats = {
                "total_chunks": len(self.chunks),
                "avg_chunk_length": sum(len(chunk.text) for chunk in self.chunks) / len(self.chunks) if self.chunks else 0,
                "unique_pages": len(set(chunk.page_number for chunk in self.chunks)),
                "unique_sections": len(set(chunk.section_title for chunk in self.chunks if chunk.section_title)),
                "unique_clauses": len(set(chunk.clause_number for chunk in self.chunks if chunk.clause_number)),
                "vector_db_stats": embedding_stats,
                "config": {
                    "chunk_size": self.config.CHUNK_SIZE,
                    "embedding_model": self.config.EMBEDDING_MODEL,
                    "llm_model": self.config.LLM_MODEL,
                    "similarity_threshold": self.config.SIMILARITY_THRESHOLD
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to generate system stats: {e}")
            return {"error": str(e)}
    
    def _log_system_stats(self):
        """Log system statistics"""
        logger.info("System Statistics:")
        logger.info(f"  Total chunks: {self.system_stats.get('total_chunks', 0)}")
        logger.info(f"  Average chunk length: {self.system_stats.get('avg_chunk_length', 0):.1f}")
        logger.info(f"  Unique pages: {self.system_stats.get('unique_pages', 0)}")
        logger.info(f"  Unique sections: {self.system_stats.get('unique_sections', 0)}")
        logger.info(f"  Unique clauses: {self.system_stats.get('unique_clauses', 0)}")
    
    def _log_interaction(self, query: str, decision: InsuranceDecision, processing_time: float):
        """Log user interaction for audit purposes"""
        if self.config.ENABLE_AUDIT_LOGGING:
            log_entry = {
                "timestamp": time.time(),
                "query": query,
                "decision": decision.decision,
                "confidence": decision.confidence_score,
                "processing_time": processing_time,
                "references_count": len(decision.references)
            }
            
            log_file = self.config.LOGS_DIR / "interactions.jsonl"
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "is_initialized": self.is_initialized,
            "chunks_loaded": len(self.chunks),
            "system_stats": self.system_stats,
            "config": {
                "pdf_file": self.config.PDF_FILE,
                "embedding_model": self.config.EMBEDDING_MODEL,
                "llm_model": self.config.LLM_MODEL,
                "vector_db_type": self.config.VECTOR_DB_TYPE
            }
        }
    
    def reset_system(self) -> bool:
        """Reset the entire system"""
        logger.warning("Resetting RAG system...")
        
        try:
            # Reset vector database
            self.embedding_manager.reset_collection()
            
            # Clear chunks
            self.chunks = []
            self.system_stats = {}
            self.is_initialized = False
            
            logger.info("System reset successfully")
            return True
            
        except Exception as e:
            logger.error(f"System reset failed: {e}")
            return False

def main():
    """Main function for testing the RAG system"""
    print("🚀 Initializing RAGFlow Insurance Policy System...")
    
    # Initialize system
    rag_system = RAGFlowSystem()
    
    # Check if system can be initialized
    if not rag_system.initialize_system():
        print("❌ Failed to initialize system")
        return
    
    print("✅ System initialized successfully!")
    
    # Test queries
    test_queries = [
        "Is ACL reconstruction surgery covered for a 46-year-old under a 3-month-old policy?",
        "What is the waiting period for cardiac surgery?",
        "Are dental treatments excluded from coverage?",
        "What are the coverage limits for knee surgery in Pune?",
        "Is maternity coverage available for 28-year-old members?"
    ]
    
    print("\n🔍 Testing with sample queries...\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}: {query}")
        print("-" * 60)
        
        response = rag_system.query_system(query, include_debug_info=False)
        
        print(f"Decision: {response['decision']}")
        print(f"Confidence: {response['confidence_score']:.2f}")
        print(f"References: {len(response['references'])}")
        print(f"Justification: {response['justification'][:200]}...")
        print(f"Processing Time: {response['processing_time_seconds']}s")
        print("\n")
    
    # Get system status
    status = rag_system.get_system_status()
    print(f"📊 System Status: {status['chunks_loaded']} chunks loaded")

if __name__ == "__main__":
    main() 