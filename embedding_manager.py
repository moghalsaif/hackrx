import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm

from config import RAGFlowConfig
from document_processor import DocumentChunk

# Set up logging
logging.basicConfig(level=getattr(logging, RAGFlowConfig.LOG_LEVEL))
logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages embedding generation and vector storage for insurance policy chunks"""
    
    def __init__(self, config: RAGFlowConfig = RAGFlowConfig()):
        self.config = config
        self.embedding_model = None
        self.vector_db = None
        self.collection = None
        self._initialize_embedding_model()
        self._initialize_vector_database()
    
    def _initialize_embedding_model(self):
        """Initialize the sentence transformer model"""
        logger.info(f"Loading embedding model: {self.config.EMBEDDING_MODEL}")
        try:
            self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _initialize_vector_database(self):
        """Initialize the vector database (Chroma)"""
        logger.info("Initializing vector database...")
        
        try:
            if self.config.VECTOR_DB_TYPE == "chroma":
                # Create Chroma client with persistent storage
                chroma_settings = Settings(
                    persist_directory=str(self.config.VECTOR_DB_DIR),
                    allow_reset=True
                )
                self.vector_db = chromadb.Client(chroma_settings)
                
                # Get or create collection
                try:
                    self.collection = self.vector_db.get_collection(
                        name=self.config.COLLECTION_NAME
                    )
                    logger.info(f"Using existing collection: {self.config.COLLECTION_NAME}")
                except:
                    self.collection = self.vector_db.create_collection(
                        name=self.config.COLLECTION_NAME,
                        metadata={"description": "Insurance policy document chunks"}
                    )
                    logger.info(f"Created new collection: {self.config.COLLECTION_NAME}")
            
            else:
                raise ValueError(f"Unsupported vector database type: {self.config.VECTOR_DB_TYPE}")
                
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch, convert_to_numpy=True)
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def embed_and_store_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Generate embeddings for chunks and store in vector database"""
        if not chunks:
            logger.warning("No chunks provided for embedding")
            return False
        
        logger.info(f"Processing {len(chunks)} chunks for embedding and storage")
        
        try:
            # Prepare data for embedding
            texts = [chunk.text for chunk in chunks]
            chunk_ids = [chunk.chunk_id for chunk in chunks]
            
            # Generate embeddings
            embeddings = self.generate_embeddings(texts)
            
            # Prepare metadata for storage
            metadatas = []
            for chunk in chunks:
                metadata = {
                    "page_number": chunk.page_number,
                    "section_title": chunk.section_title,
                    "clause_number": chunk.clause_number,
                    "subsection": chunk.subsection,
                    "text_length": len(chunk.text),
                    "original_pdf": self.config.PDF_FILE,
                    **chunk.metadata
                }
                # Convert all values to strings for Chroma compatibility
                metadata = {k: str(v) for k, v in metadata.items()}
                metadatas.append(metadata)
            
            # Store in vector database
            self._store_in_vector_db(
                embeddings=embeddings.tolist(),
                texts=texts,
                ids=chunk_ids,
                metadatas=metadatas
            )
            
            logger.info(f"Successfully stored {len(chunks)} chunks in vector database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to embed and store chunks: {e}")
            return False
    
    def _store_in_vector_db(self, embeddings: List[List[float]], texts: List[str], 
                           ids: List[str], metadatas: List[Dict[str, str]]):
        """Store embeddings in the vector database"""
        try:
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                ids=ids,
                metadatas=metadatas
            )
            
            # Persist the database
            if hasattr(self.vector_db, 'persist'):
                self.vector_db.persist()
                
        except Exception as e:
            logger.error(f"Failed to store in vector database: {e}")
            raise
    
    def search_similar_chunks(self, query: str, top_k: Optional[int] = None, 
                            filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar chunks based on query"""
        if top_k is None:
            top_k = self.config.TOP_K_RETRIEVAL
        
        logger.info(f"Searching for similar chunks: '{query[:100]}...' (top_k={top_k})")
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]
            
            # Prepare where clause for filtering
            where_clause = None
            if filters:
                where_clause = self._build_where_clause(filters)
            
            # Search in vector database
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=where_clause,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Calculate similarity score (1 - normalized distance)
                    similarity_score = 1 - distance
                    
                    # Only return results above similarity threshold
                    if similarity_score >= self.config.SIMILARITY_THRESHOLD:
                        formatted_results.append({
                            'rank': i + 1,
                            'text': doc,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'metadata': metadata,
                            'page_number': int(metadata.get('page_number', 0)),
                            'section_title': metadata.get('section_title', ''),
                            'clause_number': metadata.get('clause_number', ''),
                            'subsection': metadata.get('subsection', '')
                        })
            
            logger.info(f"Found {len(formatted_results)} relevant chunks above threshold")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _build_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build where clause for Chroma filtering"""
        where_clause = {}
        
        for key, value in filters.items():
            if isinstance(value, list):
                where_clause[key] = {"$in": value}
            elif isinstance(value, dict):
                where_clause[key] = value
            else:
                where_clause[key] = {"$eq": str(value)}
        
        return where_clause
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector collection"""
        try:
            count = self.collection.count()
            
            # Get sample documents to analyze
            sample_results = self.collection.peek(limit=10)
            
            stats = {
                "total_chunks": count,
                "collection_name": self.config.COLLECTION_NAME,
                "embedding_model": self.config.EMBEDDING_MODEL,
                "embedding_dimension": self.config.EMBEDDING_DIMENSION,
                "vector_db_type": self.config.VECTOR_DB_TYPE
            }
            
            if sample_results['metadatas']:
                # Extract metadata statistics
                pages = set()
                sections = set()
                clauses = set()
                
                for metadata in sample_results['metadatas']:
                    if metadata.get('page_number'):
                        pages.add(metadata['page_number'])
                    if metadata.get('section_title'):
                        sections.add(metadata['section_title'])
                    if metadata.get('clause_number'):
                        clauses.add(metadata['clause_number'])
                
                stats.update({
                    "sample_pages_count": len(pages),
                    "sample_sections_count": len(sections),
                    "sample_clauses_count": len(clauses)
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}
    
    def reset_collection(self) -> bool:
        """Reset the collection (delete all data)"""
        try:
            logger.warning("Resetting vector collection - all data will be deleted")
            self.vector_db.delete_collection(self.config.COLLECTION_NAME)
            
            # Recreate collection
            self.collection = self.vector_db.create_collection(
                name=self.config.COLLECTION_NAME,
                metadata={"description": "Insurance policy document chunks"}
            )
            
            logger.info("Collection reset successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            return False
    
    def save_embeddings_backup(self, chunks: List[DocumentChunk], 
                              backup_path: Optional[str] = None) -> bool:
        """Save embeddings as backup for faster reloading"""
        if backup_path is None:
            backup_path = self.config.DATA_DIR / "embeddings_backup.pkl"
        
        try:
            logger.info(f"Saving embeddings backup to {backup_path}")
            
            texts = [chunk.text for chunk in chunks]
            embeddings = self.generate_embeddings(texts)
            
            backup_data = {
                'embeddings': embeddings,
                'chunks': chunks,
                'model_name': self.config.EMBEDDING_MODEL,
                'config': {
                    'chunk_size': self.config.CHUNK_SIZE,
                    'chunk_overlap': self.config.CHUNK_OVERLAP,
                    'embedding_dimension': self.config.EMBEDDING_DIMENSION
                }
            }
            
            with open(backup_path, 'wb') as f:
                pickle.dump(backup_data, f)
            
            logger.info("Embeddings backup saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save embeddings backup: {e}")
            return False
    
    def load_embeddings_backup(self, backup_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Load embeddings from backup"""
        if backup_path is None:
            backup_path = self.config.DATA_DIR / "embeddings_backup.pkl"
        
        if not Path(backup_path).exists():
            logger.warning(f"Backup file not found: {backup_path}")
            return None
        
        try:
            logger.info(f"Loading embeddings backup from {backup_path}")
            
            with open(backup_path, 'rb') as f:
                backup_data = pickle.load(f)
            
            # Validate backup compatibility
            if backup_data['model_name'] != self.config.EMBEDDING_MODEL:
                logger.warning(f"Model mismatch: backup={backup_data['model_name']}, current={self.config.EMBEDDING_MODEL}")
                return None
            
            logger.info("Embeddings backup loaded successfully")
            return backup_data
            
        except Exception as e:
            logger.error(f"Failed to load embeddings backup: {e}")
            return None

if __name__ == "__main__":
    # Test the embedding manager
    from document_processor import DocumentProcessor
    
    processor = DocumentProcessor()
    chunks = processor.process_document()
    
    embedding_manager = EmbeddingManager()
    success = embedding_manager.embed_and_store_chunks(chunks)
    
    if success:
        print("Testing search...")
        results = embedding_manager.search_similar_chunks("knee surgery coverage")
        print(f"Found {len(results)} relevant chunks")
        
        stats = embedding_manager.get_collection_stats()
        print(f"Collection stats: {stats}") 