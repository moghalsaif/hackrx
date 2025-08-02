import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

from config import RAGFlowConfig

# Set up logging
logging.basicConfig(level=getattr(logging, RAGFlowConfig.LOG_LEVEL))
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of the insurance policy document with metadata"""
    text: str
    page_number: int
    section_title: str = ""
    clause_number: str = ""
    subsection: str = ""
    chunk_id: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        
        # Generate unique chunk ID
        if not self.chunk_id:
            self.chunk_id = f"page_{self.page_number}_{hash(self.text[:100]) % 10000}"

class DocumentProcessor:
    """Processes insurance policy PDF documents with intelligent chunking"""
    
    def __init__(self, config: RAGFlowConfig = RAGFlowConfig()):
        self.config = config
        self.section_patterns = [
            r'(?i)^(section|clause|part|chapter)\s*(\d+\.?\d*)',
            r'(?i)^(\d+\.?\d*)\s*(section|clause|part|chapter)',
            r'(?i)^(\d+\.?\d*\.?\d*)\s+([A-Z][A-Za-z\s]+):',
            r'(?i)^([A-Z\s]+):',
        ]
        self.clause_patterns = [
            r'(?i)clause\s*(\d+\.?\d*\.?\d*)',
            r'(?i)section\s*(\d+\.?\d*\.?\d*)',
            r'(?i)^\s*(\d+\.?\d*\.?\d*)\s+',
        ]
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF using multiple methods for best results"""
        logger.info(f"Processing PDF: {pdf_path}")
        pages_data = []
        
        try:
            # Method 1: Try pdfplumber first (best for text extraction)
            pages_data = self._extract_with_pdfplumber(pdf_path)
            
            if not pages_data or all(not page['text'].strip() for page in pages_data):
                logger.warning("pdfplumber failed, trying PyMuPDF")
                pages_data = self._extract_with_pymupdf(pdf_path)
            
            if not pages_data or all(not page['text'].strip() for page in pages_data):
                logger.warning("PyMuPDF failed, trying OCR")
                pages_data = self._extract_with_ocr(pdf_path)
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
        
        logger.info(f"Successfully extracted text from {len(pages_data)} pages")
        return pages_data
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text using pdfplumber"""
        pages_data = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                
                # Extract tables if present
                tables = page.extract_tables()
                table_text = ""
                if tables:
                    for table in tables:
                        for row in table:
                            if row:
                                table_text += " | ".join(str(cell) if cell else "" for cell in row) + "\n"
                
                pages_data.append({
                    'page_number': page_num,
                    'text': text,
                    'tables': table_text,
                    'bbox': page.bbox if hasattr(page, 'bbox') else None
                })
        
        return pages_data
    
    def _extract_with_pymupdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text using PyMuPDF"""
        pages_data = []
        
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            
            pages_data.append({
                'page_number': page_num + 1,
                'text': text,
                'tables': "",
                'bbox': page.rect
            })
        
        doc.close()
        return pages_data
    
    def _extract_with_ocr(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text using OCR as fallback"""
        pages_data = []
        
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Convert page to image
            mat = fitz.Matrix(2, 2)  # Increase resolution
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # Apply OCR
            text = pytesseract.image_to_string(img, config='--psm 6')
            
            pages_data.append({
                'page_number': page_num + 1,
                'text': text,
                'tables': "",
                'bbox': page.rect
            })
        
        doc.close()
        return pages_data
    
    def identify_sections_and_clauses(self, text: str) -> Dict[str, str]:
        """Identify section titles and clause numbers from text"""
        metadata = {
            'section_title': '',
            'clause_number': '',
            'subsection': ''
        }
        
        lines = text.split('\n')[:5]  # Check first few lines
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for section patterns
            for pattern in self.section_patterns:
                match = re.search(pattern, line)
                if match:
                    if 'section' in pattern.lower() or 'clause' in pattern.lower():
                        metadata['section_title'] = line
                        if len(match.groups()) >= 1:
                            metadata['clause_number'] = match.group(1)
                    break
            
            # Check for clause patterns
            for pattern in self.clause_patterns:
                match = re.search(pattern, line)
                if match:
                    metadata['clause_number'] = match.group(1)
                    break
        
        return metadata
    
    def intelligent_chunking(self, pages_data: List[Dict[str, Any]]) -> List[DocumentChunk]:
        """Create intelligent chunks preserving policy structure"""
        chunks = []
        current_section = ""
        current_clause = ""
        
        for page_data in pages_data:
            page_num = page_data['page_number']
            text = page_data['text']
            
            if not text.strip():
                continue
            
            # Split by paragraphs first
            paragraphs = self._split_into_paragraphs(text)
            
            for paragraph in paragraphs:
                if len(paragraph.strip()) < self.config.MIN_CHUNK_SIZE:
                    continue
                
                # Identify structure
                structure_info = self.identify_sections_and_clauses(paragraph)
                
                # Update current section/clause tracking
                if structure_info['section_title']:
                    current_section = structure_info['section_title']
                if structure_info['clause_number']:
                    current_clause = structure_info['clause_number']
                
                # Create chunks with size limits
                text_chunks = self._split_by_size(paragraph, self.config.CHUNK_SIZE, self.config.CHUNK_OVERLAP)
                
                for chunk_text in text_chunks:
                    chunk = DocumentChunk(
                        text=chunk_text,
                        page_number=page_num,
                        section_title=current_section,
                        clause_number=current_clause,
                        subsection=structure_info.get('subsection', ''),
                        metadata={
                            'original_pdf': self.config.PDF_FILE,
                            'extraction_method': 'intelligent_chunking',
                            'chunk_length': len(chunk_text),
                            'has_tables': bool(page_data.get('tables')),
                            'bbox': page_data.get('bbox')
                        }
                    )
                    chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} intelligent chunks")
        return chunks
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs while preserving structure"""
        # Split on double newlines, but also on clause/section boundaries
        paragraphs = re.split(r'\n\s*\n', text)
        
        refined_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Further split on numbered items or new clauses
            sub_splits = re.split(r'(?=^\d+\.?\d*\s+)', para, flags=re.MULTILINE)
            refined_paragraphs.extend([p.strip() for p in sub_splits if p.strip()])
        
        return refined_paragraphs
    
    def _split_by_size(self, text: str, max_size: int, overlap: int) -> List[str]:
        """Split text by size with overlap, respecting sentence boundaries"""
        if len(text) <= max_size:
            return [text]
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) <= max_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                if overlap > 0 and chunks:
                    overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    current_chunk = overlap_text + sentence + " "
                else:
                    current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def process_document(self, pdf_path: str = None) -> List[DocumentChunk]:
        """Main method to process the insurance policy document"""
        if pdf_path is None:
            pdf_path = self.config.PDF_FILE
        
        logger.info(f"Starting document processing for {pdf_path}")
        
        # Extract text from PDF
        pages_data = self.extract_text_from_pdf(pdf_path)
        
        # Create intelligent chunks
        chunks = self.intelligent_chunking(pages_data)
        
        # Save processed data for debugging
        if self.config.ENABLE_AUDIT_LOGGING:
            self._save_processing_log(chunks)
        
        logger.info(f"Document processing complete. Generated {len(chunks)} chunks")
        return chunks
    
    def _save_processing_log(self, chunks: List[DocumentChunk]):
        """Save processing results for audit purposes"""
        log_file = self.config.LOGS_DIR / "document_processing.log"
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Document Processing Log - {self.config.PDF_FILE}\n")
            f.write("="*50 + "\n\n")
            
            for i, chunk in enumerate(chunks):
                f.write(f"Chunk {i+1}:\n")
                f.write(f"  ID: {chunk.chunk_id}\n")
                f.write(f"  Page: {chunk.page_number}\n")
                f.write(f"  Section: {chunk.section_title}\n")
                f.write(f"  Clause: {chunk.clause_number}\n")
                f.write(f"  Length: {len(chunk.text)}\n")
                f.write(f"  Text: {chunk.text[:200]}...\n")
                f.write("-"*30 + "\n")

if __name__ == "__main__":
    processor = DocumentProcessor()
    chunks = processor.process_document()
    print(f"Processed document into {len(chunks)} chunks") 