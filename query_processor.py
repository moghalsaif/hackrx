import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import spacy
from datetime import datetime, timedelta

from config import RAGFlowConfig

# Set up logging
logging.basicConfig(level=getattr(logging, RAGFlowConfig.LOG_LEVEL))
logger = logging.getLogger(__name__)

@dataclass
class ParsedQuery:
    """Represents a parsed insurance policy query with extracted entities"""
    original_query: str
    processed_query: str
    entities: Dict[str, Any]
    query_type: str
    confidence: float
    search_terms: List[str]
    filters: Dict[str, Any]

class QueryProcessor:
    """Processes natural language queries about insurance policies"""
    
    def __init__(self, config: RAGFlowConfig = RAGFlowConfig()):
        self.config = config
        self.nlp = None
        self._initialize_nlp()
        
        # Insurance-specific patterns and entities
        self.medical_procedures = [
            "acl reconstruction", "knee surgery", "cardiac surgery", "bypass surgery",
            "cataract surgery", "hip replacement", "orthopedic surgery", "dental treatment",
            "physiotherapy", "chemotherapy", "dialysis", "angioplasty", "endoscopy",
            "mri scan", "ct scan", "x-ray", "ultrasound", "blood test", "vaccination"
        ]
        
        self.policy_terms = [
            "coverage", "exclusion", "deductible", "premium", "claim", "benefit",
            "waiting period", "pre-existing condition", "maternity", "dental", "optical",
            "emergency", "hospitalization", "outpatient", "inpatient", "copayment"
        ]
        
        self.age_pattern = r'(?i)(?:age|aged|years?\s+old|\d+\s*[-\s]*year)\s*:?\s*(\d+)'
        self.amount_pattern = r'(?i)(?:cost|amount|price|bill|expense)\s*:?\s*(?:rs\.?|₹|inr)?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)'
        self.location_pattern = r'(?i)(?:in|at|from|location|city)\s+([A-Za-z\s]+(?:,\s*[A-Za-z\s]+)*)'
        self.time_pattern = r'(?i)(\d+)\s*(?:days?|weeks?|months?|years?)\s*(?:old|ago|after|since)'
    
    def _initialize_nlp(self):
        """Initialize spaCy NLP model"""
        try:
            logger.info("Loading spaCy NLP model...")
            # Try to load the model, fallback to basic if not available
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("en_core_web_sm not found, using basic tokenizer")
                self.nlp = spacy.blank("en")
            logger.info("NLP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize NLP model: {e}")
            self.nlp = None
    
    def parse_query(self, query: str) -> ParsedQuery:
        """Parse a natural language query into structured components"""
        logger.info(f"Parsing query: '{query}'")
        
        # Clean and preprocess query
        processed_query = self._preprocess_query(query)
        
        # Extract entities
        entities = self._extract_entities(processed_query)
        
        # Determine query type
        query_type = self._classify_query_type(processed_query, entities)
        
        # Generate search terms
        search_terms = self._generate_search_terms(processed_query, entities)
        
        # Create filters for retrieval
        filters = self._create_filters(entities)
        
        # Calculate confidence
        confidence = self._calculate_confidence(entities, search_terms)
        
        parsed_query = ParsedQuery(
            original_query=query,
            processed_query=processed_query,
            entities=entities,
            query_type=query_type,
            confidence=confidence,
            search_terms=search_terms,
            filters=filters
        )
        
        logger.info(f"Query parsed successfully - Type: {query_type}, Confidence: {confidence:.2f}")
        return parsed_query
    
    def _preprocess_query(self, query: str) -> str:
        """Clean and preprocess the query text"""
        # Convert to lowercase
        processed = query.lower().strip()
        
        # Remove extra whitespace
        processed = re.sub(r'\s+', ' ', processed)
        
        # Normalize common abbreviations
        replacements = {
            'acl': 'anterior cruciate ligament',
            'mri': 'magnetic resonance imaging',
            'ct': 'computed tomography',
            'icu': 'intensive care unit',
            'ot': 'operation theater',
            'opd': 'outpatient department',
            'ipd': 'inpatient department'
        }
        
        for abbr, full_form in replacements.items():
            processed = re.sub(rf'\b{abbr}\b', full_form, processed)
        
        return processed
    
    def _extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract structured entities from the query"""
        entities = {
            'age': None,
            'procedure': [],
            'location': None,
            'amount': None,
            'time_period': None,
            'policy_terms': [],
            'body_parts': [],
            'medical_conditions': []
        }
        
        # Extract age
        age_match = re.search(self.age_pattern, query)
        if age_match:
            entities['age'] = int(age_match.group(1))
        
        # Extract amount
        amount_match = re.search(self.amount_pattern, query)
        if amount_match:
            entities['amount'] = amount_match.group(1).replace(',', '')
        
        # Extract location
        location_match = re.search(self.location_pattern, query)
        if location_match:
            entities['location'] = location_match.group(1).strip()
        
        # Extract time period
        time_match = re.search(self.time_pattern, query)
        if time_match:
            entities['time_period'] = {
                'value': int(time_match.group(1)),
                'unit': time_match.group(0).split()[-1] if len(time_match.group(0).split()) > 1 else 'days'
            }
        
        # Extract medical procedures
        for procedure in self.medical_procedures:
            if procedure in query:
                entities['procedure'].append(procedure)
        
        # Extract policy terms
        for term in self.policy_terms:
            if term in query:
                entities['policy_terms'].append(term)
        
        # Extract body parts using NLP if available
        if self.nlp:
            doc = self.nlp(query)
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'GPE']:  # Organizations, places
                    if not entities['location']:
                        entities['location'] = ent.text
                elif ent.label_ in ['MONEY']:
                    if not entities['amount']:
                        entities['amount'] = ent.text
        
        # Extract body parts and medical conditions
        body_parts = ['knee', 'hip', 'shoulder', 'heart', 'liver', 'kidney', 'brain', 'eye', 'tooth', 'bone']
        medical_conditions = ['diabetes', 'hypertension', 'cancer', 'arthritis', 'asthma', 'cardiac']
        
        for part in body_parts:
            if part in query:
                entities['body_parts'].append(part)
        
        for condition in medical_conditions:
            if condition in query:
                entities['medical_conditions'].append(condition)
        
        return entities
    
    def _classify_query_type(self, query: str, entities: Dict[str, Any]) -> str:
        """Classify the type of query"""
        coverage_keywords = ['cover', 'coverage', 'covered', 'eligible', 'benefit']
        exclusion_keywords = ['exclude', 'exclusion', 'not covered', 'denied']
        claim_keywords = ['claim', 'reimbursement', 'settlement', 'payment']
        waiting_keywords = ['waiting', 'wait', 'period', 'delay']
        limit_keywords = ['limit', 'maximum', 'minimum', 'cap']
        
        if any(keyword in query for keyword in coverage_keywords):
            return 'coverage_inquiry'
        elif any(keyword in query for keyword in exclusion_keywords):
            return 'exclusion_inquiry'
        elif any(keyword in query for keyword in claim_keywords):
            return 'claim_inquiry'
        elif any(keyword in query for keyword in waiting_keywords):
            return 'waiting_period_inquiry'
        elif any(keyword in query for keyword in limit_keywords):
            return 'limit_inquiry'
        else:
            return 'general_inquiry'
    
    def _generate_search_terms(self, query: str, entities: Dict[str, Any]) -> List[str]:
        """Generate search terms for semantic retrieval"""
        search_terms = []
        
        # Add original query
        search_terms.append(query)
        
        # Add procedures
        if entities['procedure']:
            search_terms.extend(entities['procedure'])
        
        # Add policy terms
        if entities['policy_terms']:
            search_terms.extend(entities['policy_terms'])
        
        # Add body parts and conditions
        if entities['body_parts']:
            search_terms.extend(entities['body_parts'])
        
        if entities['medical_conditions']:
            search_terms.extend(entities['medical_conditions'])
        
        # Generate combined terms
        if entities['procedure'] and entities['body_parts']:
            for proc in entities['procedure']:
                for part in entities['body_parts']:
                    search_terms.append(f"{part} {proc}")
        
        # Add age-specific terms if age is mentioned
        if entities['age']:
            age = entities['age']
            if age < 18:
                search_terms.append("pediatric coverage")
            elif age > 60:
                search_terms.append("senior citizen coverage")
        
        # Remove duplicates and empty terms
        search_terms = list(set([term.strip() for term in search_terms if term.strip()]))
        
        return search_terms
    
    def _create_filters(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Create filters for vector database search"""
        filters = {}
        
        # No specific metadata filtering for now
        # This can be extended based on how metadata is structured
        
        return filters
    
    def _calculate_confidence(self, entities: Dict[str, Any], search_terms: List[str]) -> float:
        """Calculate confidence score for the parsed query"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on extracted entities
        if entities['procedure']:
            confidence += 0.2
        
        if entities['age']:
            confidence += 0.1
        
        if entities['policy_terms']:
            confidence += 0.1
        
        if entities['body_parts']:
            confidence += 0.1
        
        if len(search_terms) > 3:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def expand_query(self, parsed_query: ParsedQuery) -> List[str]:
        """Expand query with related terms and synonyms"""
        if not self.config.ENABLE_QUERY_EXPANSION:
            return [parsed_query.processed_query]
        
        expanded_queries = [parsed_query.processed_query]
        
        # Add synonyms and related terms
        expansions = {
            'knee': ['patella', 'meniscus', 'ligament'],
            'surgery': ['operation', 'procedure', 'treatment'],
            'coverage': ['benefit', 'protection', 'insurance'],
            'exclusion': ['not covered', 'denied', 'rejected'],
            'hospitalization': ['admission', 'inpatient', 'hospital stay'],
            'emergency': ['urgent', 'critical', 'immediate care']
        }
        
        for entity_list in [parsed_query.entities['procedure'], 
                           parsed_query.entities['policy_terms'],
                           parsed_query.entities['body_parts']]:
            for entity in entity_list:
                if entity in expansions:
                    for synonym in expansions[entity]:
                        expanded_query = parsed_query.processed_query.replace(entity, synonym)
                        if expanded_query not in expanded_queries:
                            expanded_queries.append(expanded_query)
        
        logger.info(f"Expanded query into {len(expanded_queries)} variations")
        return expanded_queries
    
    def validate_query(self, query: str) -> Tuple[bool, str]:
        """Validate if the query is suitable for insurance policy search"""
        if not query or len(query.strip()) < 3:
            return False, "Query too short"
        
        if len(query) > self.config.MAX_QUERY_LENGTH:
            return False, f"Query too long (max {self.config.MAX_QUERY_LENGTH} characters)"
        
        # Check if query contains insurance-relevant terms
        insurance_keywords = self.medical_procedures + self.policy_terms
        has_insurance_context = any(keyword in query.lower() for keyword in insurance_keywords)
        
        if not has_insurance_context:
            return False, "Query does not appear to be insurance-related"
        
        return True, "Valid query"

if __name__ == "__main__":
    # Test the query processor
    processor = QueryProcessor()
    
    test_queries = [
        "Is ACL reconstruction surgery covered for a 46-year-old under a 3-month-old policy?",
        "knee surgery coverage Pune, 3 months",
        "What is the waiting period for cardiac surgery?",
        "dental treatment exclusions",
        "maternity benefits for 28 year old"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        parsed = processor.parse_query(query)
        print(f"Type: {parsed.query_type}")
        print(f"Entities: {parsed.entities}")
        print(f"Search terms: {parsed.search_terms}")
        print(f"Confidence: {parsed.confidence}")
        
        expanded = processor.expand_query(parsed)
        print(f"Expanded queries: {len(expanded)}") 