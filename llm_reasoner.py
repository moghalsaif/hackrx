import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import openai
import anthropic
import torch
from pathlib import Path

# Local model imports
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import ollama
    TRANSFORMERS_AVAILABLE = True
    OLLAMA_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    OLLAMA_AVAILABLE = False
    print(f"Warning: Some local model dependencies not available: {e}")

from config import RAGFlowConfig
from query_processor import ParsedQuery

# Set up logging
logging.basicConfig(level=getattr(logging, RAGFlowConfig.LOG_LEVEL))
logger = logging.getLogger(__name__)

@dataclass
class InsuranceDecision:
    """Represents an insurance coverage decision with supporting evidence"""
    decision: str
    justification: str
    references: List[Dict[str, Any]]
    confidence_score: float
    query_type: str
    retrieved_chunks_count: int
    
    def to_json(self) -> str:
        """Convert decision to JSON format"""
        return json.dumps({
            "decision": self.decision,
            "justification": self.justification,
            "references": self.references,
            "confidence_score": self.confidence_score,
            "metadata": {
                "query_type": self.query_type,
                "retrieved_chunks_count": self.retrieved_chunks_count
            }
        }, indent=2)

class LocalModelManager:
    """Manages local Llama model loading and inference"""
    
    def __init__(self, config: RAGFlowConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.ollama_client = None
        
    def load_model(self) -> bool:
        """Load the specified local model"""
        model_config = self.config.get_model_config()
        
        try:
            if model_config["local_type"] == "transformers":
                return self._load_transformers_model(model_config)
            elif model_config["local_type"] == "ollama":
                return self._setup_ollama_client(model_config)
            else:
                logger.error(f"Unsupported local model type: {model_config['local_type']}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            return False
    
    def _load_transformers_model(self, model_config: Dict[str, Any]) -> bool:
        """Load model using Transformers library"""
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers library not available")
            return False
        
        model_path = model_config["local_path"]
        
        if not Path(model_path).exists():
            logger.error(f"Model path does not exist: {model_path}")
            logger.info("Please download Llama 3 model files to the specified path")
            return False
        
        logger.info(f"Loading Llama 3 model from {model_path}...")
        
        try:
            # Determine device and precision settings
            device_map = "auto" if model_config["use_gpu"] and torch.cuda.is_available() else "cpu"
            
            torch_dtype = torch.float16
            if model_config["precision"] == "fp32":
                torch_dtype = torch.float32
            elif model_config["precision"] == "int8":
                torch_dtype = torch.int8
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Add pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimizations
            load_kwargs = {
                "torch_dtype": torch_dtype,
                "device_map": device_map,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            
            # Add quantization if specified
            if model_config["precision"] in ["int8", "int4"]:
                load_kwargs["load_in_8bit"] = model_config["precision"] == "int8"
                load_kwargs["load_in_4bit"] = model_config["precision"] == "int4"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **load_kwargs
            )
            
            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch_dtype,
                device_map=device_map,
                return_full_text=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            logger.info("Llama 3 model loaded successfully using Transformers")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Transformers model: {e}")
            return False
    
    def _setup_ollama_client(self, model_config: Dict[str, Any]) -> bool:
        """Setup Ollama client for local inference"""
        if not OLLAMA_AVAILABLE:
            logger.error("Ollama client not available")
            return False
        
        try:
            # Initialize Ollama client
            self.ollama_client = ollama.Client(host=model_config["ollama_url"])
            
            # Check if model is available
            models = self.ollama_client.list()
            model_name = model_config["ollama_model"]
            
            if not any(model["name"] == model_name for model in models.get("models", [])):
                logger.error(f"Ollama model '{model_name}' not found")
                logger.info(f"Available models: {[m['name'] for m in models.get('models', [])]}")
                logger.info(f"To install Llama 3: ollama pull {model_name}")
                return False
            
            logger.info(f"Ollama client setup successfully with model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup Ollama client: {e}")
            return False
    
    def generate_response(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.1) -> str:
        """Generate response using the loaded model"""
        try:
            if self.pipeline:
                return self._generate_with_transformers(prompt, max_tokens, temperature)
            elif self.ollama_client:
                return self._generate_with_ollama(prompt, max_tokens, temperature)
            else:
                raise RuntimeError("No model loaded")
                
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise
    
    def _generate_with_transformers(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate response using Transformers pipeline"""
        try:
            # Generate response
            outputs = self.pipeline(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                num_return_sequences=1
            )
            
            return outputs[0]["generated_text"].strip()
            
        except Exception as e:
            logger.error(f"Transformers generation failed: {e}")
            raise
    
    def _generate_with_ollama(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate response using Ollama client"""
        try:
            response = self.ollama_client.generate(
                model=self.config.OLLAMA_MODEL_NAME,
                prompt=prompt,
                options={
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            )
            
            return response["response"].strip()
            
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

class LLMReasoner:
    """Uses LLM to reason about insurance queries and generate structured responses"""
    
    def __init__(self, config: RAGFlowConfig = RAGFlowConfig()):
        self.config = config
        self.llm_client = None
        self.local_model_manager = None
        self._initialize_llm()
        
        # System prompts for different query types
        self.system_prompts = {
            'coverage_inquiry': self._get_coverage_system_prompt(),
            'exclusion_inquiry': self._get_exclusion_system_prompt(),
            'claim_inquiry': self._get_claim_system_prompt(),
            'waiting_period_inquiry': self._get_waiting_period_system_prompt(),
            'limit_inquiry': self._get_limit_system_prompt(),
            'general_inquiry': self._get_general_system_prompt()
        }
    
    def _initialize_llm(self):
        """Initialize the LLM based on configuration"""
        logger.info(f"Initializing LLM: {self.config.LLM_PROVIDER}")
        
        try:
            if self.config.LLM_PROVIDER == "openai":
                api_key = self.config.get_openai_api_key()
                if not api_key:
                    raise ValueError("OpenAI API key not found")
                self.llm_client = openai.OpenAI(api_key=api_key)
                
            elif self.config.LLM_PROVIDER == "anthropic":
                api_key = self.config.get_anthropic_api_key()
                if not api_key:
                    raise ValueError("Anthropic API key not found")
                self.llm_client = anthropic.Anthropic(api_key=api_key)
                
            elif self.config.LLM_PROVIDER in ["local", "ollama"]:
                self.local_model_manager = LocalModelManager(self.config)
                if not self.local_model_manager.load_model():
                    raise RuntimeError("Failed to load local model")
                
            else:
                raise ValueError(f"Unsupported LLM provider: {self.config.LLM_PROVIDER}")
                
            logger.info("LLM initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def analyze_insurance_query(self, parsed_query: ParsedQuery, 
                              retrieved_chunks: List[Dict[str, Any]]) -> InsuranceDecision:
        """Analyze insurance query using LLM reasoning"""
        logger.info(f"Analyzing insurance query: {parsed_query.query_type}")
        
        if not retrieved_chunks:
            return self._create_no_data_response(parsed_query)
        
        try:
            # Prepare context from retrieved chunks
            context = self._prepare_context(retrieved_chunks)
            
            # Get appropriate system prompt
            system_prompt = self.system_prompts.get(
                parsed_query.query_type, 
                self.system_prompts['general_inquiry']
            )
            
            # Create user prompt
            user_prompt = self._create_user_prompt(parsed_query, context)
            
            # Get LLM response
            llm_response = self._query_llm(system_prompt, user_prompt)
            
            # Parse and structure the response
            decision = self._parse_llm_response(llm_response, parsed_query, retrieved_chunks)
            
            logger.info(f"Analysis complete: {decision.decision}")
            return decision
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return self._create_error_response(parsed_query, str(e))
    
    def _prepare_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Prepare context from retrieved chunks for LLM input"""
        context_parts = []
        
        for i, chunk in enumerate(retrieved_chunks, 1):
            chunk_text = f"""
CHUNK {i}:
Page: {chunk.get('page_number', 'Unknown')}
Section: {chunk.get('section_title', 'Unknown')}
Clause: {chunk.get('clause_number', 'Unknown')}
Similarity Score: {chunk.get('similarity_score', 0):.3f}

Content: {chunk['text']}

---
"""
            context_parts.append(chunk_text)
        
        return "\n".join(context_parts)
    
    def _create_user_prompt(self, parsed_query: ParsedQuery, context: str) -> str:
        """Create user prompt combining query and context"""
        prompt = f"""
INSURANCE POLICY QUERY:
{parsed_query.original_query}

QUERY ANALYSIS:
- Type: {parsed_query.query_type}
- Extracted Entities: {json.dumps(parsed_query.entities, indent=2)}
- Confidence: {parsed_query.confidence:.2f}

RELEVANT POLICY EXCERPTS:
{context}

Please analyze this query against the provided policy excerpts and provide a structured response following the JSON format specified in the system prompt. Ensure all references include exact clause numbers, section titles, and page numbers from the provided excerpts.
"""
        return prompt
    
    def _query_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Query the LLM with system and user prompts"""
        try:
            if self.config.LLM_PROVIDER == "openai":
                response = self.llm_client.chat.completions.create(
                    model=self.config.LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.config.LLM_TEMPERATURE,
                    max_tokens=self.config.MAX_TOKENS
                )
                return response.choices[0].message.content
                
            elif self.config.LLM_PROVIDER == "anthropic":
                response = self.llm_client.messages.create(
                    model=self.config.LLM_MODEL,
                    max_tokens=self.config.MAX_TOKENS,
                    temperature=self.config.LLM_TEMPERATURE,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ]
                )
                return response.content[0].text
                
            elif self.config.LLM_PROVIDER in ["local", "ollama"]:
                # Combine system and user prompts for local models
                full_prompt = f"<|system|>\n{system_prompt}\n\n<|user|>\n{user_prompt}\n\n<|assistant|>\n"
                
                return self.local_model_manager.generate_response(
                    full_prompt,
                    max_tokens=self.config.MAX_TOKENS,
                    temperature=self.config.LLM_TEMPERATURE
                )
                
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            raise
    
    def _parse_llm_response(self, llm_response: str, parsed_query: ParsedQuery, 
                           retrieved_chunks: List[Dict[str, Any]]) -> InsuranceDecision:
        """Parse LLM response into structured decision"""
        try:
            # Try to extract JSON from response
            json_match = self._extract_json_from_text(llm_response)
            
            if json_match:
                decision_data = json.loads(json_match)
            else:
                # Fallback: create structured response from unstructured text
                decision_data = self._structure_unstructured_response(llm_response)
            
            # Validate and clean the decision data
            decision_data = self._validate_decision_data(decision_data)
            
            # Calculate confidence score
            confidence = self._calculate_response_confidence(
                decision_data, retrieved_chunks, parsed_query
            )
            
            return InsuranceDecision(
                decision=decision_data.get("decision", "uncertain"),
                justification=decision_data.get("justification", "Unable to determine from available policy information."),
                references=decision_data.get("references", []),
                confidence_score=confidence,
                query_type=parsed_query.query_type,
                retrieved_chunks_count=len(retrieved_chunks)
            )
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return self._create_error_response(parsed_query, f"Response parsing error: {str(e)}")
    
    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """Extract JSON from text response"""
        import re
        
        # Look for JSON blocks
        json_patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'\{.*\}',
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                return matches[0].strip()
        
        return None
    
    def _structure_unstructured_response(self, response: str) -> Dict[str, Any]:
        """Convert unstructured response to structured format"""
        # Basic parsing of unstructured response
        decision = "uncertain"
        justification = response
        references = []
        
        # Try to extract decision
        if any(word in response.lower() for word in ["approved", "covered", "eligible", "yes"]):
            decision = "approved"
        elif any(word in response.lower() for word in ["denied", "rejected", "excluded", "not covered", "no"]):
            decision = "rejected"
        elif any(word in response.lower() for word in ["partial", "limited", "conditions apply"]):
            decision = "conditional"
        
        return {
            "decision": decision,
            "justification": justification,
            "references": references
        }
    
    def _validate_decision_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean decision data"""
        # Ensure required fields exist
        required_fields = ["decision", "justification", "references"]
        for field in required_fields:
            if field not in data:
                data[field] = ""
        
        # Validate decision values
        valid_decisions = ["approved", "rejected", "conditional", "uncertain", "requires_review"]
        if data["decision"] not in valid_decisions:
            data["decision"] = "uncertain"
        
        # Ensure references is a list
        if not isinstance(data["references"], list):
            data["references"] = []
        
        # Validate reference structure
        validated_references = []
        for ref in data["references"]:
            if isinstance(ref, dict):
                validated_ref = {
                    "clause_number": str(ref.get("clause_number", "")),
                    "section": str(ref.get("section", "")),
                    "page": int(ref.get("page", 0)) if str(ref.get("page", "")).isdigit() else 0
                }
                validated_references.append(validated_ref)
        
        data["references"] = validated_references
        
        return data
    
    def _calculate_response_confidence(self, decision_data: Dict[str, Any], 
                                     retrieved_chunks: List[Dict[str, Any]], 
                                     parsed_query: ParsedQuery) -> float:
        """Calculate confidence score for the response"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on number of references
        if decision_data["references"]:
            confidence += min(len(decision_data["references"]) * 0.1, 0.3)
        
        # Increase confidence based on chunk similarity scores
        if retrieved_chunks:
            avg_similarity = sum(chunk.get('similarity_score', 0) for chunk in retrieved_chunks) / len(retrieved_chunks)
            confidence += avg_similarity * 0.2
        
        # Increase confidence based on query parsing confidence
        confidence += parsed_query.confidence * 0.1
        
        # Decrease confidence for uncertain decisions
        if decision_data["decision"] == "uncertain":
            confidence -= 0.2
        
        return min(max(confidence, 0.0), 1.0)
    
    def _create_no_data_response(self, parsed_query: ParsedQuery) -> InsuranceDecision:
        """Create response when no relevant data is found"""
        return InsuranceDecision(
            decision="uncertain",
            justification="No relevant policy information found to answer this query. Please check if the query relates to covered services or contact customer service for clarification.",
            references=[],
            confidence_score=0.1,
            query_type=parsed_query.query_type,
            retrieved_chunks_count=0
        )
    
    def _create_error_response(self, parsed_query: ParsedQuery, error_msg: str) -> InsuranceDecision:
        """Create response for error cases"""
        return InsuranceDecision(
            decision="error",
            justification=f"Unable to process query due to system error: {error_msg}",
            references=[],
            confidence_score=0.0,
            query_type=parsed_query.query_type,
            retrieved_chunks_count=0
        )
    
    # System prompt definitions
    def _get_coverage_system_prompt(self) -> str:
        return """You are an expert insurance claims analyst specializing in health insurance policy interpretation. Your role is to analyze coverage inquiries against policy documents and provide accurate, well-referenced decisions.

INSTRUCTIONS:
1. Carefully read the query and policy excerpts provided
2. Determine if the requested service/treatment is covered based ONLY on the policy language
3. Consider all relevant factors: waiting periods, exclusions, age limits, pre-existing conditions
4. Provide a clear decision with detailed justification
5. Cite EXACT clause numbers, section titles, and page numbers from the provided excerpts

OUTPUT FORMAT (JSON):
{
  "decision": "approved|rejected|conditional|uncertain",
  "justification": "Detailed explanation referencing specific policy provisions",
  "references": [
    {
      "clause_number": "exact clause/section number",
      "section": "exact section title",
      "page": page_number
    }
  ]
}

DECISION CRITERIA:
- "approved": Coverage is clearly stated in the policy
- "rejected": Coverage is explicitly excluded or denied
- "conditional": Coverage depends on specific conditions being met
- "uncertain": Insufficient information in provided excerpts to determine coverage

Always ground your response in the actual policy language provided."""
    
    def _get_exclusion_system_prompt(self) -> str:
        return """You are an expert insurance policy analyst specializing in exclusion clauses and coverage limitations. Your role is to identify what is NOT covered under the insurance policy.

INSTRUCTIONS:
1. Focus on exclusion clauses and limitations in the policy excerpts
2. Determine if the queried item falls under any exclusions
3. Consider permanent exclusions vs. waiting period limitations
4. Provide clear reasoning based on policy language

OUTPUT FORMAT (JSON):
{
  "decision": "excluded|covered|conditional|uncertain",
  "justification": "Detailed explanation of exclusion status with policy references",
  "references": [
    {
      "clause_number": "exact clause/section number",
      "section": "exact section title", 
      "page": page_number
    }
  ]
}

Focus particularly on exclusion clauses, limitations, and conditions that would prevent coverage."""
    
    def _get_claim_system_prompt(self) -> str:
        return """You are an expert insurance claims processor. Your role is to evaluate claim-related queries against policy provisions for reimbursement, settlement procedures, and claim processing requirements.

INSTRUCTIONS:
1. Evaluate claim eligibility based on policy terms
2. Consider documentation requirements, time limits, and procedures
3. Assess reimbursement limits and conditions
4. Reference exact policy provisions for claim processing

OUTPUT FORMAT (JSON):
{
  "decision": "eligible|ineligible|conditional|uncertain",
  "justification": "Detailed explanation of claim status and requirements",
  "references": [
    {
      "clause_number": "exact clause/section number",
      "section": "exact section title",
      "page": page_number
    }
  ]
}

Consider all aspects of claim processing including eligibility, documentation, limits, and procedures."""
    
    def _get_waiting_period_system_prompt(self) -> str:
        return """You are an expert insurance policy analyst specializing in waiting periods and time-based restrictions. Your role is to interpret waiting period clauses and their impact on coverage.

INSTRUCTIONS:
1. Identify applicable waiting periods from policy excerpts
2. Calculate if the waiting period has been satisfied
3. Consider different waiting periods for different conditions/treatments
4. Distinguish between waiting periods and permanent exclusions

OUTPUT FORMAT (JSON):
{
  "decision": "waiting_satisfied|waiting_active|permanent_exclusion|uncertain",
  "justification": "Detailed explanation of waiting period status and timeline",
  "references": [
    {
      "clause_number": "exact clause/section number",
      "section": "exact section title",
      "page": page_number
    }
  ]
}

Focus on time-based restrictions, waiting periods, and their specific application to the queried scenario."""
    
    def _get_limit_system_prompt(self) -> str:
        return """You are an expert insurance policy analyst specializing in coverage limits, caps, and financial restrictions. Your role is to interpret limit-related clauses and their application.

INSTRUCTIONS:
1. Identify applicable coverage limits (annual, lifetime, per-incident)
2. Determine if requested amount falls within policy limits
3. Consider sub-limits for specific categories of treatment
4. Reference exact limit amounts and conditions

OUTPUT FORMAT (JSON):
{
  "decision": "within_limits|exceeds_limits|conditional|uncertain",
  "justification": "Detailed explanation of applicable limits and their impact",
  "references": [
    {
      "clause_number": "exact clause/section number",
      "section": "exact section title",
      "page": page_number
    }
  ]
}

Focus on financial limits, caps, and restrictions that affect coverage amounts."""
    
    def _get_general_system_prompt(self) -> str:
        return """You are an expert insurance policy analyst. Your role is to interpret insurance policy documents and provide accurate, well-referenced answers to general policy inquiries.

INSTRUCTIONS:
1. Analyze the query against all relevant policy provisions
2. Consider coverage, exclusions, limits, waiting periods, and conditions
3. Provide comprehensive analysis based on policy language
4. Reference specific clauses supporting your conclusion

OUTPUT FORMAT (JSON):
{
  "decision": "approved|rejected|conditional|uncertain|information_only",
  "justification": "Comprehensive explanation addressing all aspects of the query",
  "references": [
    {
      "clause_number": "exact clause/section number",
      "section": "exact section title",
      "page": page_number
    }
  ]
}

Provide thorough analysis considering all relevant policy aspects."""

if __name__ == "__main__":
    # Test the LLM reasoner
    from query_processor import QueryProcessor
    
    config = RAGFlowConfig()
    reasoner = LLMReasoner(config)
    query_processor = QueryProcessor(config)
    
    # Test query
    test_query = "Is knee surgery covered for a 45-year-old?"
    parsed_query = query_processor.parse_query(test_query)
    
    # Mock retrieved chunks for testing
    mock_chunks = [
        {
            'text': 'Orthopedic surgeries including knee replacement are covered after 12 months waiting period for members above 18 years.',
            'page_number': 15,
            'section_title': 'Surgical Procedures Coverage',
            'clause_number': '4.2.1',
            'similarity_score': 0.85
        }
    ]
    
    decision = reasoner.analyze_insurance_query(parsed_query, mock_chunks)
    print("Decision JSON:")
    print(decision.to_json()) 