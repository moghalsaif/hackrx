import json
import time
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import pandas as pd

from config import RAGFlowConfig
from ragflow_system import RAGFlowSystem

# Set up logging
logging.basicConfig(level=getattr(logging, RAGFlowConfig.LOG_LEVEL))
logger = logging.getLogger(__name__)

@dataclass
class TestCase:
    """Represents a test case for the RAG system"""
    id: str
    query: str
    expected_decision: str
    category: str
    difficulty: str
    description: str
    expected_references: List[str] = None
    
    def __post_init__(self):
        if self.expected_references is None:
            self.expected_references = []

@dataclass
class TestResult:
    """Represents the result of a test case execution"""
    test_case: TestCase
    actual_response: Dict[str, Any]
    passed: bool
    score: float
    execution_time: float
    errors: List[str]
    notes: str

class RAGSystemTester:
    """Comprehensive testing system for the RAG pipeline"""
    
    def __init__(self, config: RAGFlowConfig = None):
        self.config = config or RAGFlowConfig()
        self.rag_system = None
        self.test_cases = []
        self.test_results = []
        
        # Load test cases
        self._create_test_cases()
        logger.info(f"Loaded {len(self.test_cases)} test cases")
    
    def _create_test_cases(self):
        """Create comprehensive test cases for insurance policy queries"""
        self.test_cases = [
            # Coverage Inquiries
            TestCase(
                id="COV_001",
                query="Is ACL reconstruction surgery covered for a 46-year-old under a 3-month-old policy?",
                expected_decision="conditional",  # May depend on waiting period
                category="coverage",
                difficulty="medium",
                description="ACL surgery coverage with age and policy tenure factors",
                expected_references=["waiting period", "surgical procedures", "orthopedic"]
            ),
            TestCase(
                id="COV_002",
                query="What knee surgery procedures are covered?",
                expected_decision="approved",
                category="coverage",
                difficulty="easy",
                description="General knee surgery coverage inquiry",
                expected_references=["surgical procedures", "orthopedic"]
            ),
            TestCase(
                id="COV_003",
                query="Is cardiac bypass surgery covered for emergency cases?",
                expected_decision="approved",
                category="coverage",
                difficulty="medium",
                description="Emergency cardiac surgery coverage",
                expected_references=["cardiac surgery", "emergency treatment"]
            ),
            
            # Exclusion Inquiries
            TestCase(
                id="EXC_001",
                query="Are cosmetic surgeries excluded from coverage?",
                expected_decision="rejected",
                category="exclusion",
                difficulty="easy",
                description="Cosmetic surgery exclusion check",
                expected_references=["exclusions", "cosmetic"]
            ),
            TestCase(
                id="EXC_002",
                query="Is dental treatment covered under this policy?",
                expected_decision="conditional",  # Often limited or excluded
                category="exclusion",
                difficulty="medium",
                description="Dental coverage inquiry",
                expected_references=["dental", "exclusions"]
            ),
            
            # Waiting Period Inquiries
            TestCase(
                id="WAIT_001",
                query="What is the waiting period for knee surgery?",
                expected_decision="information_only",
                category="waiting_period",
                difficulty="easy",
                description="Waiting period information request",
                expected_references=["waiting period", "surgical procedures"]
            ),
            TestCase(
                id="WAIT_002",
                query="Can I get surgery immediately after buying the policy?",
                expected_decision="rejected",
                category="waiting_period",
                difficulty="medium",
                description="Immediate surgery eligibility",
                expected_references=["waiting period"]
            ),
            
            # Limit Inquiries
            TestCase(
                id="LIM_001",
                query="What is the maximum coverage limit for orthopedic surgery?",
                expected_decision="information_only",
                category="limits",
                difficulty="medium",
                description="Coverage limit inquiry",
                expected_references=["coverage limits", "orthopedic"]
            ),
            TestCase(
                id="LIM_002",
                query="Is there a sub-limit for physiotherapy treatments?",
                expected_decision="information_only",
                category="limits",
                difficulty="medium",
                description="Sub-limit inquiry for specific treatment",
                expected_references=["sub-limits", "physiotherapy"]
            ),
            
            # Age-Specific Inquiries
            TestCase(
                id="AGE_001",
                query="What coverage is available for senior citizens above 60?",
                expected_decision="information_only",
                category="age_specific",
                difficulty="medium",
                description="Senior citizen coverage inquiry",
                expected_references=["age limits", "senior citizen"]
            ),
            TestCase(
                id="AGE_002",
                query="Are pediatric treatments covered for children under 18?",
                expected_decision="approved",
                category="age_specific",
                difficulty="easy",
                description="Pediatric coverage inquiry",
                expected_references=["pediatric", "age limits"]
            ),
            
            # Location-Specific Inquiries
            TestCase(
                id="LOC_001",
                query="Can I get treatment in Pune hospitals?",
                expected_decision="approved",
                category="location",
                difficulty="easy",
                description="Network hospital inquiry",
                expected_references=["network hospitals", "location"]
            ),
            
            # Complex Scenarios
            TestCase(
                id="COMP_001",
                query="I'm 45 years old, bought policy 6 months ago, need ACL surgery in Mumbai. What's covered?",
                expected_decision="conditional",
                category="complex",
                difficulty="hard",
                description="Multi-factor coverage analysis",
                expected_references=["waiting period", "orthopedic", "network hospitals"]
            ),
            TestCase(
                id="COMP_002",
                query="Pre-existing diabetes patient needs cardiac surgery - coverage options?",
                expected_decision="conditional",
                category="complex",
                difficulty="hard",
                description="Pre-existing condition with surgery need",
                expected_references=["pre-existing", "cardiac surgery", "waiting period"]
            ),
            
            # Edge Cases
            TestCase(
                id="EDGE_001",
                query="What happens if I need surgery on the exact day waiting period ends?",
                expected_decision="conditional",
                category="edge_case",
                difficulty="hard",
                description="Waiting period boundary condition",
                expected_references=["waiting period"]
            ),
            TestCase(
                id="EDGE_002",
                query="Is experimental cancer treatment covered?",
                expected_decision="uncertain",
                category="edge_case",
                difficulty="hard",
                description="Experimental treatment coverage",
                expected_references=["experimental", "cancer", "exclusions"]
            ),
            
            # Invalid/Irrelevant Queries
            TestCase(
                id="INV_001",
                query="What's the weather like today?",
                expected_decision="error",
                category="invalid",
                difficulty="easy",
                description="Completely irrelevant query",
                expected_references=[]
            ),
            TestCase(
                id="INV_002",
                query="How do I cook pasta?",
                expected_decision="error",
                category="invalid",
                difficulty="easy",
                description="Non-insurance related query",
                expected_references=[]
            )
        ]
    
    def run_all_tests(self, initialize_system: bool = True) -> Dict[str, Any]:
        """Run all test cases and return comprehensive results"""
        logger.info("Starting comprehensive test suite...")
        start_time = time.time()
        
        # Initialize system if needed
        if initialize_system:
            logger.info("Initializing RAG system for testing...")
            self.rag_system = RAGFlowSystem(self.config)
            
            if not self.rag_system.initialize_system():
                logger.error("Failed to initialize RAG system for testing")
                return {"error": "System initialization failed"}
        
        # Run each test case
        self.test_results = []
        for test_case in self.test_cases:
            logger.info(f"Running test case: {test_case.id}")
            result = self._run_single_test(test_case)
            self.test_results.append(result)
        
        # Generate comprehensive report
        total_time = time.time() - start_time
        report = self._generate_test_report(total_time)
        
        # Save results
        self._save_test_results(report)
        
        logger.info(f"Test suite completed in {total_time:.2f} seconds")
        return report
    
    def _run_single_test(self, test_case: TestCase) -> TestResult:
        """Run a single test case"""
        start_time = time.time()
        errors = []
        notes = ""
        
        try:
            # Execute query
            response = self.rag_system.query_system(test_case.query, include_debug_info=True)
            execution_time = time.time() - start_time
            
            # Evaluate result
            passed, score, evaluation_errors = self._evaluate_response(test_case, response)
            errors.extend(evaluation_errors)
            
            # Add notes about the response
            notes = f"Actual decision: {response.get('decision', 'N/A')}, " \
                   f"Confidence: {response.get('confidence_score', 0):.2f}, " \
                   f"References: {len(response.get('references', []))}"
            
        except Exception as e:
            execution_time = time.time() - start_time
            passed = False
            score = 0.0
            errors.append(f"Execution error: {str(e)}")
            response = {"error": str(e)}
            notes = "Test execution failed"
        
        return TestResult(
            test_case=test_case,
            actual_response=response,
            passed=passed,
            score=score,
            execution_time=execution_time,
            errors=errors,
            notes=notes
        )
    
    def _evaluate_response(self, test_case: TestCase, response: Dict[str, Any]) -> Tuple[bool, float, List[str]]:
        """Evaluate if the response meets expectations"""
        errors = []
        score = 0.0
        
        # Check if response has required fields
        required_fields = ["decision", "justification", "references", "confidence_score"]
        for field in required_fields:
            if field not in response:
                errors.append(f"Missing required field: {field}")
                return False, 0.0, errors
        
        actual_decision = response["decision"]
        expected_decision = test_case.expected_decision
        
        # Decision accuracy (40% of score)
        if actual_decision == expected_decision:
            score += 0.4
        elif self._is_acceptable_decision(actual_decision, expected_decision):
            score += 0.2  # Partial credit for reasonable alternatives
        else:
            errors.append(f"Decision mismatch: expected {expected_decision}, got {actual_decision}")
        
        # Reference quality (30% of score)
        ref_score, ref_errors = self._evaluate_references(test_case, response["references"])
        score += ref_score * 0.3
        errors.extend(ref_errors)
        
        # Response quality (20% of score)
        quality_score, quality_errors = self._evaluate_response_quality(response)
        score += quality_score * 0.2
        errors.extend(quality_errors)
        
        # Confidence appropriateness (10% of score)
        conf_score = self._evaluate_confidence(response["confidence_score"], actual_decision)
        score += conf_score * 0.1
        
        # Test passes if score >= 0.6 and no critical errors
        passed = score >= 0.6 and len(errors) == 0
        
        return passed, score, errors
    
    def _is_acceptable_decision(self, actual: str, expected: str) -> bool:
        """Check if actual decision is acceptable alternative to expected"""
        acceptable_alternatives = {
            "approved": ["conditional"],
            "conditional": ["approved", "uncertain"],
            "rejected": ["uncertain"],
            "uncertain": ["conditional"],
            "information_only": ["approved", "conditional"]
        }
        
        return actual in acceptable_alternatives.get(expected, [])
    
    def _evaluate_references(self, test_case: TestCase, references: List[Dict]) -> Tuple[float, List[str]]:
        """Evaluate quality and relevance of references"""
        errors = []
        score = 0.0
        
        if not references:
            if test_case.expected_references:
                errors.append("No references provided when expected")
                return 0.0, errors
            else:
                return 1.0, errors  # No references expected or provided
        
        # Check reference structure
        for i, ref in enumerate(references):
            if not isinstance(ref, dict):
                errors.append(f"Reference {i+1} is not a dictionary")
                continue
            
            required_ref_fields = ["clause_number", "section", "page"]
            for field in required_ref_fields:
                if field not in ref or not str(ref[field]).strip():
                    errors.append(f"Reference {i+1} missing or empty {field}")
        
        # Check reference relevance
        if test_case.expected_references:
            relevance_score = 0.0
            for expected_ref in test_case.expected_references:
                for ref in references:
                    ref_text = f"{ref.get('section', '')} {ref.get('clause_number', '')}".lower()
                    if expected_ref.lower() in ref_text:
                        relevance_score += 1.0
                        break
            
            relevance_score = min(relevance_score / len(test_case.expected_references), 1.0)
            score += relevance_score
        else:
            score = 1.0  # No specific references expected
        
        return score, errors
    
    def _evaluate_response_quality(self, response: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Evaluate overall response quality"""
        errors = []
        score = 0.0
        
        justification = response.get("justification", "")
        
        # Check justification length and content
        if len(justification.strip()) < 20:
            errors.append("Justification too short")
        else:
            score += 0.5
        
        # Check for policy-specific language
        policy_keywords = ["policy", "coverage", "clause", "section", "benefit", "exclusion"]
        if any(keyword in justification.lower() for keyword in policy_keywords):
            score += 0.5
        
        return score, errors
    
    def _evaluate_confidence(self, confidence: float, decision: str) -> float:
        """Evaluate if confidence score is appropriate for the decision"""
        if decision == "error":
            return 1.0 if confidence <= 0.2 else 0.0
        elif decision == "uncertain":
            return 1.0 if 0.1 <= confidence <= 0.6 else 0.5
        elif decision in ["approved", "rejected"]:
            return 1.0 if confidence >= 0.6 else 0.5
        else:
            return 1.0 if confidence >= 0.4 else 0.5
    
    def _generate_test_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        if not self.test_results:
            return {"error": "No test results available"}
        
        # Overall statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result.passed)
        total_score = sum(result.score for result in self.test_results)
        avg_score = total_score / total_tests if total_tests > 0 else 0
        
        # Category-wise analysis
        category_stats = {}
        for result in self.test_results:
            category = result.test_case.category
            if category not in category_stats:
                category_stats[category] = {"total": 0, "passed": 0, "avg_score": 0}
            
            category_stats[category]["total"] += 1
            if result.passed:
                category_stats[category]["passed"] += 1
            category_stats[category]["avg_score"] += result.score
        
        # Calculate averages
        for category in category_stats:
            stats = category_stats[category]
            stats["avg_score"] = stats["avg_score"] / stats["total"]
            stats["pass_rate"] = stats["passed"] / stats["total"]
        
        # Performance analysis
        avg_execution_time = sum(result.execution_time for result in self.test_results) / total_tests
        slowest_test = max(self.test_results, key=lambda x: x.execution_time)
        fastest_test = min(self.test_results, key=lambda x: x.execution_time)
        
        # Error analysis
        all_errors = []
        for result in self.test_results:
            all_errors.extend(result.errors)
        
        error_counts = {}
        for error in all_errors:
            error_counts[error] = error_counts.get(error, 0) + 1
        
        # Failed tests details
        failed_tests = [result for result in self.test_results if not result.passed]
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": len(failed_tests),
                "pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "average_score": avg_score,
                "total_execution_time": total_time,
                "average_execution_time": avg_execution_time
            },
            "category_analysis": category_stats,
            "performance_analysis": {
                "slowest_test": {
                    "id": slowest_test.test_case.id,
                    "time": slowest_test.execution_time,
                    "query": slowest_test.test_case.query
                },
                "fastest_test": {
                    "id": fastest_test.test_case.id,
                    "time": fastest_test.execution_time,
                    "query": fastest_test.test_case.query
                }
            },
            "error_analysis": {
                "total_errors": len(all_errors),
                "error_counts": error_counts
            },
            "failed_tests": [
                {
                    "id": result.test_case.id,
                    "query": result.test_case.query,
                    "expected": result.test_case.expected_decision,
                    "actual": result.actual_response.get("decision", "N/A"),
                    "score": result.score,
                    "errors": result.errors
                }
                for result in failed_tests
            ],
            "detailed_results": [
                {
                    "test_id": result.test_case.id,
                    "category": result.test_case.category,
                    "difficulty": result.test_case.difficulty,
                    "passed": result.passed,
                    "score": result.score,
                    "execution_time": result.execution_time,
                    "notes": result.notes
                }
                for result in self.test_results
            ]
        }
        
        return report
    
    def _save_test_results(self, report: Dict[str, Any]):
        """Save test results to files"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save JSON report
        json_file = self.config.LOGS_DIR / f"test_report_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save CSV summary
        csv_file = self.config.LOGS_DIR / f"test_summary_{timestamp}.csv"
        df_data = []
        for result in self.test_results:
            df_data.append({
                "test_id": result.test_case.id,
                "category": result.test_case.category,
                "difficulty": result.test_case.difficulty,
                "query": result.test_case.query,
                "expected_decision": result.test_case.expected_decision,
                "actual_decision": result.actual_response.get("decision", "N/A"),
                "passed": result.passed,
                "score": result.score,
                "execution_time": result.execution_time,
                "confidence": result.actual_response.get("confidence_score", 0),
                "references_count": len(result.actual_response.get("references", [])),
                "errors": "; ".join(result.errors)
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(csv_file, index=False)
        
        logger.info(f"Test results saved to {json_file} and {csv_file}")

def main():
    """Main function for running the test suite"""
    print("🧪 Starting RAGFlow System Test Suite...")
    
    # Initialize tester
    tester = RAGSystemTester()
    
    # Run all tests
    report = tester.run_all_tests(initialize_system=True)
    
    if "error" in report:
        print(f"❌ Test suite failed: {report['error']}")
        return
    
    # Display summary
    summary = report["summary"]
    print("\n📊 Test Results Summary:")
    print("=" * 50)
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed_tests']}")
    print(f"Failed: {summary['failed_tests']}")
    print(f"Pass Rate: {summary['pass_rate']:.1%}")
    print(f"Average Score: {summary['average_score']:.2f}")
    print(f"Total Execution Time: {summary['total_execution_time']:.2f}s")
    print(f"Average Execution Time: {summary['average_execution_time']:.2f}s")
    
    # Display category analysis
    print("\n📈 Category Analysis:")
    print("-" * 30)
    for category, stats in report["category_analysis"].items():
        print(f"{category.upper()}: {stats['passed']}/{stats['total']} "
              f"({stats['pass_rate']:.1%}) - Avg Score: {stats['avg_score']:.2f}")
    
    # Display failed tests
    if report["failed_tests"]:
        print("\n❌ Failed Tests:")
        print("-" * 20)
        for failed in report["failed_tests"][:5]:  # Show first 5 failures
            print(f"  {failed['id']}: {failed['query'][:50]}...")
            print(f"    Expected: {failed['expected']}, Got: {failed['actual']}")
            print(f"    Score: {failed['score']:.2f}")
    
    print(f"\n📁 Detailed results saved to logs directory")

if __name__ == "__main__":
    main() 