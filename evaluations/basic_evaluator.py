"""
Basic RAG evaluator for testing retrieval and generation quality

This module provides the main evaluation framework for testing RAG systems
with predefined test queries and comprehensive metrics.
"""

import logging
import os

# Import RAG components
import sys
import time
from typing import Any, Dict, List, Optional

from .metrics import EvaluationMetrics, EvaluationResult, QueryResult
from .test_datasets import EvaluationDataset, EvaluationQuery

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import RepositoryDataLoader
from src.rag_tools import RAGToolManager

logger = logging.getLogger(__name__)


class BasicRAGEvaluator:
    """Basic evaluator for RAG systems"""
    
    def __init__(
        self,
        data_dir: str = "data/mistral-repos",
        db_path: str = "db.duckdb",
        vector_db_path: Optional[str] = None,
        api_key: Optional[str] = None,
        force_rebuild_vectors: bool = False,
    ):
        """
        Initialize the RAG evaluator
        
        Args:
            data_dir: Directory containing repository data
            db_path: Path to main DuckDB database
            vector_db_path: Path to vector database (optional)
            api_key: Mistral API key
            force_rebuild_vectors: Whether to force rebuild vector embeddings
        """
        self.data_dir = data_dir
        self.db_path = db_path
        self.vector_db_path = vector_db_path
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.force_rebuild_vectors = force_rebuild_vectors
        
        # Initialize components
        self.data_loader: Optional[RepositoryDataLoader] = None
        self.rag_manager: Optional[RAGToolManager] = None
        self.test_dataset = EvaluationDataset()
        self.metrics = EvaluationMetrics()
        
        # Evaluation settings
        self.timeout_seconds = 30.0
        self.max_retries = 2
        
        logger.info(f"Initialized BasicRAGEvaluator with {len(self.test_dataset.queries)} test queries")
    
    def initialize_rag_system(self) -> bool:
        """Initialize the RAG system components"""
        try:
            logger.info("Initializing RAG system for evaluation...")
            
            # Check for API key
            if not self.api_key:
                logger.error("MISTRAL_API_KEY is required for evaluation")
                return False
            
            # Initialize data loader
            vector_db = self.vector_db_path if self.vector_db_path else self.db_path
            self.data_loader = RepositoryDataLoader(
                self.data_dir,
                self.db_path,
                vector_db_path=vector_db,
                force_rebuild_vectors=self.force_rebuild_vectors,
            )
            
            # Check available repositories
            repos = self.data_loader.get_available_repositories()
            if not repos:
                logger.error(f"No repositories found in {self.data_dir}")
                return False
            
            logger.info(f"Found {len(repos)} repositories: {', '.join(repos)}")
            
            # Initialize RAG tools
            self.rag_manager = RAGToolManager(self.data_loader, self.api_key)
            
            logger.info("RAG system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            return False
    
    def evaluate_single_query(
        self, 
        test_query: EvaluationQuery, 
        tools_to_test: Optional[List[str]] = None
    ) -> EvaluationResult:
        """
        Evaluate a single test query against specified tools
        
        Args:
            test_query: The test query to evaluate
            tools_to_test: List of tool names to test (default: all available tools)
            
        Returns:
            EvaluationResult containing results for all tested tools
        """
        if not self.rag_manager:
            raise RuntimeError("RAG system not initialized. Call initialize_rag_system() first.")
        
        start_time = time.time()
        
        # Determine which tools to test
        if tools_to_test is None:
            available_tools = [tool["name"] for tool in self.rag_manager.get_available_tools()]
            tools_to_test = available_tools
        
        logger.info(f"Evaluating query '{test_query.id}': {test_query.query}")
        logger.info(f"Testing tools: {tools_to_test}")
        
        tool_results = {}
        overall_success = False
        
        # Test each tool
        for tool_name in tools_to_test:
            tool_result = self._evaluate_tool_for_query(test_query, tool_name)
            tool_results[tool_name] = tool_result
            
            if tool_result.success:
                overall_success = True
        
        # Determine best performing tool
        best_tool = None
        if overall_success:
            successful_tools = {
                name: result for name, result in tool_results.items() 
                if result.success
            }
            # Choose tool with best combination of speed and result count
            best_tool = min(
                successful_tools.keys(),
                key=lambda t: successful_tools[t].response_time / max(successful_tools[t].result_count, 1)
            )
        
        evaluation_time = time.time() - start_time
        
        result = EvaluationResult(
            query_id=test_query.id,
            query_text=test_query.query,
            category=test_query.category,
            difficulty=test_query.difficulty,
            tool_results=tool_results,
            overall_success=overall_success,
            best_tool=best_tool,
            evaluation_time=evaluation_time
        )
        
        logger.info(f"Query '{test_query.id}' evaluation completed in {evaluation_time:.2f}s")
        logger.info(f"Overall success: {overall_success}, Best tool: {best_tool}")
        
        return result
    
    def _evaluate_tool_for_query(self, test_query: EvaluationQuery, tool_name: str) -> QueryResult:
        """Evaluate a specific tool for a given query"""
        start_time = time.time()
        
        try:
            logger.debug(f"Testing {tool_name} for query: {test_query.query}")
            
            # Execute the query with timeout
            results = self._execute_with_timeout(
                lambda: self.rag_manager.search(test_query.query, tool_name, max_results=10),
                self.timeout_seconds
            )
            
            response_time = time.time() - start_time
            
            # Check if results meet expectations
            success = self._validate_results(test_query, results, tool_name)
            
            return QueryResult(
                query_id=test_query.id,
                query_text=test_query.query,
                tool_name=tool_name,
                success=success,
                response_time=response_time,
                result_count=len(results) if results else 0,
                results=results or [],
                metadata=self._extract_result_metadata(test_query, results, tool_name)
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            logger.warning(f"Tool {tool_name} failed for query '{test_query.id}': {e}")
            
            return QueryResult(
                query_id=test_query.id,
                query_text=test_query.query,
                tool_name=tool_name,
                success=False,
                response_time=response_time,
                result_count=0,
                results=[],
                error_message=str(e)
            )
    
    def _execute_with_timeout(self, func, timeout_seconds: float):
        """Execute a function with timeout"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")
        
        # Set up timeout (Unix-like systems only)
        if hasattr(signal, 'SIGALRM'):
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_seconds))
            
            try:
                result = func()
                signal.alarm(0)  # Cancel the alarm
                return result
            finally:
                signal.signal(signal.SIGALRM, old_handler)
        else:
            # Fallback for systems without SIGALRM (like Windows)
            return func()
    
    def _validate_results(self, test_query: EvaluationQuery, results: List[Dict[str, Any]], tool_name: str) -> bool:
        """Validate if results meet the test query expectations"""
        if not results:
            return test_query.min_results == 0
        
        # Check minimum result count
        if len(results) < test_query.min_results:
            return False
        
        # Check for errors in results
        if any("error" in result for result in results):
            return False
        
        # Tool-specific validation
        if tool_name == "text_to_sql":
            return self._validate_sql_results(test_query, results)
        else:
            return self._validate_search_results(test_query, results)
    
    def _validate_sql_results(self, test_query: EvaluationQuery, results: List[Dict[str, Any]]) -> bool:
        """Validate SQL query results"""
        if not results:
            return False
        
        result = results[0]
        
        # Check if SQL query was generated
        if "sql_query" not in result:
            return False
        
        sql_query = result["sql_query"]
        
        # Check SQL pattern if specified
        if test_query.expected_sql_pattern:
            if not self.metrics.check_sql_pattern(test_query.expected_sql_pattern, sql_query):
                logger.debug(f"SQL pattern mismatch for {test_query.id}: expected {test_query.expected_sql_pattern}")
                return False
        
        # Check if query executed successfully
        if "error" in result:
            return False
        
        # Check if results were returned (for non-zero expectations)
        if test_query.min_results > 0 and not result.get("results"):
            return False
        
        return True
    
    def _validate_search_results(self, test_query: EvaluationQuery, results: List[Dict[str, Any]]) -> bool:
        """Validate text/vector search results"""
        # Check keyword coverage
        keyword_coverage = self.metrics.check_keyword_coverage(
            test_query.expected_keywords, results
        )
        
        # Check repository coverage
        repo_coverage = self.metrics.check_repository_coverage(
            test_query.expected_repositories, results
        )
        
        # Consider results valid if they have reasonable coverage
        # (at least 30% keyword coverage OR 50% repository coverage)
        return keyword_coverage >= 0.3 or repo_coverage >= 0.5
    
    def _extract_result_metadata(
        self, 
        test_query: EvaluationQuery, 
        results: List[Dict[str, Any]], 
        tool_name: str
    ) -> Dict[str, Any]:
        """Extract metadata from results for analysis"""
        if not results:
            return {}
        
        metadata = {
            "keyword_coverage": self.metrics.check_keyword_coverage(
                test_query.expected_keywords, results
            ),
            "repository_coverage": self.metrics.check_repository_coverage(
                test_query.expected_repositories, results
            ),
        }
        
        if tool_name == "text_to_sql" and results:
            result = results[0]
            metadata.update({
                "sql_generated": "sql_query" in result,
                "sql_executed": "error" not in result,
                "sql_pattern_match": (
                    self.metrics.check_sql_pattern(
                        test_query.expected_sql_pattern or "", 
                        result.get("sql_query", "")
                    )
                    if test_query.expected_sql_pattern else True
                )
            })
        
        return metadata
    
    def run_evaluation(
        self, 
        query_categories: Optional[List[str]] = None,
        query_difficulties: Optional[List[str]] = None,
        tools_to_test: Optional[List[str]] = None,
        max_queries: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation
        
        Args:
            query_categories: List of categories to test (default: all)
            query_difficulties: List of difficulties to test (default: all)
            tools_to_test: List of tools to test (default: all)
            max_queries: Maximum number of queries to test (default: all)
            
        Returns:
            Dictionary containing evaluation results and metrics
        """
        if not self.initialize_rag_system():
            return {"error": "Failed to initialize RAG system"}
        
        # Filter queries based on criteria
        queries_to_test = self._filter_queries(
            query_categories, query_difficulties, max_queries
        )
        
        logger.info(f"Starting evaluation with {len(queries_to_test)} queries")
        
        # Clear previous results
        self.metrics.clear_results()
        
        # Evaluate each query
        for i, test_query in enumerate(queries_to_test, 1):
            logger.info(f"Progress: {i}/{len(queries_to_test)} - {test_query.id}")
            
            try:
                result = self.evaluate_single_query(test_query, tools_to_test)
                self.metrics.add_result(result)
                
            except Exception as e:
                logger.error(f"Failed to evaluate query {test_query.id}: {e}")
                # Add failed result
                failed_result = EvaluationResult(
                    query_id=test_query.id,
                    query_text=test_query.query,
                    category=test_query.category,
                    difficulty=test_query.difficulty,
                    tool_results={},
                    overall_success=False,
                    evaluation_time=0.0
                )
                self.metrics.add_result(failed_result)
        
        # Generate comprehensive results
        results = {
            "evaluation_summary": {
                "total_queries": len(queries_to_test),
                "completed_queries": len(self.metrics.results),
                "evaluation_timestamp": time.time(),
                "tools_tested": tools_to_test or ["all"],
                "categories_tested": query_categories or ["all"],
                "difficulties_tested": query_difficulties or ["all"],
            },
            "basic_metrics": self.metrics.calculate_basic_metrics(),
            "category_metrics": self.metrics.calculate_category_metrics(),
            "precision_metrics": self.metrics.calculate_precision_at_k(),
            "time_percentiles": self.metrics.calculate_response_time_percentiles(),
            "best_tools": self.metrics.identify_best_tool_per_category(),
        }
        
        logger.info("Evaluation completed successfully")
        return results
    
    def _filter_queries(
        self,
        categories: Optional[List[str]] = None,
        difficulties: Optional[List[str]] = None,
        max_queries: Optional[int] = None
    ) -> List[EvaluationQuery]:
        """Filter test queries based on criteria"""
        queries = self.test_dataset.queries
        
        if categories:
            queries = [q for q in queries if q.category in categories]
        
        if difficulties:
            queries = [q for q in queries if q.difficulty in difficulties]
        
        if max_queries:
            queries = queries[:max_queries]
        
        return queries
    
    def generate_report(self, save_to_file: Optional[str] = None) -> str:
        """Generate a detailed evaluation report"""
        report = self.metrics.generate_detailed_report()
        
        if save_to_file:
            with open(save_to_file, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Report saved to {save_to_file}")
        
        return report
    
    def export_results(self, filepath: str) -> None:
        """Export detailed results to JSON file"""
        self.metrics.export_results_to_json(filepath)
        logger.info(f"Results exported to {filepath}")
    
    def quick_test(self, num_queries: int = 5) -> str:
        """Run a quick test with a small number of queries"""
        logger.info(f"Running quick test with {num_queries} queries")
        
        # Select a mix of easy queries from different categories
        easy_queries = self.test_dataset.get_queries_by_difficulty("easy")
        test_queries = easy_queries[:num_queries]
        
        results = self.run_evaluation(max_queries=num_queries)
        
        if "error" in results:
            return f"Quick test failed: {results['error']}"
        
        # Generate summary
        basic_metrics = results["basic_metrics"]
        overall = basic_metrics["overall"]
        
        summary = f"""
🚀 QUICK TEST RESULTS ({num_queries} queries)

Overall Success Rate: {overall['overall_success_rate']}%
Total Evaluation Time: {overall['total_evaluation_time']}s

Tool Performance:
"""
        
        for tool, metrics in basic_metrics["by_tool"].items():
            summary += f"  {tool}: {metrics['success_rate']}% success, {metrics['avg_response_time']}s avg time\n"
        
        return summary.strip()
