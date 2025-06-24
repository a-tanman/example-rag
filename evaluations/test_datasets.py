"""
Test datasets for RAG evaluation

This module contains predefined test questions and expected results for evaluating
the RAG system across different query types and complexity levels.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class EvaluationQuery:
    """Represents a single test query with expected results"""
    
    id: str
    query: str
    category: str
    difficulty: str  # "easy", "medium", "hard"
    expected_tools: List[str]  # Which tools should work well for this query
    expected_keywords: List[str]  # Keywords that should appear in results
    expected_repositories: List[str]  # Repositories that should be mentioned
    description: str
    expected_sql_pattern: Optional[str] = None  # For SQL queries
    min_results: int = 1  # Minimum expected results
    max_response_time: float = 10.0  # Maximum acceptable response time in seconds


class EvaluationDataset:
    """Collection of test queries for RAG evaluation"""
    
    def __init__(self):
        self.queries = self._create_test_queries()
    
    def _create_test_queries(self) -> List[EvaluationQuery]:
        """Create comprehensive test dataset"""
        
        queries = []
        
        # === FACTUAL/STATISTICAL QUERIES ===
        queries.extend([
            EvaluationQuery(
                id="fact_001",
                query="How many repositories are available?",
                category="factual",
                difficulty="easy",
                expected_tools=["text_to_sql"],
                expected_keywords=["repository", "count", "total"],
                expected_repositories=["mistralai/client-python", "mistralai/mistral-inference"],
                description="Basic counting query",
                expected_sql_pattern="COUNT",
                min_results=1,
                max_response_time=5.0
            ),
            
            EvaluationQuery(
                id="fact_002", 
                query="Which repository has the most issues?",
                category="factual",
                difficulty="easy",
                expected_tools=["text_to_sql"],
                expected_keywords=["repository", "issues", "most", "highest"],
                expected_repositories=["mistralai/client-python"],
                description="Aggregation query with ranking",
                expected_sql_pattern="COUNT.*ORDER BY.*DESC",
                min_results=1,
                max_response_time=5.0
            ),
            
            EvaluationQuery(
                id="fact_003",
                query="How many open issues are there in the client-python repository?",
                category="factual", 
                difficulty="easy",
                expected_tools=["text_to_sql"],
                expected_keywords=["open", "issues", "client-python", "count"],
                expected_repositories=["mistralai/client-python"],
                description="Filtered counting query",
                expected_sql_pattern="COUNT.*WHERE.*state.*open.*client-python",
                min_results=1,
                max_response_time=5.0
            ),
        ])
        
        # === CODE SEARCH QUERIES ===
        queries.extend([
            EvaluationQuery(
                id="code_001",
                query="authentication implementation",
                category="code_search",
                difficulty="easy",
                expected_tools=["text_search", "vector_search"],
                expected_keywords=["auth", "authentication", "login", "token", "api"],
                expected_repositories=["mistralai/client-python", "mistralai/client-js"],
                description="Basic keyword search for authentication code",
                min_results=3,
                max_response_time=8.0
            ),
            
            EvaluationQuery(
                id="code_002",
                query="error handling patterns",
                category="code_search", 
                difficulty="medium",
                expected_tools=["text_search", "vector_search"],
                expected_keywords=["error", "exception", "try", "catch", "handle"],
                expected_repositories=["mistralai/client-python", "mistralai/mistral-inference"],
                description="Search for error handling code patterns",
                min_results=2,
                max_response_time=8.0
            ),
            
            EvaluationQuery(
                id="code_003",
                query="function calling implementation",
                category="code_search",
                difficulty="medium", 
                expected_tools=["text_search", "vector_search"],
                expected_keywords=["function", "call", "invoke", "execute"],
                expected_repositories=["mistralai/client-python", "mistralai/cookbook"],
                description="Search for function calling implementations",
                min_results=2,
                max_response_time=8.0
            ),
        ])
        
        # === CONCEPTUAL/SEMANTIC QUERIES ===
        queries.extend([
            EvaluationQuery(
                id="concept_001",
                query="What is fine-tuning and how does it work?",
                category="conceptual",
                difficulty="medium",
                expected_tools=["vector_search"],
                expected_keywords=["fine-tuning", "training", "model", "adaptation"],
                expected_repositories=["mistralai/mistral-finetune", "mistralai/cookbook"],
                description="Conceptual question about fine-tuning",
                min_results=2,
                max_response_time=10.0
            ),
            
            EvaluationQuery(
                id="concept_002", 
                query="How to optimize inference performance?",
                category="conceptual",
                difficulty="medium",
                expected_tools=["vector_search"],
                expected_keywords=["performance", "optimization", "inference", "speed"],
                expected_repositories=["mistralai/mistral-inference"],
                description="Performance optimization concepts",
                min_results=2,
                max_response_time=10.0
            ),
            
            EvaluationQuery(
                id="concept_003",
                query="What are the main features of the Python client?",
                category="conceptual",
                difficulty="easy",
                expected_tools=["vector_search", "text_search"],
                expected_keywords=["python", "client", "features", "api"],
                expected_repositories=["mistralai/client-python"],
                description="Feature overview question",
                min_results=2,
                max_response_time=8.0
            ),
        ])
        
        # === COMPLEX/MULTI-HOP QUERIES ===
        queries.extend([
            EvaluationQuery(
                id="complex_001",
                query="Show me recent issues about authentication problems",
                category="complex",
                difficulty="hard",
                expected_tools=["text_to_sql", "text_search"],
                expected_keywords=["issues", "authentication", "recent", "problems"],
                expected_repositories=["mistralai/client-python", "mistralai/client-js"],
                description="Multi-faceted query combining recency and topic",
                expected_sql_pattern="WHERE.*auth.*ORDER BY.*created_at",
                min_results=1,
                max_response_time=10.0
            ),
            
            EvaluationQuery(
                id="complex_002",
                query="Compare error handling approaches between Python and JavaScript clients",
                category="complex",
                difficulty="hard", 
                expected_tools=["vector_search", "text_search"],
                expected_keywords=["error", "handling", "python", "javascript", "client"],
                expected_repositories=["mistralai/client-python", "mistralai/client-js"],
                description="Comparative analysis across repositories",
                min_results=2,
                max_response_time=12.0
            ),
        ])
        
        # === EDGE CASES ===
        queries.extend([
            EvaluationQuery(
                id="edge_001",
                query="xyz123nonexistent",
                category="edge_case",
                difficulty="easy",
                expected_tools=["text_search", "vector_search", "text_to_sql"],
                expected_keywords=[],
                expected_repositories=[],
                description="Non-existent term should return no results",
                min_results=0,
                max_response_time=5.0
            ),
            
            EvaluationQuery(
                id="edge_002",
                query="",
                category="edge_case", 
                difficulty="easy",
                expected_tools=["text_search", "vector_search", "text_to_sql"],
                expected_keywords=[],
                expected_repositories=[],
                description="Empty query should be handled gracefully",
                min_results=0,
                max_response_time=2.0
            ),
        ])
        
        return queries
    
    def get_queries_by_category(self, category: str) -> List[EvaluationQuery]:
        """Get all queries in a specific category"""
        return [q for q in self.queries if q.category == category]
    
    def get_queries_by_difficulty(self, difficulty: str) -> List[EvaluationQuery]:
        """Get all queries of a specific difficulty level"""
        return [q for q in self.queries if q.difficulty == difficulty]
    
    def get_queries_by_tool(self, tool_name: str) -> List[EvaluationQuery]:
        """Get all queries that should work well with a specific tool"""
        return [q for q in self.queries if tool_name in q.expected_tools]
    
    def get_query_by_id(self, query_id: str) -> Optional[EvaluationQuery]:
        """Get a specific query by ID"""
        for query in self.queries:
            if query.id == query_id:
                return query
        return None
    
    def get_all_categories(self) -> List[str]:
        """Get list of all query categories"""
        return list(set(q.category for q in self.queries))
    
    def get_all_difficulties(self) -> List[str]:
        """Get list of all difficulty levels"""
        return list(set(q.difficulty for q in self.queries))
    
    def export_to_json(self, filepath: str) -> None:
        """Export test dataset to JSON file"""
        data = {
            "metadata": {
                "total_queries": len(self.queries),
                "categories": self.get_all_categories(),
                "difficulties": self.get_all_difficulties(),
            },
            "queries": [
                {
                    "id": q.id,
                    "query": q.query,
                    "category": q.category,
                    "difficulty": q.difficulty,
                    "expected_tools": q.expected_tools,
                    "expected_keywords": q.expected_keywords,
                    "expected_repositories": q.expected_repositories,
                    "description": q.description,
                    "expected_sql_pattern": q.expected_sql_pattern,
                    "min_results": q.min_results,
                    "max_response_time": q.max_response_time,
                }
                for q in self.queries
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics of the test dataset"""
        categories = {}
        difficulties = {}
        tools = {}
        
        for query in self.queries:
            # Count by category
            categories[query.category] = categories.get(query.category, 0) + 1
            
            # Count by difficulty
            difficulties[query.difficulty] = difficulties.get(query.difficulty, 0) + 1
            
            # Count by expected tools
            for tool in query.expected_tools:
                tools[tool] = tools.get(tool, 0) + 1
        
        return {
            "total_queries": len(self.queries),
            "categories": categories,
            "difficulties": difficulties,
            "tools": tools,
            "avg_expected_results": sum(q.min_results for q in self.queries) / len(self.queries),
            "avg_max_response_time": sum(q.max_response_time for q in self.queries) / len(self.queries),
        }


# Create default test dataset instance
default_test_dataset = EvaluationDataset()
