"""
Evaluation metrics for RAG systems

This module provides various metrics for evaluating retrieval quality,
response accuracy, and system performance.
"""

import re
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class QueryResult:
    """Represents the result of a single query evaluation"""
    
    query_id: str
    query_text: str
    tool_name: str
    success: bool
    response_time: float
    result_count: int
    results: List[Dict[str, Any]]
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EvaluationResult:
    """Represents the complete evaluation result for a query across all tools"""
    
    query_id: str
    query_text: str
    category: str
    difficulty: str
    tool_results: Dict[str, QueryResult]
    overall_success: bool
    best_tool: Optional[str] = None
    evaluation_time: float = 0.0


class EvaluationMetrics:
    """Calculate various evaluation metrics for RAG systems"""
    
    def __init__(self):
        self.results: List[EvaluationResult] = []
    
    def add_result(self, result: EvaluationResult) -> None:
        """Add an evaluation result"""
        self.results.append(result)
    
    def calculate_basic_metrics(self) -> Dict[str, Any]:
        """Calculate basic performance metrics"""
        if not self.results:
            return {"error": "No results to evaluate"}
        
        total_queries = len(self.results)
        successful_queries = sum(1 for r in self.results if r.overall_success)
        
        # Tool-specific metrics
        tool_metrics = defaultdict(lambda: {
            "total_attempts": 0,
            "successful_attempts": 0,
            "response_times": [],
            "result_counts": [],
            "error_count": 0
        })
        
        for result in self.results:
            for tool_name, tool_result in result.tool_results.items():
                metrics = tool_metrics[tool_name]
                metrics["total_attempts"] += 1
                
                if tool_result.success:
                    metrics["successful_attempts"] += 1
                    metrics["response_times"].append(tool_result.response_time)
                    metrics["result_counts"].append(tool_result.result_count)
                else:
                    metrics["error_count"] += 1
        
        # Calculate aggregated tool metrics
        tool_summary = {}
        for tool_name, metrics in tool_metrics.items():
            success_rate = (metrics["successful_attempts"] / metrics["total_attempts"]) * 100
            
            avg_response_time = (
                statistics.mean(metrics["response_times"]) 
                if metrics["response_times"] else 0
            )
            
            avg_result_count = (
                statistics.mean(metrics["result_counts"]) 
                if metrics["result_counts"] else 0
            )
            
            tool_summary[tool_name] = {
                "success_rate": round(success_rate, 2),
                "avg_response_time": round(avg_response_time, 3),
                "avg_result_count": round(avg_result_count, 1),
                "total_attempts": metrics["total_attempts"],
                "successful_attempts": metrics["successful_attempts"],
                "error_count": metrics["error_count"]
            }
        
        return {
            "overall": {
                "total_queries": total_queries,
                "successful_queries": successful_queries,
                "overall_success_rate": round((successful_queries / total_queries) * 100, 2),
                "total_evaluation_time": round(sum(r.evaluation_time for r in self.results), 2)
            },
            "by_tool": tool_summary
        }
    
    def calculate_category_metrics(self) -> Dict[str, Any]:
        """Calculate metrics by query category"""
        category_metrics = defaultdict(lambda: {
            "total": 0,
            "successful": 0,
            "response_times": [],
            "difficulties": defaultdict(int)
        })
        
        for result in self.results:
            metrics = category_metrics[result.category]
            metrics["total"] += 1
            metrics["difficulties"][result.difficulty] += 1
            
            if result.overall_success:
                metrics["successful"] += 1
                # Use the fastest successful tool's response time
                fastest_time = min(
                    tr.response_time for tr in result.tool_results.values() 
                    if tr.success
                )
                metrics["response_times"].append(fastest_time)
        
        # Calculate summary statistics
        category_summary = {}
        for category, metrics in category_metrics.items():
            success_rate = (metrics["successful"] / metrics["total"]) * 100
            avg_response_time = (
                statistics.mean(metrics["response_times"]) 
                if metrics["response_times"] else 0
            )
            
            category_summary[category] = {
                "total_queries": metrics["total"],
                "successful_queries": metrics["successful"],
                "success_rate": round(success_rate, 2),
                "avg_response_time": round(avg_response_time, 3),
                "difficulty_breakdown": dict(metrics["difficulties"])
            }
        
        return category_summary
    
    def calculate_precision_at_k(self, k: int = 5) -> Dict[str, float]:
        """Calculate Precision@K for each tool"""
        tool_precisions = defaultdict(list)
        
        for result in self.results:
            for tool_name, tool_result in result.tool_results.items():
                if tool_result.success and tool_result.results:
                    # Simple relevance check based on result count
                    # In a real scenario, you'd have ground truth relevance labels
                    relevant_results = min(tool_result.result_count, k)
                    precision = relevant_results / k if k > 0 else 0
                    tool_precisions[tool_name].append(precision)
        
        return {
            tool: round(statistics.mean(precisions), 3) if precisions else 0.0
            for tool, precisions in tool_precisions.items()
        }
    
    def calculate_response_time_percentiles(self) -> Dict[str, Dict[str, float]]:
        """Calculate response time percentiles for each tool"""
        tool_times = defaultdict(list)
        
        for result in self.results:
            for tool_name, tool_result in result.tool_results.items():
                if tool_result.success:
                    tool_times[tool_name].append(tool_result.response_time)
        
        percentiles = {}
        for tool, times in tool_times.items():
            if times:
                percentiles[tool] = {
                    "p50": round(statistics.median(times), 3),
                    "p90": round(statistics.quantiles(times, n=10)[8], 3) if len(times) >= 10 else round(max(times), 3),
                    "p95": round(statistics.quantiles(times, n=20)[18], 3) if len(times) >= 20 else round(max(times), 3),
                    "min": round(min(times), 3),
                    "max": round(max(times), 3)
                }
            else:
                percentiles[tool] = {
                    "p50": 0.0, "p90": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0
                }
        
        return percentiles
    
    def identify_best_tool_per_category(self) -> Dict[str, str]:
        """Identify the best performing tool for each category"""
        category_tool_performance = defaultdict(lambda: defaultdict(list))
        
        for result in self.results:
            for tool_name, tool_result in result.tool_results.items():
                if tool_result.success:
                    # Score based on success and speed (lower time is better)
                    score = 1.0 / (tool_result.response_time + 0.1)  # Add small constant to avoid division by zero
                    category_tool_performance[result.category][tool_name].append(score)
        
        best_tools = {}
        for category, tool_scores in category_tool_performance.items():
            if tool_scores:
                avg_scores = {
                    tool: statistics.mean(scores) 
                    for tool, scores in tool_scores.items()
                }
                best_tools[category] = max(avg_scores, key=avg_scores.get)
        
        return best_tools
    
    def check_keyword_coverage(self, expected_keywords: List[str], results: List[Dict[str, Any]]) -> float:
        """Check how many expected keywords appear in the results"""
        if not expected_keywords:
            return 1.0  # Perfect score if no keywords expected
        
        found_keywords = set()
        
        for result in results:
            content = str(result.get("content", "")).lower()
            for keyword in expected_keywords:
                if keyword.lower() in content:
                    found_keywords.add(keyword.lower())
        
        return len(found_keywords) / len(expected_keywords)
    
    def check_repository_coverage(self, expected_repos: List[str], results: List[Dict[str, Any]]) -> float:
        """Check how many expected repositories appear in the results"""
        if not expected_repos:
            return 1.0  # Perfect score if no repos expected
        
        found_repos = set()
        
        for result in results:
            metadata = result.get("metadata", {})
            repo = metadata.get("repository", "")
            
            for expected_repo in expected_repos:
                if expected_repo.lower() in repo.lower():
                    found_repos.add(expected_repo.lower())
        
        return len(found_repos) / len(expected_repos)
    
    def check_sql_pattern(self, expected_pattern: str, sql_query: str) -> bool:
        """Check if SQL query matches expected pattern"""
        if not expected_pattern or not sql_query:
            return True
        
        try:
            return bool(re.search(expected_pattern, sql_query, re.IGNORECASE | re.DOTALL))
        except re.error:
            return False
    
    def generate_detailed_report(self) -> str:
        """Generate a comprehensive evaluation report"""
        if not self.results:
            return "No evaluation results available."
        
        basic_metrics = self.calculate_basic_metrics()
        category_metrics = self.calculate_category_metrics()
        precision_metrics = self.calculate_precision_at_k()
        time_percentiles = self.calculate_response_time_percentiles()
        best_tools = self.identify_best_tool_per_category()
        
        report = []
        report.append("=" * 80)
        report.append("RAG SYSTEM EVALUATION REPORT")
        report.append("=" * 80)
        
        # Overall metrics
        overall = basic_metrics["overall"]
        report.append("\n📊 OVERALL PERFORMANCE")
        report.append(f"Total Queries: {overall['total_queries']}")
        report.append(f"Successful Queries: {overall['successful_queries']}")
        report.append(f"Overall Success Rate: {overall['overall_success_rate']}%")
        report.append(f"Total Evaluation Time: {overall['total_evaluation_time']}s")
        
        # Tool performance
        report.append("\n🔧 TOOL PERFORMANCE")
        for tool, metrics in basic_metrics["by_tool"].items():
            report.append(f"\n{tool.upper()}:")
            report.append(f"  Success Rate: {metrics['success_rate']}%")
            report.append(f"  Avg Response Time: {metrics['avg_response_time']}s")
            report.append(f"  Avg Result Count: {metrics['avg_result_count']}")
            report.append(f"  Total Attempts: {metrics['total_attempts']}")
            report.append(f"  Errors: {metrics['error_count']}")
        
        # Category performance
        report.append("\n📂 PERFORMANCE BY CATEGORY")
        for category, metrics in category_metrics.items():
            report.append(f"\n{category.upper()}:")
            report.append(f"  Success Rate: {metrics['success_rate']}%")
            report.append(f"  Avg Response Time: {metrics['avg_response_time']}s")
            report.append(f"  Total Queries: {metrics['total_queries']}")
            report.append(f"  Difficulty Breakdown: {metrics['difficulty_breakdown']}")
        
        # Best tools per category
        report.append("\n🏆 BEST TOOLS BY CATEGORY")
        for category, tool in best_tools.items():
            report.append(f"  {category}: {tool}")
        
        # Response time analysis
        report.append("\n⏱️  RESPONSE TIME PERCENTILES")
        for tool, percentiles in time_percentiles.items():
            report.append(f"\n{tool.upper()}:")
            report.append(f"  P50 (Median): {percentiles['p50']}s")
            report.append(f"  P90: {percentiles['p90']}s")
            report.append(f"  P95: {percentiles['p95']}s")
            report.append(f"  Min: {percentiles['min']}s")
            report.append(f"  Max: {percentiles['max']}s")
        
        # Precision metrics
        report.append("\n🎯 PRECISION@5 METRICS")
        for tool, precision in precision_metrics.items():
            report.append(f"  {tool}: {precision}")
        
        # Failed queries analysis
        failed_results = [r for r in self.results if not r.overall_success]
        if failed_results:
            report.append(f"\n❌ FAILED QUERIES ({len(failed_results)} total)")
            for result in failed_results[:5]:  # Show first 5 failures
                report.append(f"  - {result.query_id}: {result.query_text[:50]}...")
                for tool, tool_result in result.tool_results.items():
                    if not tool_result.success:
                        error = tool_result.error_message or "Unknown error"
                        report.append(f"    {tool}: {error[:100]}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def export_results_to_json(self, filepath: str) -> None:
        """Export evaluation results to JSON file"""
        import json
        
        data = {
            "metadata": {
                "total_results": len(self.results),
                "evaluation_timestamp": time.time(),
            },
            "basic_metrics": self.calculate_basic_metrics(),
            "category_metrics": self.calculate_category_metrics(),
            "precision_metrics": self.calculate_precision_at_k(),
            "time_percentiles": self.calculate_response_time_percentiles(),
            "best_tools": self.identify_best_tool_per_category(),
            "detailed_results": [
                {
                    "query_id": r.query_id,
                    "query_text": r.query_text,
                    "category": r.category,
                    "difficulty": r.difficulty,
                    "overall_success": r.overall_success,
                    "best_tool": r.best_tool,
                    "evaluation_time": r.evaluation_time,
                    "tool_results": {
                        tool: {
                            "success": tr.success,
                            "response_time": tr.response_time,
                            "result_count": tr.result_count,
                            "error_message": tr.error_message,
                        }
                        for tool, tr in r.tool_results.items()
                    }
                }
                for r in self.results
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def clear_results(self) -> None:
        """Clear all evaluation results"""
        self.results.clear()
