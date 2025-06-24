"""
CLI runner for RAG evaluation

This script provides a command-line interface for running RAG evaluations
with various configuration options.
"""

import argparse
import logging
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluations.basic_evaluator import BasicRAGEvaluator
from evaluations.test_datasets import TestDataset


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration"""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Run RAG system evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 3 queries
  python run_evaluation.py --quick-test 3

  # Test only factual queries
  python run_evaluation.py --categories factual

  # Test only easy queries with text search
  python run_evaluation.py --difficulties easy --tools text_search

  # Full evaluation with detailed report
  python run_evaluation.py --report evaluation_report.txt --export results.json

  # Test specific categories and save results
  python run_evaluation.py --categories factual code_search --max-queries 10 --export results.json
        """
    )
    
    # Basic configuration
    parser.add_argument(
        "--data-dir",
        default="data/mistral-repos",
        help="Directory containing repository data (default: data/mistral-repos)"
    )
    parser.add_argument(
        "--db-path",
        default="db.duckdb",
        help="Path to main DuckDB database (default: db.duckdb)"
    )
    parser.add_argument(
        "--vector-db-path",
        help="Path to vector database (optional, defaults to main database)"
    )
    parser.add_argument(
        "--api-key",
        help="Mistral API key (can also use MISTRAL_API_KEY env var)"
    )
    parser.add_argument(
        "--force-rebuild-vectors",
        action="store_true",
        help="Force rebuild of vector embeddings"
    )
    
    # Evaluation configuration
    parser.add_argument(
        "--categories",
        nargs="+",
        help="Query categories to test (factual, code_search, conceptual, complex, edge_case)"
    )
    parser.add_argument(
        "--difficulties",
        nargs="+",
        help="Query difficulties to test (easy, medium, hard)"
    )
    parser.add_argument(
        "--tools",
        nargs="+",
        help="Tools to test (text_search, vector_search, text_to_sql)"
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        help="Maximum number of queries to test"
    )
    
    # Output options
    parser.add_argument(
        "--report",
        help="Save detailed report to file"
    )
    parser.add_argument(
        "--export",
        help="Export results to JSON file"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    
    # Special modes
    parser.add_argument(
        "--quick-test",
        type=int,
        metavar="N",
        help="Run quick test with N queries (default: 5)"
    )
    parser.add_argument(
        "--list-queries",
        action="store_true",
        help="List available test queries and exit"
    )
    parser.add_argument(
        "--dataset-summary",
        action="store_true",
        help="Show test dataset summary and exit"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.quiet:
        setup_logging("WARNING")
    else:
        setup_logging(args.log_level)
    
    logger = logging.getLogger(__name__)
    
    # Handle special modes
    if args.list_queries:
        list_queries()
        return
    
    if args.dataset_summary:
        show_dataset_summary()
        return
    
    # Validate API key
    api_key = args.api_key or os.getenv("MISTRAL_API_KEY")
    if not api_key:
        logger.error("MISTRAL_API_KEY is required. Set it as environment variable or use --api-key")
        sys.exit(1)
    
    # Initialize evaluator
    logger.info("Initializing RAG evaluator...")
    evaluator = BasicRAGEvaluator(
        data_dir=args.data_dir,
        db_path=args.db_path,
        vector_db_path=args.vector_db_path,
        api_key=api_key,
        force_rebuild_vectors=args.force_rebuild_vectors,
    )
    
    try:
        # Handle quick test mode
        if args.quick_test is not None:
            num_queries = args.quick_test if args.quick_test > 0 else 5
            logger.info(f"Running quick test with {num_queries} queries...")
            
            result = evaluator.quick_test(num_queries)
            print("\n" + result)
            return
        
        # Run full evaluation
        logger.info("Starting evaluation...")
        results = evaluator.run_evaluation(
            query_categories=args.categories,
            query_difficulties=args.difficulties,
            tools_to_test=args.tools,
            max_queries=args.max_queries
        )
        
        if "error" in results:
            logger.error(f"Evaluation failed: {results['error']}")
            sys.exit(1)
        
        # Print summary
        print_evaluation_summary(results)
        
        # Generate and save report
        if args.report:
            logger.info("Generating detailed report...")
            report = evaluator.generate_report(args.report)
            if not args.quiet:
                print(f"\nDetailed report saved to: {args.report}")
        
        # Export results
        if args.export:
            logger.info(f"Exporting results to {args.export}...")
            evaluator.export_results(args.export)
            if not args.quiet:
                print(f"Results exported to: {args.export}")
        
        logger.info("Evaluation completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}")
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        sys.exit(1)


def list_queries():
    """List all available test queries"""
    dataset = TestDataset()
    
    print("📋 AVAILABLE TEST QUERIES")
    print("=" * 50)
    
    for category in dataset.get_all_categories():
        queries = dataset.get_queries_by_category(category)
        print(f"\n{category.upper()} ({len(queries)} queries):")
        
        for query in queries:
            tools_str = ", ".join(query.expected_tools)
            print(f"  {query.id}: {query.query[:60]}...")
            print(f"    Difficulty: {query.difficulty}, Tools: {tools_str}")


def show_dataset_summary():
    """Show test dataset summary"""
    dataset = TestDataset()
    summary = dataset.summary()
    
    print("📊 TEST DATASET SUMMARY")
    print("=" * 40)
    print(f"Total Queries: {summary['total_queries']}")
    print(f"Average Expected Results: {summary['avg_expected_results']:.1f}")
    print(f"Average Max Response Time: {summary['avg_max_response_time']:.1f}s")
    
    print("\nCategories:")
    for category, count in summary['categories'].items():
        print(f"  {category}: {count}")
    
    print("\nDifficulties:")
    for difficulty, count in summary['difficulties'].items():
        print(f"  {difficulty}: {count}")
    
    print("\nTools:")
    for tool, count in summary['tools'].items():
        print(f"  {tool}: {count}")


def print_evaluation_summary(results: dict):
    """Print a concise evaluation summary"""
    print("\n" + "=" * 60)
    print("🎯 EVALUATION SUMMARY")
    print("=" * 60)
    
    # Overall metrics
    overall = results["basic_metrics"]["overall"]
    print("\nOverall Performance:")
    print(f"  Total Queries: {overall['total_queries']}")
    print(f"  Successful Queries: {overall['successful_queries']}")
    print(f"  Success Rate: {overall['overall_success_rate']}%")
    print(f"  Total Time: {overall['total_evaluation_time']:.1f}s")
    
    # Tool performance
    print("\nTool Performance:")
    for tool, metrics in results["basic_metrics"]["by_tool"].items():
        print(f"  {tool}:")
        print(f"    Success Rate: {metrics['success_rate']}%")
        print(f"    Avg Response Time: {metrics['avg_response_time']:.2f}s")
        print(f"    Avg Results: {metrics['avg_result_count']:.1f}")
    
    # Category performance
    print("\nCategory Performance:")
    for category, metrics in results["category_metrics"].items():
        print(f"  {category}: {metrics['success_rate']}% success")
    
    # Best tools
    best_tools = results["best_tools"]
    if best_tools:
        print("\nBest Tools by Category:")
        for category, tool in best_tools.items():
            print(f"  {category}: {tool}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
