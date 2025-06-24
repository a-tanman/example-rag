# RAG Evaluation Framework

A comprehensive evaluation framework for assessing the performance of your RAG (Retrieval-Augmented Generation) system. This framework provides automated testing, detailed metrics, and performance analysis across multiple search modalities.

## 🎯 Features

### **Comprehensive Testing**
- **17 predefined test queries** across 5 categories
- **Multiple difficulty levels** (easy, medium, hard)
- **Tool-specific validation** for each search method
- **Edge case testing** for robustness

### **Multi-Modal Evaluation**
- **Text Search**: BM25-based keyword search evaluation
- **Vector Search**: Semantic similarity assessment
- **Text-to-SQL**: Query generation and execution validation

### **Rich Metrics**
- **Success rates** and response times
- **Precision@K** measurements
- **Category-specific performance** analysis
- **Response time percentiles**
- **Keyword and repository coverage** analysis

### **Flexible Execution**
- **Quick tests** for rapid feedback
- **Category filtering** for focused evaluation
- **Tool selection** for specific testing
- **Detailed reporting** and JSON export

## 📁 Framework Structure

```
evaluations/
├── __init__.py              # Package initialization
├── test_datasets.py         # Test queries and expected results
├── metrics.py               # Evaluation metrics and calculations
├── basic_evaluator.py       # Main evaluation framework
├── run_evaluation.py        # CLI runner
└── README.md               # This documentation
```

## 🚀 Quick Start

### 1. **Basic Demo**
```bash
# Run the demo script
python demo_evaluation.py
```

### 2. **Quick Test**
```bash
# Test with 3 queries
python evaluations/run_evaluation.py --quick-test 3
```

### 3. **Category-Specific Testing**
```bash
# Test only factual queries
python evaluations/run_evaluation.py --categories factual

# Test code search queries
python evaluations/run_evaluation.py --categories code_search
```

### 4. **Full Evaluation with Report**
```bash
# Complete evaluation with detailed report
python evaluations/run_evaluation.py --report evaluation_report.txt --export results.json
```

## 📊 Test Dataset

The framework includes 17 carefully designed test queries:

### **Categories**
- **Factual** (3 queries): Statistical and counting queries
- **Code Search** (3 queries): Keyword-based code searches
- **Conceptual** (3 queries): Semantic understanding questions
- **Complex** (2 queries): Multi-hop reasoning queries
- **Edge Cases** (2 queries): Error handling and robustness

### **Example Queries**

**Factual:**
- "How many repositories are available?"
- "Which repository has the most issues?"
- "How many open issues are there in the client-python repository?"

**Code Search:**
- "authentication implementation"
- "error handling patterns"
- "function calling implementation"

**Conceptual:**
- "What is fine-tuning and how does it work?"
- "How to optimize inference performance?"
- "What are the main features of the Python client?"

**Complex:**
- "Show me recent issues about authentication problems"
- "Compare error handling approaches between Python and JavaScript clients"

## 🔧 Usage Examples

### **Command Line Interface**

```bash
# View available test queries
python evaluations/run_evaluation.py --list-queries

# Show dataset summary
python evaluations/run_evaluation.py --dataset-summary

# Test specific tools only
python evaluations/run_evaluation.py --tools text_search vector_search

# Test specific difficulty levels
python evaluations/run_evaluation.py --difficulties easy medium

# Limit number of queries
python evaluations/run_evaluation.py --max-queries 5

# Quiet mode with minimal output
python evaluations/run_evaluation.py --quiet --quick-test 3
```

### **Programmatic Usage**

```python
from evaluations.basic_evaluator import BasicRAGEvaluator

# Initialize evaluator
evaluator = BasicRAGEvaluator(
    data_dir="data/mistral-repos",
    db_path="db.duckdb",
    api_key="your-mistral-api-key"
)

# Quick test
result = evaluator.quick_test(num_queries=5)
print(result)

# Full evaluation
results = evaluator.run_evaluation(
    query_categories=["factual", "code_search"],
    max_queries=10
)

# Generate report
report = evaluator.generate_report("my_evaluation_report.txt")

# Export detailed results
evaluator.export_results("detailed_results.json")
```

## 📈 Metrics and Analysis

### **Basic Metrics**
- **Overall Success Rate**: Percentage of queries that returned valid results
- **Tool Success Rates**: Individual performance of each search tool
- **Average Response Times**: Speed analysis per tool
- **Result Counts**: Number of results returned per query

### **Advanced Metrics**
- **Precision@K**: Relevance of top-K results
- **Keyword Coverage**: How well results match expected keywords
- **Repository Coverage**: Accuracy of repository targeting
- **SQL Pattern Matching**: Correctness of generated SQL queries

### **Performance Analysis**
- **Response Time Percentiles**: P50, P90, P95 analysis
- **Category Performance**: Success rates by query type
- **Best Tool Identification**: Optimal tool for each category
- **Error Analysis**: Common failure patterns

## 📋 Sample Output

```
🎯 EVALUATION SUMMARY
============================================================

Overall Performance:
  Total Queries: 10
  Successful Queries: 8
  Success Rate: 80.0%
  Total Time: 45.2s

Tool Performance:
  text_search:
    Success Rate: 90.0%
    Avg Response Time: 0.25s
    Avg Results: 4.2
  vector_search:
    Success Rate: 70.0%
    Avg Response Time: 2.1s
    Avg Results: 3.8
  text_to_sql:
    Success Rate: 80.0%
    Avg Response Time: 1.5s
    Avg Results: 1.0

Category Performance:
  factual: 100.0% success
  code_search: 66.7% success
  conceptual: 75.0% success

Best Tools by Category:
  factual: text_to_sql
  code_search: text_search
  conceptual: vector_search
============================================================
```

## 🔍 Validation Criteria

### **Text/Vector Search Validation**
- **Minimum result count** met
- **Keyword coverage** ≥ 30% OR **Repository coverage** ≥ 50%
- **No error messages** in results
- **Response time** within limits

### **SQL Query Validation**
- **SQL query generated** successfully
- **Query executed** without errors
- **Expected SQL patterns** matched (if specified)
- **Results returned** (for non-zero expectations)

### **Overall Success Criteria**
A query is considered successful if:
1. At least one tool returns valid results
2. Results meet the validation criteria
3. Response time is within acceptable limits
4. No critical errors occurred

## ⚙️ Configuration

### **Environment Variables**
```bash
# Required
export MISTRAL_API_KEY="your-mistral-api-key"

# Optional
export DATA_DIR="data/mistral-repos"
export DB_PATH="db.duckdb"
```

### **Evaluation Settings**
- **Timeout**: 30 seconds per query
- **Max Results**: 10 per tool
- **Retry Attempts**: 2 per failed query
- **Validation Thresholds**: 30% keyword, 50% repository coverage

## 🛠️ Extending the Framework

### **Adding New Test Queries**
```python
# In test_datasets.py
TestQuery(
    id="custom_001",
    query="Your custom query here",
    category="custom",
    difficulty="medium",
    expected_tools=["text_search"],
    expected_keywords=["keyword1", "keyword2"],
    expected_repositories=["mistralai/repo-name"],
    description="Description of what this tests",
    min_results=1,
    max_response_time=10.0
)
```

### **Custom Metrics**
```python
# In metrics.py
def calculate_custom_metric(self) -> Dict[str, float]:
    """Add your custom metric calculation"""
    # Implementation here
    pass
```

### **New Validation Rules**
```python
# In basic_evaluator.py
def _validate_custom_results(self, test_query, results, tool_name) -> bool:
    """Add custom validation logic"""
    # Implementation here
    pass
```

## 🐛 Troubleshooting

### **Common Issues**

**1. API Key Error**
```
Error: MISTRAL_API_KEY is required
```
**Solution**: Set your Mistral API key as an environment variable

**2. No Repositories Found**
```
Error: No repositories found in data/mistral-repos
```
**Solution**: Ensure repository data files are present in the data directory

**3. Vector Search Timeout**
```
Tool vector_search failed: Operation timed out
```
**Solution**: Vector embeddings may need to be rebuilt or the timeout increased

**4. SQL Generation Errors**
```
SQL Error: syntax error at or near "SELECT"
```
**Solution**: Check database schema and ensure proper SQL context

### **Debug Mode**
```bash
# Enable detailed logging
python evaluations/run_evaluation.py --log-level DEBUG --quick-test 1
```

## 📊 Performance Benchmarks

### **Expected Performance**
- **Text Search**: ~0.2s response time, 90%+ success rate
- **Vector Search**: ~2s response time, 70%+ success rate  
- **Text-to-SQL**: ~1.5s response time, 80%+ success rate

### **Optimization Tips**
1. **Pre-build vector embeddings** for faster startup
2. **Use connection pooling** for database queries
3. **Cache frequent queries** for repeated testing
4. **Limit result counts** for faster evaluation

## 🤝 Contributing

1. **Add test queries** for new use cases
2. **Implement additional metrics** for deeper analysis
3. **Create specialized evaluators** for specific domains
4. **Improve validation logic** for better accuracy
5. **Add visualization tools** for result analysis

## 📄 License

This evaluation framework is part of the RAG system project and follows the same licensing terms.

---

**Built for comprehensive RAG system evaluation** 🚀
