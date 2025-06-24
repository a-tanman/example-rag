"""
Tests for the RAG tools module
"""

from unittest.mock import Mock, patch

import pandas as pd
import pytest
from llama_index.core import Document

from src.data_loader import RepositoryDataLoader
from src.rag_tools import (
    RAGToolManager,
    TextSearchTool,
    TextToSQLTool,
    VectorSearchTool,
)


class TestTextSearchTool:
    """Test cases for TextSearchTool"""

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing"""
        return [
            Document(
                text="def authenticate_user(username, password): return True",
                metadata={
                    "repository": "client-python",
                    "file_path": "auth.py",
                    "type": "source_code",
                },
            ),
            Document(
                text="Issue: Authentication error when using API key",
                metadata={
                    "repository": "client-python",
                    "type": "github_issue",
                    "author": "user1",
                },
            ),
        ]

    def test_init_and_search(self, sample_documents):
        """Test TextSearchTool initialization and basic search"""
        tool = TextSearchTool(sample_documents)
        assert tool.name == "text_search"
        assert "BM25 ranking algorithm" in tool.description

        # Test search functionality
        results = tool.search("authentication")
        assert len(results) >= 1
        assert results[0]["score"] > 0
        assert "content" in results[0]
        assert "metadata" in results[0]


class TestVectorSearchTool:
    """Test cases for VectorSearchTool"""

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing"""
        return [
            Document(
                text="Machine learning model for natural language processing",
                metadata={"repository": "mistral-inference", "type": "source_code"},
            )
        ]

    def test_init_no_api_key(self, sample_documents):
        """Test initialization without API key"""
        with patch.dict("os.environ", {}, clear=True):
            # VectorSearchTool will fail when trying to build the index due to missing api key
            with pytest.raises(ValueError, match="No API key found"):
                VectorSearchTool(sample_documents, api_key=None)


class TestTextToSQLTool:
    """Test cases for TextToSQLTool"""

    @pytest.fixture
    def mock_data_loader(self):
        """Create mock data loader"""
        loader = Mock(spec=RepositoryDataLoader)
        loader.get_database_schema.return_value = {
            "issues": ["title", "body", "state", "_sdc_repository"]
        }
        return loader

    @patch("src.rag_tools.MistralAI")
    def test_init_and_search(self, mock_llm, mock_data_loader):
        """Test TextToSQLTool initialization and search"""
        mock_llm_instance = Mock()
        mock_response = Mock()
        mock_response.text = "SELECT COUNT(*) FROM issues WHERE state = 'open' AND _sdc_repository = 'mistralai/client-python'"
        mock_llm_instance.complete.return_value = mock_response
        mock_llm.return_value = mock_llm_instance

        # Mock successful query execution
        mock_df = pd.DataFrame({"count": [5]})
        mock_data_loader.execute_sql_query.return_value = mock_df

        tool = TextToSQLTool(mock_data_loader, api_key="test_key")
        assert tool.name == "text_to_sql"

        results = tool.search("How many open issues?")
        assert len(results) == 1
        assert "sql_query" in results[0]
        # The TextToSQLTool returns results in a different format
        assert "results" in results[0]

    @patch("src.rag_tools.MistralAI")
    def test_duckdb_json_queries(self, mock_llm, mock_data_loader):
        """Test DuckDB-specific JSON queries"""
        mock_llm_instance = Mock()
        mock_response = Mock()
        mock_response.text = "SELECT title, JSON_EXTRACT_STRING(user, '$.login') as author FROM issues WHERE _sdc_repository = 'mistralai/client-python' LIMIT 10"
        mock_llm_instance.complete.return_value = mock_response
        mock_llm.return_value = mock_llm_instance

        # Mock successful query execution with JSON data
        mock_df = pd.DataFrame({"title": ["Test Issue"], "author": ["testuser"]})
        mock_data_loader.execute_sql_query.return_value = mock_df

        tool = TextToSQLTool(mock_data_loader, api_key="test_key")
        results = tool.search("Show me issues with authors")

        assert len(results) == 1
        assert "sql_query" in results[0]
        assert "JSON_EXTRACT_STRING" in results[0]["sql_query"]
        # The TextToSQLTool returns results in a different format
        assert "results" in results[0]
        assert any("testuser" in str(row) for row in results[0]["results"])


class TestRAGToolManager:
    """Test cases for RAGToolManager"""

    @pytest.fixture
    def mock_data_loader(self):
        """Create mock data loader"""
        loader = Mock(spec=RepositoryDataLoader)
        loader.get_database_schema.return_value = {"issues": ["title", "body"]}
        loader.load_all_repositories.return_value = [
            Document(text="Sample code", metadata={"repository": "test-repo"})
        ]
        loader.use_vectors = False
        return loader

    @patch("src.rag_tools.MistralAI")
    @patch("src.rag_tools.TextSearchTool")
    @patch("src.rag_tools.TextToSQLTool")
    def test_init_and_search(
        self, mock_sql_tool, mock_text_tool, mock_llm, mock_data_loader
    ):
        """Test RAGToolManager initialization and search"""
        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance

        # Mock tool instances
        mock_text_instance = Mock()
        mock_text_instance.search.return_value = [{"content": "text result"}]
        mock_text_tool.return_value = mock_text_instance

        mock_sql_instance = Mock()
        mock_sql_tool.return_value = mock_sql_instance

        manager = RAGToolManager(mock_data_loader, api_key="test_key")
        assert manager.data_loader == mock_data_loader

        # Test search
        results = manager.search("test query", "text_search")
        assert len(results) == 1
        assert results[0]["content"] == "text result"

    @patch("src.rag_tools.MistralAI")
    @patch("src.rag_tools.TextSearchTool")
    @patch("src.rag_tools.TextToSQLTool")
    def test_search_all_and_generate_response(
        self, mock_sql_tool, mock_text_tool, mock_llm, mock_data_loader
    ):
        """Test search_all and response generation"""
        mock_llm_instance = Mock()
        mock_response = Mock()
        mock_response.text = "Generated response"
        mock_llm_instance.complete.return_value = mock_response
        mock_llm.return_value = mock_llm_instance

        # Mock tool instances
        mock_text_instance = Mock()
        mock_text_instance.search.return_value = [{"content": "text result"}]
        mock_text_tool.return_value = mock_text_instance

        mock_sql_instance = Mock()
        mock_sql_instance.search.return_value = [{"sql_query": "SELECT * FROM issues"}]
        mock_sql_tool.return_value = mock_sql_instance

        manager = RAGToolManager(mock_data_loader, api_key="test_key")

        # Test search_all
        results = manager.search_all("test query")
        assert "text_search" in results
        assert "text_to_sql" in results

        # Test response generation
        response = manager.generate_response("test query", results)
        assert response == "Generated response"


class TestTextToSQLHighestIssuesQuery:
    """Test cases specifically for the 'highest number of issues' query"""

    @pytest.fixture
    def mock_data_loader_with_issues(self):
        """Create mock data loader with realistic issues data"""
        loader = Mock(spec=RepositoryDataLoader)
        loader.get_database_schema.return_value = {
            "issues": ["title", "body", "state", "_sdc_repository", "created_at", "user"],
            "pull_requests": ["title", "body", "state", "_sdc_repository", "user"],
            "comments": ["body", "_sdc_repository", "user", "created_at"]
        }
        return loader

    @patch("src.rag_tools.MistralAI")
    def test_highest_issues_query_returns_values(self, mock_llm, mock_data_loader_with_issues):
        """Test that 'which repository has the highest number of issues?' returns actual values"""
        # Setup mock LLM to generate appropriate SQL
        mock_llm_instance = Mock()
        mock_response = Mock()
        mock_response.text = """
        SELECT _sdc_repository, COUNT(*) as issue_count 
        FROM issues 
        GROUP BY _sdc_repository 
        ORDER BY issue_count DESC 
        LIMIT 10
        """
        mock_llm_instance.complete.return_value = mock_response
        mock_llm.return_value = mock_llm_instance

        # Mock realistic query results
        mock_df = pd.DataFrame({
            "_sdc_repository": [
                "mistralai/client-python",
                "mistralai/mistral-inference", 
                "mistralai/client-js",
                "mistralai/cookbook",
                "mistralai/mistral-common"
            ],
            "issue_count": [45, 32, 28, 15, 12]
        })
        mock_data_loader_with_issues.execute_sql_query.return_value = mock_df

        # Initialize tool and execute query
        tool = TextToSQLTool(mock_data_loader_with_issues, api_key="test_key")
        results = tool.search("which repository has the highest number of issues?")

        # Verify results structure
        assert len(results) == 1, "Should return exactly one result"
        result = results[0]
        
        # Verify required fields are present
        assert "sql_query" in result, "Result should contain SQL query"
        # The TextToSQLTool doesn't return markdown_results, check for results instead
        assert "results" in result, "Result should contain results"
        assert "row_count" in result, "Result should contain row count"
        
        # Verify SQL query was generated and executed
        assert "SELECT" in result["sql_query"].upper(), "Should contain SELECT statement"
        assert "COUNT" in result["sql_query"].upper(), "Should contain COUNT function"
        assert "_sdc_repository" in result["sql_query"], "Should query repository column"
        assert "GROUP BY" in result["sql_query"].upper(), "Should group by repository"
        assert "ORDER BY" in result["sql_query"].upper(), "Should order results"
        
        # Verify data was returned
        assert result["row_count"] == 5, "Should return 5 repositories"
        assert result["limited_to"] == 5, "Should show all 5 results"
        
        # Verify results contain the data
        results_data = result["results"]
        assert len(results_data) > 0, "Should contain result data"
        assert any("mistralai/client-python" in str(row) for row in results_data), "Should contain top repository"
        assert any("45" in str(row) for row in results_data), "Should contain highest issue count"
        
        # Verify the LLM was called with appropriate context
        mock_llm_instance.complete.assert_called_once()
        call_args = mock_llm_instance.complete.call_args[0][0]
        assert "which repository has the highest number of issues?" in call_args
        assert "Database Schema" in call_args
        assert "_sdc_repository" in call_args

    @patch("src.rag_tools.MistralAI")
    def test_highest_issues_query_handles_empty_results(self, mock_llm, mock_data_loader_with_issues):
        """Test handling when no issues are found"""
        # Setup mock LLM
        mock_llm_instance = Mock()
        mock_response = Mock()
        mock_response.text = "SELECT _sdc_repository, COUNT(*) as issue_count FROM issues GROUP BY _sdc_repository ORDER BY issue_count DESC"
        mock_llm_instance.complete.return_value = mock_response
        mock_llm.return_value = mock_llm_instance

        # Mock empty results
        mock_df = pd.DataFrame(columns=["_sdc_repository", "issue_count"])
        mock_data_loader_with_issues.execute_sql_query.return_value = mock_df

        tool = TextToSQLTool(mock_data_loader_with_issues, api_key="test_key")
        results = tool.search("which repository has the highest number of issues?")

        assert len(results) == 1
        result = results[0]
        assert result["row_count"] == 0
        # For empty results, check that results list is empty
        assert result["results"] == []

    @patch("src.rag_tools.MistralAI")
    def test_highest_issues_query_handles_sql_error(self, mock_llm, mock_data_loader_with_issues):
        """Test handling when SQL execution fails"""
        # Setup mock LLM
        mock_llm_instance = Mock()
        mock_response = Mock()
        mock_response.text = "INVALID SQL QUERY"
        mock_llm_instance.complete.return_value = mock_response
        mock_llm.return_value = mock_llm_instance

        # Mock SQL execution error
        mock_data_loader_with_issues.execute_sql_query.side_effect = Exception("SQL syntax error")

        tool = TextToSQLTool(mock_data_loader_with_issues, api_key="test_key")
        results = tool.search("which repository has the highest number of issues?")

        assert len(results) == 1
        result = results[0]
        assert "error" in result
        assert "SQL syntax error" in result["error"]

    @patch("src.rag_tools.MistralAI")
    def test_highest_issues_query_with_realistic_schema(self, mock_llm, mock_data_loader_with_issues):
        """Test with more realistic database schema context"""
        # Setup mock LLM with more detailed schema
        mock_llm_instance = Mock()
        mock_response = Mock()
        mock_response.text = """
        SELECT _sdc_repository, 
               COUNT(*) as total_issues,
               COUNT(CASE WHEN state = 'open' THEN 1 END) as open_issues,
               COUNT(CASE WHEN state = 'closed' THEN 1 END) as closed_issues
        FROM issues 
        GROUP BY _sdc_repository 
        ORDER BY total_issues DESC 
        LIMIT 5
        """
        mock_llm_instance.complete.return_value = mock_response
        mock_llm.return_value = mock_llm_instance

        # Mock detailed results
        mock_df = pd.DataFrame({
            "_sdc_repository": ["mistralai/client-python", "mistralai/mistral-inference"],
            "total_issues": [45, 32],
            "open_issues": [12, 8],
            "closed_issues": [33, 24]
        })
        mock_data_loader_with_issues.execute_sql_query.return_value = mock_df

        tool = TextToSQLTool(mock_data_loader_with_issues, api_key="test_key")
        results = tool.search("which repository has the highest number of issues?")

        result = results[0]
        assert result["row_count"] == 2
        
        # Verify detailed data is in results
        results_data = result["results"]
        assert len(results_data) == 2, "Should return 2 repositories"
        assert any("total_issues" in str(row) for row in results_data), "Should contain total_issues"
        assert any("45" in str(row) for row in results_data), "Should contain highest count"
        assert any("32" in str(row) for row in results_data), "Should contain second highest count"
