"""
Tests for the data loader module
"""

import tempfile
from pathlib import Path

import duckdb
import pytest

from src.data_loader import RepositoryDataLoader


class TestRepositoryDataLoader:
    """Test cases for RepositoryDataLoader"""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory with test XML files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "test-repos"
            data_dir.mkdir()

            # Create test repository directory
            repo_dir = data_dir / "test-repo"
            repo_dir.mkdir()

            # Create test XML file
            xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<repository>
    <directory_structure>
src/
  main.py
README.md
</directory_structure>
    <files>
        <file path="src/main.py">
def main():
    print("Hello World")
        </file>
        <file path="README.md">
# Test Repository
This is a test repository.
        </file>
    </files>
</repository>"""

            xml_file = repo_dir / "repomix-output.xml"
            xml_file.write_text(xml_content)

            yield str(data_dir)

    @pytest.fixture
    def temp_db(self):
        """Create temporary database"""
        import os

        temp_fd, db_path = tempfile.mkstemp(suffix=".duckdb")
        os.close(temp_fd)
        os.unlink(db_path)

        # Create test database with sample data
        conn = duckdb.connect(db_path)

        # Create test tables with all required columns
        conn.execute(
            """
            CREATE TABLE issues (
                title VARCHAR,
                body VARCHAR,
                state VARCHAR,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                user JSON,
                labels JSON,
                _sdc_repository VARCHAR
            )
        """
        )

        conn.execute(
            """
            CREATE TABLE pull_requests (
                title VARCHAR,
                body VARCHAR,
                state VARCHAR,
                created_at TIMESTAMP,
                merged_at TIMESTAMP,
                user JSON,
                _sdc_repository VARCHAR
            )
        """
        )

        conn.execute(
            """
            CREATE TABLE comments (
                body VARCHAR,
                created_at TIMESTAMP,
                author_association VARCHAR,
                user JSON,
                _sdc_repository VARCHAR
            )
        """
        )

        # Insert test data
        conn.execute(
            """
            INSERT INTO issues VALUES 
            ('Test Issue', 'This is a test issue', 'open', '2024-01-01', '2024-01-02', 
             '{"login": "testuser"}', '["bug", "enhancement"]', 'mistralai/test-repo'),
            ('Another Issue', 'Another test issue', 'closed', '2024-01-03', '2024-01-04',
             '{"login": "anotheruser"}', '["documentation"]', 'mistralai/test-repo')
        """
        )

        conn.execute(
            """
            INSERT INTO pull_requests VALUES
            ('Test PR', 'This is a test PR', 'merged', '2024-01-01', '2024-01-02',
             '{"login": "testuser"}', 'mistralai/test-repo')
        """
        )

        conn.execute(
            """
            INSERT INTO comments VALUES
            ('Great work!', '2024-01-01', 'MEMBER', '{"login": "reviewer"}', 'mistralai/test-repo')
        """
        )

        conn.close()

        yield db_path

        # Cleanup
        Path(db_path).unlink(missing_ok=True)

    def test_init_and_repositories(self, temp_data_dir, temp_db):
        """Test RepositoryDataLoader initialization and repository discovery"""
        loader = RepositoryDataLoader(temp_data_dir, temp_db)
        assert loader.data_dir == Path(temp_data_dir)
        assert loader.db_path == temp_db
        assert loader.conn is not None

        # Test getting available repositories
        repos = loader.get_available_repositories()
        assert "test-repo" in repos
        assert len(repos) == 1

    def test_load_data_and_operations(self, temp_data_dir, temp_db):
        """Test loading repository code and GitHub data"""
        loader = RepositoryDataLoader(temp_data_dir, temp_db)

        # Test repository code loading
        code_documents = loader.load_repository_code("test-repo")
        assert len(code_documents) > 0

        file_docs = [
            doc for doc in code_documents if doc.metadata.get("type") == "source_code"
        ]
        assert len(file_docs) == 2  # main.py and README.md

        main_py_doc = next(
            doc for doc in file_docs if "main.py" in doc.metadata.get("file_path", "")
        )
        assert "def main():" in main_py_doc.text
        assert main_py_doc.metadata["repository"] == "test-repo"

        # Test GitHub data loading
        github_documents = loader.load_github_data("test-repo")
        assert len(github_documents) > 0

        issue_docs = [
            doc
            for doc in github_documents
            if doc.metadata.get("type") == "github_issue"
        ]
        pr_docs = [
            doc for doc in github_documents if doc.metadata.get("type") == "github_pr"
        ]

        assert len(issue_docs) == 2
        assert len(pr_docs) == 1

        issue_doc = issue_docs[0]
        assert "Test Issue" in issue_doc.text or "Another Issue" in issue_doc.text
        assert issue_doc.metadata["repository"] == "test-repo"

    def test_database_operations(self, temp_data_dir, temp_db):
        """Test database schema and SQL operations"""
        loader = RepositoryDataLoader(temp_data_dir, temp_db)

        # Test schema
        schema = loader.get_database_schema()
        assert "issues" in schema
        assert "pull_requests" in schema
        assert "title" in schema["issues"]
        assert "_sdc_repository" in schema["issues"]

        # Test SQL query
        result = loader.execute_sql_query("SELECT COUNT(*) as count FROM issues")
        assert len(result) == 1
        assert result.iloc[0]["count"] == 2

        # Test query with WHERE clause
        result = loader.execute_sql_query(
            "SELECT title FROM issues WHERE state = 'open'"
        )
        assert len(result) == 1
        assert result.iloc[0]["title"] == "Test Issue"

    def test_load_all_repositories(self, temp_data_dir, temp_db):
        """Test loading all repositories"""
        loader = RepositoryDataLoader(temp_data_dir, temp_db)
        documents = loader.load_all_repositories()

        assert len(documents) > 0

        # Should have both code and GitHub documents
        code_docs = [
            doc for doc in documents if doc.metadata.get("type") == "source_code"
        ]
        github_docs = [
            doc for doc in documents if doc.metadata.get("type").startswith("github_")
        ]

        assert len(code_docs) > 0
        assert len(github_docs) > 0

    def test_vector_support(self, temp_data_dir, temp_db):
        """Test vector support functionality"""
        # Test with vectors disabled - but note that use_vectors defaults to True
        # unless there's no vector connection
        loader = RepositoryDataLoader(
            temp_data_dir, temp_db, force_rebuild_vectors=False
        )
        # The loader defaults to use_vectors=True when a database connection exists
        assert loader.use_vectors is True
        assert loader.vector_conn is not None

        vector_store = loader.get_vector_store()
        # Since use_vectors=True, vector store should be available
        assert vector_store is not None

        # Test with vectors enabled (mock)
        import os

        temp_fd, vector_db_path = tempfile.mkstemp(suffix=".duckdb")
        os.close(temp_fd)
        os.unlink(vector_db_path)

        try:
            loader_with_vectors = RepositoryDataLoader(
                temp_data_dir,
                temp_db,
                vector_db_path=vector_db_path,
                force_rebuild_vectors=True,
            )

            assert loader_with_vectors.use_vectors is True
            assert loader_with_vectors.vector_conn is not None
            assert loader_with_vectors.vector_db_path == vector_db_path

            # Should have separate connections
            assert loader_with_vectors.conn != loader_with_vectors.vector_conn
            assert loader_with_vectors.db_path != loader_with_vectors.vector_db_path

            loader_with_vectors.close()

        finally:
            Path(vector_db_path).unlink(missing_ok=True)
