"""
Data loader for processing repository XML files and database content
"""

import logging
import re
from pathlib import Path
from typing import Dict, List

import duckdb
import pandas as pd
from llama_index.core import Document

logger = logging.getLogger(__name__)


class RepositoryDataLoader:
    """Loads and processes repository data from XML files and database"""

    def __init__(
        self,
        data_dir: str = "data/mistral-repos",
        db_path: str = "db.duckdb",
        vector_db_path: str = None,  # Use main db for vectors
        force_rebuild_vectors: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.db_path = db_path
        # Use provided vector_db_path or default to main database
        self.vector_db_path = vector_db_path if vector_db_path is not None else db_path
        self.force_rebuild_vectors = force_rebuild_vectors
        self.conn = None
        self.vector_conn = None

        # Connect to main database
        if Path(db_path).exists():
            try:
                self.conn = duckdb.connect(db_path)
                self._setup_vector_database()
            except Exception as e:
                logger.warning(f"Could not connect to main database {db_path}: {e}")
        else:
            logger.warning(f"Main database {db_path} does not exist")

        # Connect to vector database (may be same as main database)
        if self.vector_db_path != db_path:
            # Separate vector database
            try:
                self.vector_conn = duckdb.connect(self.vector_db_path)
                self._setup_vector_database()
            except Exception as e:
                logger.warning(f"Could not connect to vector database {self.vector_db_path}: {e}")
        else:
            # Use same connection for vectors
            self.vector_conn = self.conn

        # Determine if we should use vectors based on existing tables or force flag
        self.use_vectors = self._should_use_vectors()

    def get_available_repositories(self) -> List[str]:
        """Get list of available repositories"""
        repos = []
        if self.data_dir.exists():
            for repo_dir in self.data_dir.iterdir():
                if repo_dir.is_dir() and (repo_dir / "repomix-output.xml").exists():
                    repos.append(repo_dir.name)
        return repos

    def load_repository_code(self, repo_name: str) -> List[Document]:
        """Load repository source code from XML file using simple text parsing"""
        xml_path = self.data_dir / repo_name / "repomix-output.xml"
        if not xml_path.exists():
            raise FileNotFoundError(f"Repository XML not found: {xml_path}")

        documents = []

        try:
            with open(xml_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract directory structure
            dir_match = re.search(
                r"<directory_structure>(.*?)</directory_structure>", content, re.DOTALL
            )
            if dir_match:
                dir_content = dir_match.group(1).strip()
                if dir_content:
                    metadata = {"repository": repo_name, "type": "directory_structure"}
                    doc = Document(
                        text=f"Directory Structure for {repo_name}:\n\n{dir_content}",
                        metadata=metadata,
                    )
                    documents.append(doc)

            # Split content on file tags and extract files
            file_pattern = r'<file path="([^"]+)">(.*?)(?=<file path="|</files>|$)'
            file_matches = re.findall(file_pattern, content, re.DOTALL)

            for file_path, file_content in file_matches:
                # Clean up the content by removing any trailing XML tags
                file_content = file_content.strip()
                if file_content.endswith("</file>"):
                    file_content = file_content[:-7].strip()

                if file_content:
                    metadata = {
                        "repository": repo_name,
                        "file_path": file_path,
                        "type": "source_code",
                        "file_extension": Path(file_path).suffix,
                    }

                    doc = Document(
                        text=f"File: {file_path}\n\n{file_content}", metadata=metadata
                    )
                    documents.append(doc)

        except Exception as e:
            logger.error(f"Error parsing XML for {repo_name}: {e}")

        return documents

    def _setup_vector_database(self) -> None:
        """Setup vector database using LlamaIndex DuckDB vector store"""
        if not self.vector_conn:
            return

        try:
            # The DuckDBVectorStore will handle the setup automatically
            logger.info("Vector database connection established")

        except Exception as e:
            logger.error(f"Error setting up vector database: {e}")

    def _should_use_vectors(self) -> bool:
        """
        Determine if vectors should be used for vector search based on:
        1. force_rebuild_vectors flag (CLI input)
        2. Whether embedding tables exist in the vector database
        3. Default to True to initialize vectors by default
        """
        # If force rebuild is requested, always use vectors
        if self.force_rebuild_vectors:
            logger.info(
                "Force rebuild vectors flag is set - will create embeddings from scratch"
            )
            return True

        # If no vector connection, can't use vectors
        if not self.vector_conn:
            logger.warning("No vector database connection - vectors disabled")
            return False

        # Check if embedding tables exist
        try:
            tables_result = self.vector_conn.execute("SHOW TABLES").fetchall()
            table_names = [row[0] for row in tables_result]

            # Look for common embedding table names
            embedding_tables = [
                name
                for name in table_names
                if "embedding" in name.lower() or "vector" in name.lower()
            ]

            if embedding_tables:
                logger.info(
                    f"Found existing embedding tables: {embedding_tables} - vectors enabled"
                )
                return True
            else:
                logger.info(
                    "No embedding tables found in vector database - initializing vectors by default"
                )
                return True  # Initialize vectors by default for vector search

        except Exception as e:
            logger.info(
                f"Error checking vector database tables: {e} - initializing vectors by default"
            )
            return True  # Initialize vectors by default for vector search

    def load_github_data(self, repo_name: str) -> List[Document]:
        """Load GitHub issues, PRs, and other metadata from database"""
        documents = []

        # Skip if no database connection
        if not self.conn:
            logger.warning(
                f"No database connection available for loading GitHub data for {repo_name}"
            )
            return documents

        # Map repository names to database format
        db_repo_name = f"mistralai/{repo_name}"

        # Load issues with improved error handling
        issues_query = """
        SELECT title, body, state, created_at, updated_at, 
               JSON_EXTRACT_STRING(user, '$.login') as author,
               JSON_EXTRACT_STRING(labels, '$') as labels
        FROM issues 
        WHERE _sdc_repository = ? AND body IS NOT NULL AND body != ''
        ORDER BY created_at DESC
        LIMIT 100
        """

        try:
            issues_df = self.conn.execute(issues_query, [db_repo_name]).df()
            for _, issue in issues_df.iterrows():
                metadata = {
                    "repository": repo_name,
                    "type": "github_issue",
                    "state": issue["state"],
                    "author": issue["author"],
                    "created_at": str(issue["created_at"]),
                    "labels": issue["labels"],
                }

                text = f"Issue: {issue['title']}\n\n{issue['body']}"
                doc = Document(text=text, metadata=metadata)
                documents.append(doc)

        except Exception as e:
            logger.warning(f"Error loading issues for {repo_name}: {e}")

        # Load pull requests with improved error handling
        pr_query = """
        SELECT title, body, state, created_at, merged_at,
               JSON_EXTRACT_STRING(user, '$.login') as author
        FROM pull_requests 
        WHERE _sdc_repository = ? AND body IS NOT NULL AND body != ''
        ORDER BY created_at DESC
        LIMIT 50
        """

        try:
            prs_df = self.conn.execute(pr_query, [db_repo_name]).df()
            for _, pr in prs_df.iterrows():
                metadata = {
                    "repository": repo_name,
                    "type": "github_pr",
                    "state": pr["state"],
                    "author": pr["author"],
                    "created_at": str(pr["created_at"]),
                    "merged_at": str(pr["merged_at"]) if pr["merged_at"] else None,
                }

                text = f"Pull Request: {pr['title']}\n\n{pr['body']}"
                doc = Document(text=text, metadata=metadata)
                documents.append(doc)

        except Exception as e:
            logger.warning(f"Error loading PRs for {repo_name}: {e}")

        # Load comments with improved error handling
        comments_query = """
        SELECT body, created_at, author_association,
               JSON_EXTRACT_STRING(user, '$.login') as author
        FROM comments 
        WHERE _sdc_repository = ? AND body IS NOT NULL AND body != ''
        ORDER BY created_at DESC
        LIMIT 100
        """

        try:
            comments_df = self.conn.execute(comments_query, [db_repo_name]).df()
            for _, comment in comments_df.iterrows():
                metadata = {
                    "repository": repo_name,
                    "type": "github_comment",
                    "author": comment["author"],
                    "author_association": comment["author_association"],
                    "created_at": str(comment["created_at"]),
                }

                text = f"Comment by {comment['author']}:\n\n{comment['body']}"
                doc = Document(text=text, metadata=metadata)
                documents.append(doc)

        except Exception as e:
            logger.warning(f"Error loading comments for {repo_name}: {e}")

        return documents

    def load_all_repositories(self) -> List[Document]:
        """Load all available repository data"""
        import time

        start_time = time.time()

        all_documents = []
        repos = self.get_available_repositories()

        logger.info(f"Starting to load {len(repos)} repositories...")

        for i, repo in enumerate(repos, 1):
            repo_start = time.time()
            logger.info(f"Loading repository {i}/{len(repos)}: {repo}")

            # Load source code
            code_start = time.time()
            try:
                code_docs = self.load_repository_code(repo)
                all_documents.extend(code_docs)
                code_time = time.time() - code_start
                logger.info(
                    f"✅ Loaded {len(code_docs)} code documents for {repo} in {code_time:.2f}s"
                )
            except Exception as e:
                code_time = time.time() - code_start
                logger.error(
                    f"❌ Error loading code for {repo} after {code_time:.2f}s: {e}"
                )

            # Load GitHub data
            github_start = time.time()
            try:
                github_docs = self.load_github_data(repo)
                all_documents.extend(github_docs)
                github_time = time.time() - github_start
                logger.info(
                    f"✅ Loaded {len(github_docs)} GitHub documents for {repo} in {github_time:.2f}s"
                )
            except Exception as e:
                github_time = time.time() - github_start
                logger.error(
                    f"❌ Error loading GitHub data for {repo} after {github_time:.2f}s: {e}"
                )

            repo_time = time.time() - repo_start
            logger.info(f"Repository {repo} completed in {repo_time:.2f}s")

        total_time = time.time() - start_time
        logger.info(
            f"🎉 Loaded all {len(repos)} repositories with {len(all_documents)} total documents in {total_time:.2f}s"
        )

        # Log document type breakdown
        doc_types = {}
        for doc in all_documents:
            doc_type = (
                doc.metadata.get("type", "unknown")
                if hasattr(doc, "metadata")
                else "unknown"
            )
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        logger.info(f"Document breakdown: {doc_types}")

        return all_documents

    def get_database_schema(self) -> Dict[str, List[str]]:
        """Get database schema information"""
        if not self.conn:
            return {}

        try:
            tables_query = "SHOW TABLES"
            tables_df = self.conn.execute(tables_query).df()

            schema = {}
            for table_name in tables_df["name"]:
                columns_query = f"DESCRIBE {table_name}"
                columns_df = self.conn.execute(columns_query).df()
                schema[table_name] = columns_df["column_name"].tolist()

            return schema
        except Exception as e:
            logger.error(f"Error getting database schema: {e}")
            return {}

    def execute_sql_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return results"""
        if not self.conn:
            raise ValueError("No database connection available")

        try:
            return self.conn.execute(query).df()
        except Exception as e:
            logger.error(f"SQL query error: {e}")
            raise

    def get_vector_store(self):
        """Get DuckDB vector store for embeddings (only if use_vectors=True)"""
        if not self.use_vectors:
            return None

        try:
            from llama_index.vector_stores.duckdb import DuckDBVectorStore

            vector_store = DuckDBVectorStore(
                database_name=self.vector_db_path,
                persist_dir='./',
                table_name="bge_vectors",
                embed_dim=384,  # BGE-small-en-v1.5 dimension
            )
            return vector_store
        except ImportError:
            logger.error(
                "DuckDBVectorStore not available. Install with: pip install llama-index-vector-stores-duckdb"
            )
            return None
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            return None

    def close(self) -> None:
        """Close database connections"""
        if self.conn:
            self.conn.close()
        # Close vector_conn if it's different from main conn
        if self.vector_conn and self.vector_conn != self.conn:
            self.vector_conn.close()
