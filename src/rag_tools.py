"""
RAG tools for text search, vector search, and text-to-SQL functionality using Mistral APIs
"""

import logging
import math
import os
import re
import time
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from colorama import Fore, Style
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.mistralai import MistralAI
from llama_index.vector_stores.duckdb import DuckDBVectorStore

from .data_loader import RepositoryDataLoader

logger = logging.getLogger(__name__)


def _colorize(text: str, color: str = "") -> str:
    """Apply color to text if colors are enabled"""
    return f"{color}{text}{Style.RESET_ALL}"


def _print_separator(char: str = "=", length: int = 80) -> None:
    """Print a colored separator line"""
    separator = char * length
    print(_colorize(separator, Fore.CYAN))


def _log_search_start(tool_name: str, query: str) -> float:
    """Log the start of a search operation"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n{_colorize('🔍 SEARCH START', Fore.YELLOW + Style.BRIGHT)} [{timestamp}]")
    print(
        f"{_colorize('Tool:', Fore.CYAN)} {_colorize(tool_name.upper(), Fore.WHITE + Style.BRIGHT)}"
    )
    print(f"{_colorize('Query:', Fore.CYAN)} {_colorize(query, Fore.WHITE)}")
    return time.time()


def _log_search_results(
    tool_name: str,
    query: str,
    results: List[Dict[str, Any]],
    start_time: float,
    error: Optional[str] = None,
) -> None:
    """Log the results of a search operation"""
    duration = time.time() - start_time
    timestamp = datetime.now().strftime("%H:%M:%S")

    if error:
        print(
            f"{_colorize('❌ SEARCH ERROR', Fore.RED + Style.BRIGHT)} [{timestamp}] ({duration:.2f}s)"
        )
        print(f"{_colorize('Error:', Fore.RED)} {error}")
        return

    print(
        f"{_colorize('✅ SEARCH COMPLETE', Fore.GREEN + Style.BRIGHT)} [{timestamp}] ({duration:.2f}s)"
    )
    print(f"{_colorize('Results found:', Fore.GREEN)} {len(results)}")

    if tool_name == "text_to_sql" and results:
        # Special handling for SQL results
        result = results[0]
        if "sql_query" in result:
            print(f"{_colorize('Generated SQL:', Fore.BLUE)}")
            print(f"  {_colorize(result['sql_query'], Fore.WHITE)}")

        if "results" in result and result["results"]:
            row_count = result.get("row_count", len(result["results"]))
            limited_to = result.get("limited_to", len(result["results"]))
            print(
                f"{_colorize('Data rows:', Fore.BLUE)} {row_count} total, showing {limited_to}"
            )

            # Show first few rows
            for i, row in enumerate(result["results"][:3]):
                print(
                    f"  {_colorize(f'Row {i+1}:', Fore.MAGENTA)} {str(row)[:100]}{'...' if len(str(row)) > 100 else ''}"
                )

        elif "error" in result:
            print(f"{_colorize('SQL Error:', Fore.RED)} {result['error']}")

    else:
        # Regular search results
        for i, result in enumerate(results[:3]):  # Show top 3 results
            score = result.get("score", "N/A")
            content = result.get("content", str(result))
            metadata = result.get("metadata", {})

            print(f"{_colorize(f'Result {i+1}:', Fore.MAGENTA)} Score: {score}")

            # Show repository and file info if available
            if metadata:
                repo = metadata.get("repository", metadata.get("repo", ""))
                file_path = metadata.get("file_path", metadata.get("path", ""))
                if repo:
                    print(f"  {_colorize('Repository:', Fore.CYAN)} {repo}")
                if file_path:
                    print(f"  {_colorize('File:', Fore.CYAN)} {file_path}")

            # Show content preview
            preview = content[:150] + "..." if len(content) > 150 else content
            print(f"  {_colorize('Content:', Fore.WHITE)} {preview}")

    _print_separator("-", 60)


class BaseRAGTool(ABC):
    """Base class for RAG tools"""

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description

    @abstractmethod
    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Execute search and return results"""
        pass


class TextSearchTool(BaseRAGTool):
    """Full-text search tool using BM25 ranking algorithm"""

    def __init__(self, documents: List[Document], k1: float = 1.2, b: float = 0.75) -> None:
        super().__init__(
            name="text_search",
            description="Search through repository content using BM25 ranking algorithm. Good for finding specific code patterns, function names, or exact text matches with relevance scoring.",
        )
        self.documents = documents
        self.k1 = k1  # Controls term frequency saturation
        self.b = b    # Controls document length normalization
        self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        """Build BM25 index with term frequencies and document statistics"""
        logger.info(f"Building BM25 index for {len(self.documents)} documents...")
        
        # Initialize data structures
        self.doc_texts: List[str] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_lengths: List[int] = []
        self.term_frequencies: List[Dict[str, int]] = []
        self.document_frequencies: Dict[str, int] = defaultdict(int)
        self.vocabulary: set = set()
        
        # Process each document
        for i, doc in enumerate(self.documents):
            # Create searchable text combining content and metadata
            searchable_text = doc.text.lower()
            if hasattr(doc, "metadata") and doc.metadata:
                for key, value in doc.metadata.items():
                    if isinstance(value, str):
                        searchable_text += f" {value.lower()}"
            
            # Tokenize text
            tokens = re.findall(r"\w+", searchable_text)
            
            # Calculate term frequencies for this document
            tf = Counter(tokens)
            
            # Store document data
            self.doc_texts.append(searchable_text)
            self.doc_tokens.append(tokens)
            self.doc_lengths.append(len(tokens))
            self.term_frequencies.append(tf)
            
            # Update vocabulary and document frequencies
            unique_terms = set(tokens)
            self.vocabulary.update(unique_terms)
            for term in unique_terms:
                self.document_frequencies[term] += 1
        
        # Calculate average document length
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        self.total_docs = len(self.documents)
        
        logger.info(f"BM25 index built: {len(self.vocabulary)} unique terms, avg doc length: {self.avg_doc_length:.1f}")

    def _calculate_bm25_score(self, query_terms: List[str], doc_idx: int) -> float:
        """Calculate BM25 score for a document given query terms"""
        score = 0.0
        doc_length = self.doc_lengths[doc_idx]
        tf_dict = self.term_frequencies[doc_idx]
        
        for term in query_terms:
            if term not in tf_dict:
                continue
                
            # Term frequency in document
            tf = tf_dict[term]
            
            # Document frequency (number of documents containing the term)
            df = self.document_frequencies[term]
            
            # Inverse document frequency with smoothing to avoid negative values
            # Use max to ensure IDF is at least a small positive value
            idf = max(0.1, math.log((self.total_docs - df + 0.5) / (df + 0.5)))
            
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            
            term_score = idf * (numerator / denominator)
            score += term_score
        
        return score

    def search(
        self, query: str, max_results: int = 10, **kwargs
    ) -> List[Dict[str, Any]]:
        """Search using BM25 ranking algorithm"""
        start_time = _log_search_start(self.name, query)

        try:
            # Tokenize query
            query_lower = query.lower()
            query_terms = re.findall(r"\w+", query_lower)
            
            if not query_terms:
                _log_search_results(self.name, query, [], start_time)
                return []

            results = []

            # Calculate BM25 score for each document
            for doc_idx, document in enumerate(self.documents):
                score = self._calculate_bm25_score(query_terms, doc_idx)
                
                if score > 0:
                    results.append(
                        {
                            "document": document,
                            "score": round(score, 4),
                            "content": (
                                document.text[:500] + "..."
                                if len(document.text) > 500
                                else document.text
                            ),
                            "metadata": (
                                document.metadata
                                if hasattr(document, "metadata")
                                else {}
                            ),
                        }
                    )

            # Sort by BM25 score (descending) and return top results
            results.sort(key=lambda x: x["score"], reverse=True)
            final_results = results[:max_results]

            _log_search_results(self.name, query, final_results, start_time)
            return final_results

        except Exception as e:
            _log_search_results(self.name, query, [], start_time, error=str(e))
            raise


class VectorSearchTool(BaseRAGTool):
    """Vector-based semantic search tool using BGE embeddings"""

    def __init__(
        self,
        documents: List[Document],
        db_path: Optional[str] = None,
        api_key: Optional[str] = None,
        data_loader: Optional[RepositoryDataLoader] = None,
    ) -> None:
        super().__init__(
            name="vector_search",
            description="Semantic search through repository content using BGE vector embeddings stored in DuckDB. Good for finding conceptually similar content and answering questions about functionality.",
        )
        self.documents = documents
        # Use data loader's database path if no specific db_path is provided
        if db_path is None and data_loader is not None:
            self.db_path = data_loader.db_path
        else:
            self.db_path = db_path or "./vectors.duckdb"
        self.data_loader = data_loader

        # Initialize BGE embedding model (using small model with correct dimensions)
        self.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5",  # 384 dimensions
            trust_remote_code=True,
            embed_batch_size=64,
        )

        # Initialize Mistral LLM for query engine (keep for text generation)
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if self.api_key:
            self.llm = MistralAI(
                model="mistral-medium-latest", api_key=self.api_key, temperature=0.1
            )
        else:
            # Fallback to a basic LLM if no Mistral API key
            logger.warning("No MISTRAL_API_KEY found, using basic LLM")
            self.llm = None

        # Set global settings
        Settings.embed_model = self.embed_model
        if self.llm:
            Settings.llm = self.llm

        self._build_vector_index()

    def _build_vector_index(self) -> None:
        """Build vector index using DuckDB and Mistral embeddings"""
        import time

        start_time = time.time()

        try:
            logger.info(
                f"Starting vector index build with {len(self.documents)} documents"
            )

            # Step 1: Initialize vector store
            vector_store_start = time.time()
            if self.data_loader and self.data_loader.use_vectors:
                logger.info("Checking data loader's vector store...")
                vector_store = self.data_loader.get_vector_store()
                if vector_store:
                    logger.info("Using data loader's vector store")
                else:
                    logger.info(
                        "Data loader vector store not available, creating fallback"
                    )
                    vector_store = DuckDBVectorStore(
                        database_name=self.db_path,
                        table_name="bge_vectors",
                        persist_dir='./',
                        embed_dim=384,  # BGE-small-en-v1.5 dimension
                    )
            else:
                logger.info(f"Creating new DuckDB vector store at {self.db_path}")
                vector_store = DuckDBVectorStore(
                    database_name=self.db_path,
                    table_name="bge_vectors",
                    persist_dir='./',
                    embed_dim=384,  # BGE-small-en-v1.5 dimension
                )

            vector_store_time = time.time() - vector_store_start
            logger.info(
                f"Vector store initialization took {vector_store_time:.2f} seconds"
            )

            # Step 2: Create storage context
            storage_start = time.time()
            logger.info("Creating storage context...")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            storage_time = time.time() - storage_start
            logger.info(f"Storage context creation took {storage_time:.2f} seconds")

            # Step 3: Build or load vector index
            index_start = time.time()

            # Check if vector table actually exists before trying to load
            try:
                # Try to check if the vector table exists by querying it
                logger.info("Checking if vector table exists...")
                # Use the vector store's connection to check table existence
                if hasattr(vector_store, "_client") and vector_store._client:
                    vector_store._client.execute(
                        f"SELECT COUNT(*) FROM {vector_store.table_name} LIMIT 1"
                    ).fetchone()
                elif hasattr(vector_store, "client") and vector_store.client:
                    vector_store.client.execute(
                        f"SELECT COUNT(*) FROM {vector_store.table_name} LIMIT 1"
                    ).fetchone()
                else:
                    # Try to access the connection through duckdb directly
                    import duckdb

                    conn = duckdb.connect(vector_store.database_name)
                    conn.execute(
                        f"SELECT COUNT(*) FROM {vector_store.table_name} LIMIT 1"
                    ).fetchone()
                    conn.close()

                logger.info(
                    "Vector table exists, attempting to load existing vector index..."
                )

                self.index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store, embed_model=self.embed_model
                )
                logger.info("Successfully loaded existing vector index!")

            except Exception as e:
                logger.info(
                    f"Vector table doesn't exist or is empty ({e}), creating new vector index..."
                )
                logger.info(
                    f"Building new vector index with {len(self.documents)} documents..."
                )
                logger.info(
                    "This may take several minutes as embeddings are generated..."
                )

                # Log document types for context
                doc_types = {}
                for doc in self.documents:
                    doc_type = (
                        doc.metadata.get("type", "unknown")
                        if hasattr(doc, "metadata")
                        else "unknown"
                    )
                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                logger.info(f"Document types: {doc_types}")

                # Create new index from documents
                self.index = VectorStoreIndex.from_documents(
                    self.documents,
                    storage_context=storage_context,
                    embed_model=self.embed_model,
                    show_progress=True,
                )
                logger.info("Successfully created new vector index!")

            index_time = time.time() - index_start
            logger.info(
                f"Vector index initialization took {index_time:.2f} seconds ({index_time/60:.1f} minutes)"
            )

            # Step 5: Create retriever
            retriever_start = time.time()
            logger.info("Creating vector retriever...")
            self.retriever = VectorIndexRetriever(index=self.index, similarity_top_k=10)
            retriever_time = time.time() - retriever_start
            logger.info(f"Retriever creation took {retriever_time:.2f} seconds")

            # Step 6: Create query engine
            query_engine_start = time.time()
            logger.info("Creating query engine...")
            self.query_engine = RetrieverQueryEngine(
                retriever=self.retriever,
                node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
            )
            query_engine_time = time.time() - query_engine_start
            logger.info(f"Query engine creation took {query_engine_time:.2f} seconds")

            total_time = time.time() - start_time
            logger.info(
                f"✅ Vector index build completed in {total_time:.2f} seconds ({total_time/60:.1f} minutes)"
            )
            logger.info(
                f"Vector index ready with {len(self.documents)} documents using Mistral embeddings"
            )

        except Exception as e:
            total_time = time.time() - start_time
            logger.error(
                f"❌ Error building vector index after {total_time:.2f} seconds: {e}"
            )
            raise

    def search(
        self, query: str, max_results: int = 10, **kwargs
    ) -> List[Dict[str, Any]]:
        """Search using vector similarity with Mistral embeddings"""
        start_time = _log_search_start(self.name, query)

        try:
            # Retrieve relevant nodes
            nodes = self.retriever.retrieve(query)

            results = []
            for node in nodes[:max_results]:
                if isinstance(node, NodeWithScore):
                    results.append(
                        {
                            "content": (
                                node.node.text[:500] + "..."
                                if len(node.node.text) > 500
                                else node.node.text
                            ),
                            "score": float(node.score) if node.score else 0.0,
                            "metadata": (
                                node.node.metadata
                                if hasattr(node.node, "metadata")
                                else {}
                            ),
                        }
                    )

            _log_search_results(self.name, query, results, start_time)
            return results

        except Exception as e:
            _log_search_results(self.name, query, [], start_time, error=str(e))
            logger.error(f"Error in vector search: {e}")
            return []


class TextToSQLTool(BaseRAGTool):
    """Text-to-SQL tool for querying the database using Mistral LLM"""

    def __init__(
        self, data_loader: RepositoryDataLoader, api_key: Optional[str] = None
    ) -> None:
        super().__init__(
            name="text_to_sql",
            description="Convert natural language questions to SQL queries and execute them against the GitHub data database using Mistral AI. Good for statistical queries and structured data analysis.",
        )
        self.data_loader = data_loader

        # Initialize Mistral LLM
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY environment variable is required")

        self.llm = MistralAI(
            model="mistral-medium-latest", api_key=self.api_key, temperature=0
        )

        self.schema = data_loader.get_database_schema()
        self._build_schema_context()

    def _build_schema_context(self) -> None:
        """Build context about database schema dynamically"""
        schema_text = "Database Schema (DuckDB):\n\n"

        # Generate schema information dynamically from the database
        for table_name, columns in self.schema.items():
            schema_text += f"Table: {table_name}\n"

            # Get detailed column information from database
            try:
                if self.data_loader.conn:
                    # Get column details with types
                    describe_query = f"DESCRIBE {table_name}"
                    describe_result = self.data_loader.conn.execute(
                        describe_query
                    ).fetchall()

                    schema_text += "Columns:\n"
                    for col_info in describe_result:
                        col_name = col_info[0]
                        col_type = col_info[1]
                        schema_text += f"  - {col_name}: {col_type}\n"

                    # Get sample data to understand the table structure
                    sample_query = f"SELECT * FROM {table_name} LIMIT 1"
                    sample_result = self.data_loader.conn.execute(
                        sample_query
                    ).fetchall()

                    if sample_result:
                        schema_text += f"Sample row: {sample_result[0]}\n"

                else:
                    # Fallback to basic column list if no connection
                    schema_text += f"Columns: {', '.join(columns)}\n"

            except Exception as e:
                # Fallback to basic schema if query fails
                logger.warning(f"Could not get detailed schema for {table_name}: {e}")
                schema_text += f"Columns: {', '.join(columns)}\n"

            schema_text += "\n"

        # Add DuckDB-specific syntax guidance
        schema_text += """
DuckDB SQL Syntax Guidelines:
- Use DuckDB/ANSI SQL syntax only
- _sdc_repository: Contains repository names in format "mistralai/repo-name"
- JSON columns: Use JSON_EXTRACT_STRING(column, '$.field') for text values
- JSON columns: Use JSON_EXTRACT(column, '$.field') for any type
- For nested JSON: JSON_EXTRACT_STRING(user, '$.login') to get username
- Timestamps: Use standard SQL date functions (DATE, TIMESTAMP functions)
- String operations: Use LIKE, ILIKE (case-insensitive), REGEXP for pattern matching
- Always use LIMIT to avoid returning too many rows (max 100)
- Use DISTINCT when needed to avoid duplicates
- For arrays in JSON: Use JSON_EXTRACT to get array elements
- Date filtering: Use CURRENT_DATE, INTERVAL syntax
- IMPORTANT: events.created_at is VARCHAR, not TIMESTAMP - use string comparisons or cast
- Format the results nicely in markdown, instead of json when printed in a UI

Tested DuckDB JSON patterns:
- Get user login: JSON_EXTRACT_STRING(user, '$.login')
- Get user ID: JSON_EXTRACT(user, '$.id')
- Check if JSON field exists: JSON_EXTRACT(column, '$.field') IS NOT NULL
- Extract multiple JSON fields: JSON_EXTRACT_STRING(user, '$.login') as author, JSON_EXTRACT_STRING(user, '$.avatar_url') as avatar
- Handle JSON arrays: JSON_EXTRACT_STRING(labels, '$') for full array as string

Verified example queries:
- Issues: SELECT title, body, state, JSON_EXTRACT_STRING(user, '$.login') as author FROM issues WHERE _sdc_repository = 'mistralai/client-python' LIMIT 10
- Pull requests: SELECT title, state, JSON_EXTRACT_STRING(user, '$.login') as author FROM pull_requests WHERE _sdc_repository = 'mistralai/client-python' AND state = 'open'
- Comments: SELECT body, JSON_EXTRACT_STRING(user, '$.login') as author, created_at FROM comments WHERE _sdc_repository = 'mistralai/client-python' ORDER BY created_at DESC LIMIT 20
- Recent activity: SELECT type, JSON_EXTRACT_STRING(actor, '$.login') as actor, created_at FROM events WHERE _sdc_repository = 'mistralai/client-python' ORDER BY created_at DESC LIMIT 10
- Count by repository: SELECT _sdc_repository, COUNT(*) as count FROM issues GROUP BY _sdc_repository ORDER BY count DESC
- Date filtering: SELECT title FROM issues WHERE _sdc_repository = 'mistralai/client-python' AND created_at >= CURRENT_DATE - INTERVAL 30 DAY
- Text search: SELECT title, body FROM issues WHERE _sdc_repository = 'mistralai/client-python' AND (title ILIKE '%api%' OR body ILIKE '%api%')
- Events (note VARCHAR created_at): SELECT type, COUNT(*) FROM events WHERE created_at >= '2025-05-01' GROUP BY type ORDER BY COUNT(*) DESC
"""

        self.schema_context = schema_text

    def _generate_sql_query(self, question: str) -> str:
        """Generate SQL query from natural language question using Mistral"""
        prompt = f"""You are an expert SQL developer. Convert the natural language question to a SQL query.

{self.schema_context}

Question: {question}

Generate a SQL query that answers this question. Return only the SQL query without any explanation or markdown formatting.

SQL Query:"""

        try:
            response = self.llm.complete(prompt)
            sql_query = response.text.strip()

            # Clean up the SQL query
            if sql_query.startswith("```sql"):
                sql_query = sql_query[6:]
            if sql_query.endswith("```"):
                sql_query = sql_query[:-3]
            if sql_query.startswith("```"):
                sql_query = sql_query[3:]

            return sql_query.strip()

        except Exception as e:
            logger.error(f"Error generating SQL query: {e}")
            raise

    def search(self, query: str, max_retries: int = 3, **kwargs) -> List[Dict[str, Any]]:
        """Convert question to SQL and execute with retry logic"""
        start_time = _log_search_start(self.name, query)
        
        original_query = query
        error_history = []
        
        for attempt in range(max_retries):
            try:
                # Generate SQL query (potentially improved based on previous errors)
                sql_query = self._generate_sql_query(query)
                logger.info(f"Generated SQL (attempt {attempt + 1}): {sql_query}")

                # Execute query
                df = self.data_loader.execute_sql_query(sql_query)

                # Format results as JSON
                json_results = self._format_results_as_json(df, sql_query)

                final_results = [
                    {
                        "sql_query": sql_query,
                        "results": json_results,
                        "row_count": len(df),
                        "limited_to": min(50, len(df)),
                        "attempts": attempt + 1,
                        "error_history": error_history,
                    }
                ]

                _log_search_results(self.name, original_query, final_results, start_time)
                return final_results

            except Exception as e:
                error_msg = str(e)
                error_history.append(f"Attempt {attempt + 1}: {error_msg}")
                logger.warning(f"SQL execution failed (attempt {attempt + 1}/{max_retries}): {error_msg}")
                
                # If this is not the last attempt, try to improve the prompt
                if attempt < max_retries - 1:
                    improved_query = self._improve_query_from_error(
                        original_query, error_msg, sql_query if 'sql_query' in locals() else None
                    )
                    if improved_query != query:
                        logger.info(f"Improved query for retry: {improved_query}")
                        query = improved_query
                    
                    # Add a small delay before retry
                    time.sleep(0.5)
                else:
                    # Final attempt failed
                    error_results = [
                        {
                            "error": error_msg,
                            "sql_query": sql_query if "sql_query" in locals() else None,
                            "attempts": max_retries,
                            "error_history": error_history,
                            "original_query": original_query,
                        }
                    ]
                    _log_search_results(
                        self.name, original_query, error_results, start_time, error=error_msg
                    )
                    logger.error(f"All {max_retries} attempts failed for text-to-SQL query")
                    return error_results

    def _improve_query_from_error(self, original_query: str, error_message: str, failed_sql: Optional[str] = None) -> str:
        """
        Improve the natural language query based on the SQL error message.
        
        Args:
            original_query: The original natural language query
            error_message: The SQL error message
            failed_sql: The SQL query that failed (if available)
            
        Returns:
            Improved natural language query with additional context
        """
        # Enhanced error patterns and their specific fixes
        error_improvements = {
            "no such table": "Please use only existing tables from the schema. Available tables: issues, pull_requests, comments, assignees, events, commits, reviews, stargazers. ",
            "no such column": "Please use only existing columns from the schema. Check the table structure carefully. ",
            "syntax error": "Please use correct DuckDB SQL syntax. ",
            "json_extract": "Please use JSON_EXTRACT_STRING for text fields and JSON_EXTRACT for other types. ",
            "malformed json": "The column contains non-JSON data. For assignees table, use direct column access (e.g., assignees.login) instead of JSON_EXTRACT. ",
            "referenced table": "Table alias scope issue. Avoid complex subqueries with aliases that reference tables outside their scope. Use simpler JOIN syntax or separate queries. ",
            "binder error": "Column or table reference issue. Check table aliases and column names. For assignees table, use direct column access instead of JSON functions. ",
            "ambiguous column": "Please use table aliases to avoid ambiguous column references. ",
            "division by zero": "Please add checks to avoid division by zero. ",
            "invalid date": "Please use proper date format or string comparison for date fields. ",
            "group by": "Please include all non-aggregate columns in GROUP BY clause. ",
            "invalid input error": "Data type mismatch. For assignees table, the 'user' column contains string 'duckdb', not JSON. Use assignees.login directly. ",
        }
        
        # Build improved query with error context
        improved_query = original_query
        
        # Add specific guidance based on error type
        error_lower = error_message.lower()
        guidance_added = False
        
        for error_pattern, guidance in error_improvements.items():
            if error_pattern in error_lower:
                improved_query = f"{guidance}{improved_query}"
                guidance_added = True
                break
        
        # Special handling for specific error patterns we discovered
        if "malformed json" in error_lower and "assignees" in (failed_sql or "").lower():
            improved_query = f"IMPORTANT: The assignees table 'user' column contains string data, not JSON. Use assignees.login, assignees.id, etc. directly instead of JSON_EXTRACT functions. {improved_query}"
        elif "referenced table" in error_lower and "not found" in error_lower:
            improved_query = f"Avoid complex subqueries with table aliases. Use simple JOINs or separate the query into simpler parts. {improved_query}"
        
        # Add failed SQL context if available
        if failed_sql:
            improved_query += f" The previous SQL query failed: {failed_sql}. Please generate a corrected version."
        
        # Add general error context
        improved_query += f" Previous error: {error_message}. Please fix this issue."
        
        return improved_query

    def _format_results_as_json(self, df: pd.DataFrame, sql_query: str) -> List[Dict[str, Any]]:
        """Format query results as JSON with proper data type handling"""
        if df.empty:
            return []

        # Limit to first 50 rows for display
        df_display = df.head(50)
        
        results = []
        for _, row in df_display.iterrows():
            row_dict = {}
            for col in df_display.columns:
                value = row[col]
                if pd.isna(value):
                    row_dict[col] = None
                elif isinstance(value, (int, float, bool)):
                    row_dict[col] = value
                else:
                    # Convert to string and handle special cases
                    str_value = str(value)
                    # Try to parse JSON strings back to objects
                    if str_value.startswith('{') or str_value.startswith('['):
                        try:
                            import json
                            row_dict[col] = json.loads(str_value)
                        except (json.JSONDecodeError, ValueError):
                            row_dict[col] = str_value
                    else:
                        row_dict[col] = str_value
            results.append(row_dict)
        
        return results


class RAGToolManager:
    """Manager for coordinating multiple RAG tools using Mistral APIs"""

    def __init__(
        self, data_loader: RepositoryDataLoader, api_key: Optional[str] = None
    ) -> None:
        self.data_loader = data_loader
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY environment variable is required")

        self.llm = MistralAI(
            model="mistral-medium-latest", api_key=self.api_key, temperature=0.1
        )

        self.tools: Dict[str, BaseRAGTool] = {}
        self._initialize_tools()

    def _initialize_tools(self) -> None:
        """Initialize all RAG tools"""
        import time

        total_start = time.time()

        # Step 1: Load documents
        doc_start = time.time()
        logger.info("Loading documents from all repositories...")
        documents = self.data_loader.load_all_repositories()
        doc_time = time.time() - doc_start
        logger.info(f"✅ Loaded {len(documents)} documents in {doc_time:.2f} seconds")

        # Step 2: Initialize text search tool (fast)
        text_start = time.time()
        logger.info("Initializing text search tool...")
        self.tools["text_search"] = TextSearchTool(documents)
        text_time = time.time() - text_start
        logger.info(f"✅ Text search tool initialized in {text_time:.2f} seconds")

        # Step 3: Initialize vector search tool (potentially slow)
        vector_start = time.time()
        logger.info("Initializing vector search tool...")
        logger.info(
            "⚠️  This step may take several minutes if embeddings need to be created..."
        )

        # Only initialize vector search if vectors are enabled
        if self.data_loader.use_vectors:
            logger.info("Vector search enabled - proceeding with initialization")
            self.tools["vector_search"] = VectorSearchTool(
                documents, api_key=self.api_key, data_loader=self.data_loader
            )
            vector_time = time.time() - vector_start
            logger.info(
                f"✅ Vector search tool initialized in {vector_time:.2f} seconds ({vector_time/60:.1f} minutes)"
            )
        else:
            logger.info("Vector search disabled - skipping vector tool initialization")
            vector_time = 0

        # Step 4: Initialize text-to-SQL tool (fast)
        sql_start = time.time()
        logger.info("Initializing text-to-SQL tool...")
        self.tools["text_to_sql"] = TextToSQLTool(
            self.data_loader, api_key=self.api_key
        )
        sql_time = time.time() - sql_start
        logger.info(f"✅ Text-to-SQL tool initialized in {sql_time:.2f} seconds")

        total_time = time.time() - total_start
        logger.info(
            f"🎉 All tools initialized successfully in {total_time:.2f} seconds ({total_time/60:.1f} minutes)"
        )
        logger.info(
            f"Time breakdown: Documents={doc_time:.1f}s, Text={text_time:.1f}s, Vector={vector_time:.1f}s, SQL={sql_time:.1f}s"
        )

    def search(self, query: str, tool_name: str, **kwargs) -> List[Dict[str, Any]]:
        """Search using specified tool"""
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        return self.tools[tool_name].search(query, **kwargs)

    def search_all(
        self, query: str, max_results_per_tool: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Search using all tools and return combined results"""
        results = {}

        for tool_name, tool in self.tools.items():
            try:
                tool_results = tool.search(query, max_results=max_results_per_tool)
                results[tool_name] = tool_results
            except Exception as e:
                logger.error(f"Error in {tool_name}: {e}")
                results[tool_name] = [{"error": str(e)}]

        return results

    def get_available_tools(self) -> List[Dict[str, str]]:
        """Get list of available tools with descriptions"""
        return [
            {"name": tool.name, "description": tool.description}
            for tool in self.tools.values()
        ]

    def generate_response(
        self, query: str, search_results: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        """Generate a comprehensive response using Mistral LLM"""
        # Format search results for the prompt
        context = "Search Results:\n\n"

        for tool_name, results in search_results.items():
            context += f"=== {tool_name.upper()} RESULTS ===\n"
            for i, result in enumerate(results[:5]):  # Limit to top 5 per tool
                if "error" in result:
                    context += f"Error: {result['error']}\n"
                elif tool_name == "text_to_sql":
                    context += f"SQL Query: {result.get('sql_query', 'N/A')}\n"
                    if "results" in result and result["results"]:
                        # Show first few JSON results
                        import json
                        json_str = json.dumps(result["results"][:5], indent=2)
                        context += f"Results (JSON): {json_str}\n"
                else:
                    context += (
                        f"Result {i+1}: {result.get('content', str(result))[:1000]}...\n"
                    )
            context += "\n"

        prompt = f"""You are an expert assistant helping users understand Mistral AI repositories. 
Based on the search results below, provide a comprehensive and helpful answer to the user's question.

User Question: {query}

{context}

Instructions:
- Synthesize information from all relevant search results, prioritising sql results and vector search.
- Ignore irrelevant information
- Provide specific examples and code snippets when available
- If SQL results are present, interpret and explain them clearly.
- Be concise but thorough
- If the search results don't contain enough information, say so clearly
- Format the SQL results JSON as markdown table in the response

Answer:"""

        try:
            logger.info(prompt)
            response = self.llm.complete(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I encountered an error while generating a response: {str(e)}"
