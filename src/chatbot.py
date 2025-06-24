"""
Gradio chatbot interface for the RAG system
"""

import argparse
import logging
import os
from typing import List, Optional, Tuple

import gradio as gr
from dotenv import load_dotenv

from .data_loader import RepositoryDataLoader
from .rag_tools import RAGToolManager

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class MistralRAGChatbot:
    """Gradio-based chatbot for querying Mistral repositories"""

    def __init__(
        self,
        data_dir: str = "data/mistral-repos",
        db_path: str = "db.duckdb",
        vector_db_path: str = None,  # Use main database by default
        force_rebuild_vectors: bool = False,
    ):
        self.data_dir = data_dir
        self.db_path = db_path
        self.vector_db_path = vector_db_path
        self.force_rebuild_vectors = force_rebuild_vectors
        self.data_loader: Optional[RepositoryDataLoader] = None
        self.rag_manager: Optional[RAGToolManager] = None
        self.initialized = False

    def initialize(self) -> str:
        """Initialize the RAG system"""
        try:
            # Check for API key
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                return "❌ Error: MISTRAL_API_KEY environment variable is required"

            # Initialize data loader with vector parameters
            # Use main database for vectors if no separate vector db specified
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
                return f"❌ Error: No repositories found in {self.data_dir}"

            # Initialize RAG tools
            self.rag_manager = RAGToolManager(self.data_loader, api_key)

            self.initialized = True

            vector_status = "enabled" if self.data_loader.use_vectors else "disabled"
            return f"✅ Successfully initialized RAG system with {len(repos)} repositories: {', '.join(repos)}. Vectors: {vector_status}"

        except Exception as e:
            logger.error(f"Initialization error: {e}")
            return f"❌ Error initializing system: {str(e)}"

    def chat(
        self, message: str, history: List[Tuple[str, str]], tool_selection: str
    ) -> Tuple[List[Tuple[str, str]], str]:
        """Process chat message and return response"""
        if not self.initialized:
            init_result = self.initialize()
            if "Error" in init_result:
                history.append((message, init_result))
                return history, ""

        if not message.strip():
            return history, ""

        try:
            # Determine which tools to use
            if tool_selection == "All Tools":
                if self.rag_manager is None:
                    raise RuntimeError("RAG manager not initialized")
                search_results = self.rag_manager.search_all(
                    message, max_results_per_tool=3
                )
                response = self.rag_manager.generate_response(message, search_results)

                # Add tool results summary
                tools_used = []
                for tool_name, results in search_results.items():
                    if results and not any("error" in r for r in results):
                        tools_used.append(tool_name)

                if tools_used:
                    response += f"\n\n*Tools used: {', '.join(tools_used)}*"

            else:
                # Use specific tool
                tool_map = {
                    "Text Search": "text_search",
                    "Vector Search": "vector_search",
                    "Text to SQL": "text_to_sql",
                }

                tool_name = tool_map.get(tool_selection)
                if tool_name is None:
                    response = f"❌ Unknown tool: {tool_selection}"
                else:
                    if self.rag_manager is None:
                        raise RuntimeError("RAG manager not initialized")
                    results = self.rag_manager.search(message, tool_name, max_results=5)

                    if tool_name == "text_to_sql":
                        # Format SQL results using JSON format
                        if results and not any("error" in r for r in results):
                            result = results[0]
                            import json

                            # Format the JSON results nicely
                            json_results = result.get("results", [])
                            if json_results:
                                # Create a formatted JSON response
                                response = f"**SQL Query:** `{result.get('sql_query', 'N/A')}`\n\n"
                                response += f"**Results:** {result.get('row_count', 0)} rows found"
                                if result.get('limited_to', 0) < result.get('row_count', 0):
                                    response += f" (showing first {result.get('limited_to', 0)})"
                                response += "\n\n"
                                
                                # Format as JSON code block
                                response += "```json\n"
                                response += json.dumps(json_results, indent=2, ensure_ascii=False)
                                response += "\n```"
                            else:
                                response = f"**SQL Query:** `{result.get('sql_query', 'N/A')}`\n\nNo results returned."
                        else:
                            error_msg = (
                                results[0].get("error", "Unknown error")
                                if results
                                else "No results"
                            )
                            sql_query = results[0].get("sql_query", "N/A") if results else "N/A"
                            response = f"❌ **SQL Error:** {error_msg}\n\n**Query:** `{sql_query}`"
                    else:
                        # Format text/vector search results
                        if results:
                            response = f"Found {len(results)} results:\n\n"
                            for i, result in enumerate(results[:5]):  # Show top 5
                                content = result.get("content", str(result))
                                metadata = result.get("metadata", {})
                                repo = metadata.get("repository", "Unknown")
                                file_path = metadata.get("file_path", "")

                                response += f"**Result {i+1}** (Repository: {repo}"
                                if file_path:
                                    response += f", File: {file_path}"
                                response += f"):\n{content}\n\n"
                        else:
                            response = "No results found."

                    response += f"\n*Tool used: {tool_selection}*"

            history.append((message, response))

        except Exception as e:
            logger.error(f"Chat error: {e}")
            error_response = f"❌ Error processing your request: {str(e)}"
            history.append((message, error_response))

        return history, ""

    def get_system_info(self) -> str:
        """Get system information"""
        if not self.initialized:
            return "System not initialized"

        try:
            if self.data_loader is None or self.rag_manager is None:
                raise RuntimeError("System not properly initialized")
            repos = self.data_loader.get_available_repositories()
            schema = self.data_loader.get_database_schema()
            tools = self.rag_manager.get_available_tools()

            info = f"""
## System Information

**Available Repositories:** {len(repos)}
{', '.join(repos)}

**Database Tables:** {len(schema)}
{', '.join(schema.keys())}

**Available Tools:** {len(tools)}
"""
            for tool in tools:
                info += f"- **{tool['name']}**: {tool['description']}\n"

            return info

        except Exception as e:
            return f"Error getting system info: {str(e)}"

    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface"""
        with gr.Blocks(
            title="Mistral AI Repository RAG Chatbot", theme=gr.themes.Soft()
        ) as interface:

            gr.Markdown(
                """
            # 🤖 Mistral AI Repository RAG Chatbot
            
            Ask questions about Mistral AI repositories using three powerful search tools:
            - **Text Search**: Keyword-based search for exact matches
            - **Vector Search**: Semantic search using Mistral embeddings  
            - **Text to SQL**: Natural language queries converted to SQL
            
            **Note**: Requires `MISTRAL_API_KEY` environment variable to be set.
            """
            )

            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        height=500, label="Chat History", show_label=True
                    )

                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Ask a question about Mistral repositories...",
                            label="Your Question",
                            scale=4,
                        )
                        tool_selector = gr.Dropdown(
                            choices=[
                                "All Tools",
                                "Text Search",
                                "Vector Search",
                                "Text to SQL",
                            ],
                            value="All Tools",
                            label="Tool Selection",
                            scale=1,
                        )

                    with gr.Row():
                        submit_btn = gr.Button("Send", variant="primary")
                        clear_btn = gr.Button("Clear Chat")

                with gr.Column(scale=1):
                    gr.Markdown("### System Status")
                    status_display = gr.Markdown("Click 'Initialize System' to start")

                    init_btn = gr.Button("Initialize System", variant="secondary")

                    gr.Markdown("### System Information")
                    info_display = gr.Markdown("System not initialized")

                    refresh_info_btn = gr.Button("Refresh Info", variant="secondary")

            # Example queries
            gr.Markdown(
                """
            ### 💡 Example Queries

            Note that all the below queries can be run in 'All Tools' mode.
            
            **Text Search Examples:**
            - "Function calling"
            - "Authentication implementation"
            - "Error handling patterns"
            
            **Vector Search Examples:**  
            - "What are the main features of the mistral-finetune repository?"
            - "Performance optimization techniques for mistral-inference?"
            - "How to use the Python client library?"
            - "Best practices for model deployment"
            
            **Text to SQL Examples (Tested & Verified):**
            
            *Basic Statistics:*
            - "How many issues are there in total?"
            - "How many open issues are there?"
            - "How many closed pull requests are there?"
            - "Count the total number of comments"
            
            *Repository Analysis:*
            - "Which repository has the highest number of issues?"
            - "Show me all repositories and their issue counts"
            - "Which repository has the most pull requests?"
            - "Show repositories with more than 10 issues"
            - "Show the distribution of issue states across repositories"
            
            *User & Contributor Analysis:*
            - "Who are the top 5 most active issue creators?"
            - "Which users have the most comments?"
            - "Find the most active contributors across all repositories"
            - "Show me issue authors and their usernames"
            
            *Content-Based Searches:*
            - "Find issues that mention 'API' in the title"
            - "Show me issues related to authentication"
            - "Find pull requests with 'bug' in the title"
            - "Search for issues containing 'error' or 'exception'"
            
            *Time-Based Queries:*
            - "Show me the most recent 10 issues"
            - "Find issues created in 2024"
            - "What are the oldest issues in the database?"
            - "Show pull requests created in the last month"
            
            *Advanced Analytics:*
            - "Find the most commented issues"
            - "Calculate average issues per repository"
            - "Show repositories ordered by activity level"
            - "Find issues with labels"
            """
            )

            # Event handlers
            def initialize_system() -> Tuple[str, str]:
                result = self.initialize()
                info = (
                    self.get_system_info()
                    if self.initialized
                    else "System not initialized"
                )
                return result, info

            def refresh_system_info() -> str:
                return self.get_system_info()

            # Button events
            init_btn.click(fn=initialize_system, outputs=[status_display, info_display])

            refresh_info_btn.click(fn=refresh_system_info, outputs=[info_display])

            # Chat events
            submit_btn.click(
                fn=self.chat,
                inputs=[msg, chatbot, tool_selector],
                outputs=[chatbot, msg],
            )

            msg.submit(
                fn=self.chat,
                inputs=[msg, chatbot, tool_selector],
                outputs=[chatbot, msg],
            )

            clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, msg])

        return interface

    def launch(self, **kwargs) -> None:
        """Launch the Gradio interface"""
        interface = self.create_interface()
        interface.launch(**kwargs)


def main() -> None:
    """Main function to launch the chatbot"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Mistral RAG Chatbot")
    parser.add_argument(
        "--data-dir",
        default="data/mistral-repos",
        help="Directory containing repository data",
    )
    parser.add_argument(
        "--db-path", default="db.duckdb", help="Path to main DuckDB database"
    )
    parser.add_argument(
        "--vector-db-path",
        default=None,
        help="Path to vector DuckDB database (defaults to main database)",
    )
    parser.add_argument(
        "--force-rebuild-vectors",
        action="store_true",
        help="Force rebuild of vector embeddings",
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port to run the server on"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to run the server on")

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create and launch chatbot
    chatbot = MistralRAGChatbot(
        data_dir=args.data_dir,
        db_path=args.db_path,
        vector_db_path=args.vector_db_path,
        force_rebuild_vectors=args.force_rebuild_vectors,
    )

    logger.info(
        f"Starting chatbot with vectors {'enabled' if args.force_rebuild_vectors else 'auto-detected'}"
    )

    chatbot.launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == "__main__":
    main()
