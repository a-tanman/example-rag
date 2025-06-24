"""
Tests for the chatbot module
"""

from unittest.mock import Mock, patch

import gradio as gr
import pytest

from src.chatbot import MistralRAGChatbot


class TestMistralRAGChatbot:
    """Test cases for MistralRAGChatbot"""

    @pytest.fixture
    def chatbot(self):
        """Create chatbot instance for testing"""
        return MistralRAGChatbot(data_dir="test_data", db_path="test.db")

    def test_init(self, chatbot):
        """Test chatbot initialization"""
        assert chatbot.data_dir == "test_data"
        assert chatbot.db_path == "test.db"
        assert chatbot.data_loader is None
        assert chatbot.rag_manager is None
        assert chatbot.initialized is False

    @patch("src.chatbot.RepositoryDataLoader")
    @patch("src.chatbot.RAGToolManager")
    @patch.dict("os.environ", {"MISTRAL_API_KEY": "test_key"})
    def test_initialize_success(self, mock_rag_manager, mock_data_loader, chatbot):
        """Test successful initialization"""
        # Mock data loader
        mock_loader_instance = Mock()
        mock_loader_instance.get_available_repositories.return_value = [
            "repo1",
            "repo2",
        ]
        mock_data_loader.return_value = mock_loader_instance

        # Mock RAG manager
        mock_rag_instance = Mock()
        mock_rag_manager.return_value = mock_rag_instance

        result = chatbot.initialize()

        assert "Successfully initialized" in result
        assert "repo1, repo2" in result
        assert chatbot.initialized is True
        assert chatbot.data_loader == mock_loader_instance
        assert chatbot.rag_manager == mock_rag_instance

    @patch.dict("os.environ", {}, clear=True)
    def test_initialize_no_api_key(self, chatbot):
        """Test initialization without API key"""
        result = chatbot.initialize()

        assert "MISTRAL_API_KEY environment variable is required" in result
        assert chatbot.initialized is False

    @patch("src.chatbot.RepositoryDataLoader")
    @patch("src.chatbot.RAGToolManager")
    @patch.dict("os.environ", {"MISTRAL_API_KEY": "test_key"})
    def test_chat_functionality(self, mock_rag_manager, mock_data_loader, chatbot):
        """Test chat functionality with different tools"""
        # Setup mocks
        mock_loader_instance = Mock()
        mock_loader_instance.get_available_repositories.return_value = ["repo1"]
        mock_data_loader.return_value = mock_loader_instance

        mock_rag_instance = Mock()
        mock_rag_instance.search_all.return_value = {
            "text_search": [{"content": "text result"}],
            "vector_search": [{"content": "vector result"}],
        }
        mock_rag_instance.generate_response.return_value = "Generated response"
        mock_rag_manager.return_value = mock_rag_instance

        # Initialize chatbot
        chatbot.initialize()

        # Test chat with all tools
        history, msg = chatbot.chat("test question", [], "All Tools")

        assert len(history) == 1
        assert history[0][0] == "test question"
        assert "Generated response" in history[0][1]
        assert "Tools used:" in history[0][1]
        assert msg == ""

        # Test empty message
        history, msg = chatbot.chat("", [], "All Tools")
        assert history == []
        assert msg == ""

    @patch("src.chatbot.RepositoryDataLoader")
    @patch("src.chatbot.RAGToolManager")
    @patch.dict("os.environ", {"MISTRAL_API_KEY": "test_key"})
    def test_system_info(self, mock_rag_manager, mock_data_loader, chatbot):
        """Test system info functionality"""
        # Setup mocks
        mock_loader_instance = Mock()
        mock_loader_instance.get_available_repositories.return_value = [
            "repo1",
            "repo2",
        ]
        mock_loader_instance.get_database_schema.return_value = {
            "issues": ["title", "body"],
            "pull_requests": ["title", "body"],
        }
        mock_data_loader.return_value = mock_loader_instance

        mock_rag_instance = Mock()
        mock_rag_instance.get_available_tools.return_value = [
            {"name": "text_search", "description": "Text search tool"},
            {"name": "vector_search", "description": "Vector search tool"},
        ]
        mock_rag_manager.return_value = mock_rag_instance

        # Test before initialization
        result = chatbot.get_system_info()
        assert result == "System not initialized"

        # Initialize chatbot
        chatbot.initialize()

        # Test after initialization
        result = chatbot.get_system_info()

        assert "**Available Repositories:** 2" in result
        assert "repo1, repo2" in result
        assert "**Database Tables:** 2" in result
        assert "issues, pull_requests" in result
        assert "**Available Tools:** 2" in result
        assert "text_search" in result
        assert "vector_search" in result

    def test_interface_creation(self, chatbot):
        """Test Gradio interface creation and launch"""
        interface = chatbot.create_interface()
        assert isinstance(interface, gr.Blocks)

        # Test launch method
        with patch.object(chatbot, "create_interface") as mock_create:
            mock_interface = Mock()
            mock_interface.launch.return_value = "launched"
            mock_create.return_value = mock_interface

            result = chatbot.launch(server_port=7860)
            assert result is None
            mock_interface.launch.assert_called_once_with(server_port=7860)
