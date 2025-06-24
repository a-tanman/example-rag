"""
Main application entry point for the Mistral RAG Chatbot
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.chatbot import main  # noqa: E402

if __name__ == "__main__":
    main()
