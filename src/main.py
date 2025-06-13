#!/usr/bin/env python3
"""
Main entry point for the Scalping Bot.
"""
import os
import sys
import logging
from dotenv import load_dotenv
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'logs' / 'bot.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main function to run the bot."""
    try:
        # Load environment variables
        load_dotenv()
        
        logger.info("Starting Scalping Bot...")
        
        # TODO: Initialize and run the bot
        # This will be implemented in future commits
        
        logger.info("Bot started successfully")
        
    except Exception as e:
        logger.error(f"Error starting bot: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 