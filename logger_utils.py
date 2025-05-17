import logging
from datetime import datetime
import os
import json
from typing import Dict, Any

class RAGLogger:
    def __init__(self):
        # Create logs directory if it doesn't exist
        self.logs_dir = "logs"
        self.log_file = os.path.join(self.logs_dir, f"rag_log_{datetime.now().strftime('%Y%m%d')}.log")
        self.query_history_file = os.path.join(self.logs_dir, "query_history.json")
        
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def log_query(self, question: str, answer: str, metadata: Dict[Any, Any] = None):
        """Log a query and its answer"""
        timestamp = datetime.now().isoformat()
        query_data = {
            "timestamp": timestamp,
            "question": question,
            "answer": answer,
            "metadata": metadata or {}
        }
        
        # Log to regular log file
        self.logger.info(f"Query: {question}")
        self.logger.info(f"Answer: {answer[:100]}...")  # Log first 100 chars of answer
        
        # Append to query history JSON
        try:
            if os.path.exists(self.query_history_file):
                with open(self.query_history_file, 'r') as f:
                    history = json.load(f)
            else:
                history = []
                
            history.append(query_data)
            
            with open(self.query_history_file, 'w') as f:
                json.dump(history, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving query history: {str(e)}")
    
    def log_error(self, error_msg: str, context: Dict[Any, Any] = None):
        """Log an error with optional context"""
        self.logger.error(f"Error: {error_msg}")
        if context:
            self.logger.error(f"Context: {context}")
