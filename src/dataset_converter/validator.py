import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

class ChatValidator:
    """Validates the chat format dataset for Phi-3 Mini fine-tuning."""
    
    @staticmethod
    def validate(sample: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validates a single converted chat sample.
        Returns a tuple of (is_valid, reason).
        """
        if "messages" not in sample:
            return False, "Missing 'messages' key"
            
        messages = sample["messages"]
        if not isinstance(messages, list) or len(messages) != 2:
            return False, f"Expected exactly two messages, got {len(messages) if isinstance(messages, list) else 'non-list'}"
            
        user_msg = messages[0]
        assistant_msg = messages[1]
        
        if user_msg.get("role") != "user":
            return False, f"First message role must be 'user', got '{user_msg.get('role')}'"
            
        if assistant_msg.get("role") != "assistant":
            return False, f"Second message role must be 'assistant', got '{assistant_msg.get('role')}'"
            
        if not str(user_msg.get("content", "")).strip():
            return False, "Empty instruction in user message"
            
        if not str(assistant_msg.get("content", "")).strip():
            return False, "Empty answer in assistant message"
            
        if "metadata" not in sample:
            return False, "Missing 'metadata' key"
            
        metadata = sample["metadata"]
        required_meta = ["source", "chunk_id", "category", "difficulty", "quality_score"]
        for key in required_meta:
            if key not in metadata:
                return False, f"Missing '{key}' in metadata"
                
        return True, "Valid"
