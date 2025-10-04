#!/usr/bin/env python3
"""
Conversation Memory Manager for Multi-LLM Chatbot

This module provides intelligent conversation buffer management that adapts to
different LLM context sizes, maintains conversation history, and provides
automatic truncation with optional summarization.
"""

import json
import re
from system_prompts import get_master_system_prompt
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import sentencepiece as spm


@dataclass
class ConversationMessage:
    """Represents a single message in the conversation."""
    role: str  # "system", "user", "assistant"
    content: str
    timestamp: datetime
    token_count: int = 0
    is_pinned: bool = False  # Pinned messages are never removed
    is_summary: bool = False  # Indicates if this is a generated summary


@dataclass
class ModelConfig:
    """Configuration for a specific LLM model."""
    name: str
    max_tokens: int
    reserve_tokens: int = 1000  # Reserve for response generation
    summary_threshold: float = 0.7  # Start summarizing when buffer reaches 70%


class TokenCounter:
    """Handles token counting for different models."""
    
    def __init__(self):
        try:
            self.tokenizer = spm.SentencePieceProcessor()
            # fallback since we don’t have a trained tokenizer
            self.tokenizer = None
            print("Info: Using character-based token approximation (SentencePiece model not configured)")
        except Exception as e:
            print(f"Warning: Could not initialize SentencePiece tokenizer: {e}")
            self.tokenizer = None
    
    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        if not text:
            return 0
        if self.tokenizer:
            try:
                tokens = self.tokenizer.encode(text, out_type=int)
                return len(tokens)
            except Exception as e:
                print(f"SentencePiece tokenization error: {e}")
                pass
        return max(1, len(text) // 4)
    
    def count_message_tokens(self, message: Dict[str, str], model: Optional[str] = None) -> int:
        base_tokens = self.count_tokens(message.get('content', ''), model)
        overhead = 10
        return base_tokens + overhead


class ConversationSummarizer:
    """Generates compact summaries of conversation segments."""
    
    @staticmethod
    def create_summary(messages: List[ConversationMessage]) -> str:
        if not messages:
            return ""
        user_messages = [msg for msg in messages if msg.role == "user"]
        assistant_messages = [msg for msg in messages if msg.role == "assistant"]
        
        summary_parts = []
        
        if user_messages:
            user_topics = []
            for msg in user_messages:
                content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                user_topics.append(content)
            if len(user_topics) == 1:
                summary_parts.append(f"User asked: {user_topics[0]}")
            else:
                summary_parts.append(f"User discussed: {'; '.join(user_topics[:3])}")
                if len(user_topics) > 3:
                    summary_parts[-1] += f" and {len(user_topics) - 3} other topics"
        
        if assistant_messages:
            if len(assistant_messages) == 1:
                response_summary = assistant_messages[0].content[:150]
                if len(assistant_messages[0].content) > 150:
                    response_summary += "..."
                summary_parts.append(f"Assistant responded: {response_summary}")
            else:
                summary_parts.append(f"Assistant provided {len(assistant_messages)} detailed responses")
        
        return " | ".join(summary_parts)


class ConversationMemoryManager:
    """
    Main conversation memory manager that handles buffer management,
    token counting, and automatic truncation with summarization.
    """
    
    MODEL_CONFIGS = {
        'meta/llama-4-maverick-17b-128e-instruct': ModelConfig("Llama 4 Maverick", 1000000),
        'deepseek-ai/deepseek-r1': ModelConfig("DeepSeek R1", 128000),
        'qwen/qwen2.5-coder-32b-instruct': ModelConfig("Qwen 2.5 Coder", 32000),
        'qwen/qwen3-coder-480b-a35b-instruct': ModelConfig("Qwen3 Coder 480B", 256000),
        'deepseek-ai/deepseek-v3.1': ModelConfig("DeepSeek V3.1", 128000),
        'openai/gpt-oss-120b': ModelConfig("GPT OSS", 128000),
        'qwen/qwen3-235b-a22b:free': ModelConfig("Qwen3 235B", 131000),
        'google/gemma-3-27b-it:free': ModelConfig("Gemma 3", 96000),
        'x-ai/grok-4-fast:free': ModelConfig("Grok 4", 2000000),
    }
    
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.messages: List[ConversationMessage] = []
        self.token_counter = TokenCounter()
        self.summarizer = ConversationSummarizer()
        self.current_model: Optional[str] = None
        
        # Initialize with system prompt
        self._add_system_prompt()
    
    def _add_system_prompt(self):
        """Add the default system prompt as a pinned message."""
        system_content = get_master_system_prompt()
        system_message = ConversationMessage(
            role="system",
            content=system_content,
            timestamp=datetime.now(timezone.utc),
            token_count=self.token_counter.count_tokens(system_content),
            is_pinned=True
        )
        self.messages.append(system_message)
    
    def set_model(self, model_name: str):
        self.current_model = model_name
        if model_name not in self.MODEL_CONFIGS:
            self.MODEL_CONFIGS[model_name] = ModelConfig(f"Unknown-{model_name}", 32000)
    
    def get_model_config(self) -> ModelConfig:
        if not self.current_model:
            return ModelConfig("Default", 32000)
        return self.MODEL_CONFIGS.get(self.current_model, ModelConfig("Default", 32000))
    
    def add_message(self, role: str, content: str, is_pinned: bool = False) -> ConversationMessage:
        message = ConversationMessage(
            role=role,
            content=content,
            timestamp=datetime.now(timezone.utc),
            token_count=self.token_counter.count_tokens(content, self.current_model),
            is_pinned=is_pinned
        )
        self.messages.append(message)
        self._manage_buffer_size()
        return message
    
    def _calculate_total_tokens(self) -> int:
        return sum(msg.token_count for msg in self.messages)
    
    def _manage_buffer_size(self):
        config = self.get_model_config()
        max_tokens = config.max_tokens - config.reserve_tokens
        current_tokens = self._calculate_total_tokens()
        if current_tokens <= max_tokens:
            return
        summary_threshold = int(max_tokens * config.summary_threshold)
        if current_tokens > summary_threshold:
            self._truncate_with_summary(max_tokens)
        else:
            self._simple_truncate(max_tokens)
    
    def _simple_truncate(self, max_tokens: int):
        current_tokens = self._calculate_total_tokens()
        removable_indices = []
        for i, msg in enumerate(self.messages):
            if not msg.is_pinned and msg.role != "system":
                removable_indices.append(i)
        for i in sorted(removable_indices):
            if current_tokens <= max_tokens:
                break
            removed_msg = self.messages.pop(i)
            current_tokens -= removed_msg.token_count
            removable_indices = [idx - 1 if idx > i else idx for idx in removable_indices if idx != i]
    
    def _truncate_with_summary(self, max_tokens: int):
        current_tokens = self._calculate_total_tokens()
        pinned_messages = [msg for msg in self.messages if msg.is_pinned or msg.role == "system"]
        unpinned_messages = [msg for msg in self.messages if not msg.is_pinned and msg.role != "system"]
        if len(unpinned_messages) <= 2:
            self._simple_truncate(max_tokens)
            return
        pinned_tokens = sum(msg.token_count for msg in pinned_messages)
        target_unpinned_tokens = max_tokens - pinned_tokens
        messages_to_keep = []
        messages_to_summarize = []
        current_unpinned_tokens = 0
        for msg in reversed(unpinned_messages):
            if current_unpinned_tokens + msg.token_count <= target_unpinned_tokens:
                messages_to_keep.insert(0, msg)
                current_unpinned_tokens += msg.token_count
            else:
                messages_to_summarize.insert(0, msg)
        if messages_to_summarize:
            summary_content = self.summarizer.create_summary(messages_to_summarize)
            summary_message = ConversationMessage(
                role="system",
                content=f"[CONVERSATION SUMMARY] {summary_content}",
                timestamp=datetime.now(timezone.utc),
                token_count=self.token_counter.count_tokens(summary_content),
                is_summary=True,
                is_pinned=True
            )
            self.messages = pinned_messages + [summary_message] + messages_to_keep
    
    def get_conversation_buffer(self) -> List[Dict[str, Any]]:
        api_messages = []
        for msg in self.messages:
            api_message: Dict[str, Any] = {
                "role": msg.role,
                "content": msg.content
            }
            if hasattr(msg, 'is_summary') and msg.is_summary:
                api_message["_metadata"] = {"is_summary": True}
            api_messages.append(api_message)
        return api_messages
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        config = self.get_model_config()
        total_tokens = self._calculate_total_tokens()
        return {
            "session_id": self.session_id,
            "current_model": self.current_model,
            "total_messages": len(self.messages),
            "total_tokens": total_tokens,
            "max_tokens": config.max_tokens,
            "utilization_percent": round((total_tokens / config.max_tokens) * 100, 2),
            "pinned_messages": sum(1 for msg in self.messages if msg.is_pinned),
            "summary_messages": sum(1 for msg in self.messages if getattr(msg, 'is_summary', False))
        }
    
    def clear_conversation(self, keep_system_prompt: bool = True):
        if keep_system_prompt:
            self.messages = [msg for msg in self.messages if msg.is_pinned and msg.role == "system"]
        else:
            self.messages = []
            self._add_system_prompt()
    
    def export_conversation(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "current_model": self.current_model,
            "messages": [asdict(msg) for msg in self.messages],
            "stats": self.get_conversation_stats()
        }
    
    def import_conversation(self, data: Dict[str, Any]):
        self.session_id = data.get("session_id", "default")
        self.current_model = data.get("current_model")
        self.messages = []
        for msg_data in data.get("messages", []):
            if isinstance(msg_data.get("timestamp"), str):
                msg_data["timestamp"] = datetime.fromisoformat(msg_data["timestamp"].replace('Z', '+00:00'))
            self.messages.append(ConversationMessage(**msg_data))


# Global memory manager instances
_memory_managers: Dict[str, ConversationMemoryManager] = {}


def get_memory_manager(session_id: str = "default") -> ConversationMemoryManager:
    if session_id not in _memory_managers:
        _memory_managers[session_id] = ConversationMemoryManager(session_id)
    return _memory_managers[session_id]


def cleanup_old_sessions(max_sessions: int = 100):
    if len(_memory_managers) > max_sessions:
        oldest_sessions = sorted(_memory_managers.keys())[:len(_memory_managers) - max_sessions]
        for session_id in oldest_sessions:
            del _memory_managers[session_id]
