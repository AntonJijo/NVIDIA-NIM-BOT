#!/usr/bin/env python3
"""
Conversation Memory Manager for Multi-LLM Chatbot

This module provides intelligent conversation buffer management that adapts to
different LLM context sizes, maintains conversation history, and provides
automatic truncation with optional summarization.
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import tiktoken


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
        # Initialize tokenizer (using GPT-3.5 encoding as baseline)
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None
    
    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Estimate token count for given text.
        Uses tiktoken for accurate counting when available, falls back to approximation.
        """
        if not text:
            return 0
            
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        
        # Fallback approximation: ~4 characters per token
        return max(1, len(text) // 4)
    
    def count_message_tokens(self, message: Dict[str, str], model: Optional[str] = None) -> int:
        """Count tokens for a complete message including role overhead."""
        base_tokens = self.count_tokens(message.get('content', ''), model)
        
        # Add overhead for message structure (role, formatting, etc.)
        overhead = 10  # Conservative estimate for JSON structure
        
        return base_tokens + overhead


class ConversationSummarizer:
    """Generates compact summaries of conversation segments."""
    
    @staticmethod
    def create_summary(messages: List[ConversationMessage]) -> str:
        """
        Create a compact summary of multiple conversation messages.
        """
        if not messages:
            return ""
        
        # Extract key information from messages
        user_messages = [msg for msg in messages if msg.role == "user"]
        assistant_messages = [msg for msg in messages if msg.role == "assistant"]
        
        summary_parts = []
        
        if user_messages:
            # Summarize user topics/questions
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
            # Summarize assistant responses
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
    
    # Model configurations with their context limits
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
        """Initialize conversation memory manager for a specific session."""
        self.session_id = session_id
        self.messages: List[ConversationMessage] = []
        self.token_counter = TokenCounter()
        self.summarizer = ConversationSummarizer()
        self.current_model: Optional[str] = None
        
        # Initialize with system prompt (pinned)
        self._add_system_prompt()
    
    def _add_system_prompt(self):
        """Add the default system prompt as a pinned message."""
        system_content = """You are a highly professional, authoritative, and reliable AI assistant. When providing code examples, you MUST format them properly with markdown code blocks.

PROFESSIONAL COMMUNICATION GUIDELINES - FOLLOW THESE EXACTLY:

**Communication Standards:**
- Communicate clearly, concisely, and without ambiguity
- Maintain a consistently formal, courteous, and respectful tone
- Avoid offensive, inappropriate, or unprofessional language under all circumstances
- Prioritize user objectives and intentions in every response
- Adapt explanations to the user's knowledge level and context
- Deliver actionable, practical, and high-value insights wherever possible

**Content Quality:**
- Minimize filler words, redundancy, and irrelevant content
- Ensure all sentences are grammatically correct, well-structured, and polished
- Favor clarity and readability over verbosity or unnecessary complexity
- Provide information that is factually accurate, verifiable, and reliable
- Never fabricate, guess, or hallucinate information
- Clearly indicate uncertainty when present

**Formatting Requirements:**
- Use bullet points for enumerating items, examples, or options
- Use numbered lists for stepwise instructions or processes
- Highlight key terms or phrases with bold formatting
- Use italics selectively for emphasis or clarification
- Structure long responses into sections with clear headings or subheadings
- Keep paragraphs concise (2–4 sentences preferred)
- Maintain formatting consistency throughout responses

**Response Structure:**
- Conclude responses with actionable takeaways or recommended next steps
- Summarize key insights at the conclusion of explanations
- Provide direct answers before context, background, or elaboration
- End responses with a concise summary, actionable takeaway, or next step guidance

**Professional Standards:**
- Follow user instructions exactly, without deviation unless clarification is needed
- Handle incomplete, vague, or partially provided queries gracefully
- Maintain composure, neutrality, and professionalism in all interactions
- Acknowledge limitations or knowledge gaps transparently
- Correct any inaccuracies promptly and professionally

IMPORTANT: If you provide ANY code, it MUST be wrapped in proper markdown code blocks. No exceptions."""
        
        system_message = ConversationMessage(
            role="system",
            content=system_content,
            timestamp=datetime.now(timezone.utc),
            token_count=self.token_counter.count_tokens(system_content),
            is_pinned=True
        )
        
        self.messages.append(system_message)
    
    def set_model(self, model_name: str):
        """Set the current model and its configuration."""
        self.current_model = model_name
        if model_name not in self.MODEL_CONFIGS:
            # Default configuration for unknown models
            self.MODEL_CONFIGS[model_name] = ModelConfig(f"Unknown-{model_name}", 32000)
    
    def get_model_config(self) -> ModelConfig:
        """Get configuration for the current model."""
        if not self.current_model:
            return ModelConfig("Default", 32000)
        return self.MODEL_CONFIGS.get(self.current_model, ModelConfig("Default", 32000))
    
    def add_message(self, role: str, content: str, is_pinned: bool = False) -> ConversationMessage:
        """Add a new message to the conversation."""
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
        """Calculate total tokens in the current conversation buffer."""
        return sum(msg.token_count for msg in self.messages)
    
    def _manage_buffer_size(self):
        """Manage buffer size according to model limits."""
        config = self.get_model_config()
        max_tokens = config.max_tokens - config.reserve_tokens
        
        current_tokens = self._calculate_total_tokens()
        
        # If we're under the limit, no action needed
        if current_tokens <= max_tokens:
            return
        
        # Check if we should start summarizing
        summary_threshold = int(max_tokens * config.summary_threshold)
        
        if current_tokens > summary_threshold:
            self._truncate_with_summary(max_tokens)
        else:
            self._simple_truncate(max_tokens)
    
    def _simple_truncate(self, max_tokens: int):
        """Simple truncation by removing oldest non-pinned messages."""
        current_tokens = self._calculate_total_tokens()
        
        # Find removable messages (not pinned, not system)
        removable_indices = []
        for i, msg in enumerate(self.messages):
            if not msg.is_pinned and msg.role != "system":
                removable_indices.append(i)
        
        # Remove oldest messages until we're under the limit
        for i in sorted(removable_indices):
            if current_tokens <= max_tokens:
                break
            
            removed_msg = self.messages.pop(i)
            current_tokens -= removed_msg.token_count
            
            # Adjust indices for subsequent removals
            removable_indices = [idx - 1 if idx > i else idx for idx in removable_indices if idx != i]
    
    def _truncate_with_summary(self, max_tokens: int):
        """Truncate with summarization of removed content."""
        current_tokens = self._calculate_total_tokens()
        
        # Find messages to summarize (oldest non-pinned messages)
        pinned_messages = [msg for msg in self.messages if msg.is_pinned or msg.role == "system"]
        unpinned_messages = [msg for msg in self.messages if not msg.is_pinned and msg.role != "system"]
        
        if len(unpinned_messages) <= 2:
            # Not enough messages to summarize, use simple truncation
            self._simple_truncate(max_tokens)
            return
        
        # Calculate how many messages we need to remove
        pinned_tokens = sum(msg.token_count for msg in pinned_messages)
        target_unpinned_tokens = max_tokens - pinned_tokens
        
        # Find the split point
        messages_to_keep = []
        messages_to_summarize = []
        
        current_unpinned_tokens = 0
        for msg in reversed(unpinned_messages):  # Start from newest
            if current_unpinned_tokens + msg.token_count <= target_unpinned_tokens:
                messages_to_keep.insert(0, msg)
                current_unpinned_tokens += msg.token_count
            else:
                messages_to_summarize.insert(0, msg)
        
        # Create summary if we have messages to summarize
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
            
            # Rebuild messages list
            self.messages = pinned_messages + [summary_message] + messages_to_keep
    
    def get_conversation_buffer(self) -> List[Dict[str, Any]]:
        """
        Get the current conversation buffer formatted for LLM API calls.
        Returns messages in the format expected by OpenAI-compatible APIs.
        """
        api_messages = []
        
        for msg in self.messages:
            api_message: Dict[str, Any] = {
                "role": msg.role,
                "content": msg.content
            }
            
            # Add metadata for debugging (optional)
            if hasattr(msg, 'is_summary') and msg.is_summary:
                api_message["_metadata"] = {"is_summary": True}
            
            api_messages.append(api_message)
        
        return api_messages
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about the current conversation."""
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
        """Clear the conversation buffer."""
        if keep_system_prompt:
            # Keep only pinned system messages
            self.messages = [msg for msg in self.messages if msg.is_pinned and msg.role == "system"]
        else:
            self.messages = []
            self._add_system_prompt()
    
    def export_conversation(self) -> Dict[str, Any]:
        """Export conversation for persistence/debugging."""
        return {
            "session_id": self.session_id,
            "current_model": self.current_model,
            "messages": [asdict(msg) for msg in self.messages],
            "stats": self.get_conversation_stats()
        }
    
    def import_conversation(self, data: Dict[str, Any]):
        """Import conversation from exported data."""
        self.session_id = data.get("session_id", "default")
        self.current_model = data.get("current_model")
        
        self.messages = []
        for msg_data in data.get("messages", []):
            # Convert timestamp back to datetime object
            if isinstance(msg_data.get("timestamp"), str):
                msg_data["timestamp"] = datetime.fromisoformat(msg_data["timestamp"].replace('Z', '+00:00'))
            
            self.messages.append(ConversationMessage(**msg_data))


# Global memory manager instances (one per session)
_memory_managers: Dict[str, ConversationMemoryManager] = {}


def get_memory_manager(session_id: str = "default") -> ConversationMemoryManager:
    """Get or create a memory manager for a specific session."""
    if session_id not in _memory_managers:
        _memory_managers[session_id] = ConversationMemoryManager(session_id)
    return _memory_managers[session_id]


def cleanup_old_sessions(max_sessions: int = 100):
    """Clean up old sessions to prevent memory leaks."""
    if len(_memory_managers) > max_sessions:
        # Remove oldest sessions (simple cleanup strategy)
        oldest_sessions = sorted(_memory_managers.keys())[:len(_memory_managers) - max_sessions]
        for session_id in oldest_sessions:
            del _memory_managers[session_id]