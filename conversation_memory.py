#!/usr/bin/env python3
"""
Conversation Memory Manager for Multi-LLM Chatbot

This module provides:
- Intelligent conversation buffer management
- Context-size aware truncation & summarization
- Global persona rules (imported from system_prompts)
- Strict formatting enforcement for assistant messages
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

# SentencePiece is optional - if not available, we'll use fallback tokenization
try:
    import sentencepiece as spm  # type: ignore
except ImportError:
    spm = None
    print("Warning: SentencePiece not available, using fallback tokenization")

# Import strict global persona + formatting enforcement
from system_prompts import get_master_system_prompt, enforce_formatting


@dataclass
class ConversationMessage:
    """Represents a single message in the conversation."""
    role: str  # "system", "user", "assistant"
    content: str
    timestamp: datetime
    token_count: int = 0
    is_pinned: bool = False
    is_summary: bool = False


@dataclass
class ModelConfig:
    """Configuration for a specific LLM model."""
    name: str
    max_tokens: int
    reserve_tokens: int = 1000
    summary_threshold: float = 0.7


class TokenCounter:
    """Handles token counting for different models."""

    def __init__(self):
        self.tokenizer = None
        try:
            # Note: SentencePiece requires a trained model file to function
            # For now, we use an improved fallback estimation
            print("Info: Using improved token approximation (no trained model available)")
        except Exception as e:
            print(f"Warning: Could not initialize SentencePiece tokenizer: {e}")
            self.tokenizer = None

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        if not text:
            return 0
            
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text, out_type=int))
            except Exception as e:
                print(f"SentencePiece tokenization error: {e}")
                
        # Improved fallback: more accurate token estimation
        # Split by whitespace to approximate words, then estimate tokens per word
        words = text.split()
        word_count = len(words)
        
        # More accurate estimation:
        # - Average 1.3 tokens per word for English text
        # - Add tokens for punctuation and special characters
        # - Account for subword tokenization
        estimated_tokens = int(word_count * 1.3)
        
        # Add extra tokens for special characters, numbers, and punctuation
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        estimated_tokens += max(0, special_chars // 3)
        
        # Minimum of 1 token for non-empty text
        return max(1, estimated_tokens)

    def count_message_tokens(self, message: Dict[str, str], model: Optional[str] = None) -> int:
        return self.count_tokens(message.get("content", ""), model) + 10


class ConversationSummarizer:
    """Generates compact summaries of conversation segments."""

    @staticmethod
    def create_summary(messages: List[ConversationMessage]) -> str:
        if not messages:
            return ""
        user_msgs = [m for m in messages if m.role == "user"]
        asst_msgs = [m for m in messages if m.role == "assistant"]

        parts = []
        if user_msgs:
            topics = []
            for msg in user_msgs:
                snippet = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                topics.append(snippet)
            if len(topics) == 1:
                parts.append(f"User asked: {topics[0]}")
            else:
                parts.append(f"User discussed: {'; '.join(topics[:3])}")
                if len(topics) > 3:
                    parts[-1] += f" and {len(topics)-3} other topics"

        if asst_msgs:
            if len(asst_msgs) == 1:
                snippet = asst_msgs[0].content[:150]
                if len(asst_msgs[0].content) > 150:
                    snippet += "..."
                parts.append(f"Assistant responded: {snippet}")
            else:
                parts.append(f"Assistant provided {len(asst_msgs)} detailed responses")

        return " | ".join(parts)


class ConversationMemoryManager:
    """Main conversation memory manager with strict formatting."""

    MODEL_CONFIGS = {
        "meta/llama-4-maverick-17b-128e-instruct": ModelConfig("Llama 4 Maverick", 1_000_000),
        "deepseek-ai/deepseek-r1": ModelConfig("DeepSeek R1", 128_000),
        "qwen/qwen2.5-coder-32b-instruct": ModelConfig("Qwen 2.5 Coder", 32_000),
        "qwen/qwen3-coder-480b-a35b-instruct": ModelConfig("Qwen3 Coder 480B", 256_000),
        "deepseek-ai/deepseek-v3.1": ModelConfig("DeepSeek V3.1", 128_000),
        "openai/gpt-oss-120b": ModelConfig("GPT OSS", 128_000),
        "qwen/qwen3-235b-a22b:free": ModelConfig("Qwen3 235B", 131_000),
        "google/gemma-3-27b-it:free": ModelConfig("Gemma 3", 96_000),
        "x-ai/grok-4-fast:free": ModelConfig("Grok 4", 2_000_000),
    }

    def __init__(self, session_id: str = "default", fmt: str = "markdown"):
        self.session_id = session_id
        self.messages: List[ConversationMessage] = []
        self.token_counter = TokenCounter()
        self.summarizer = ConversationSummarizer()
        self.current_model: Optional[str] = None
        self.format = fmt  # "markdown" (default), "plaintext", "json", "yaml"

        # Add pinned global system rules
        self._add_system_prompt()

    def _add_system_prompt(self):
        system_content = get_master_system_prompt()
        system_msg = ConversationMessage(
            role="system",
            content=system_content,
            timestamp=datetime.now(timezone.utc),
            token_count=self.token_counter.count_tokens(system_content),
            is_pinned=True,
        )
        self.messages.append(system_msg)

    def set_model(self, model_name: str):
        self.current_model = model_name
        if model_name not in self.MODEL_CONFIGS:
            self.MODEL_CONFIGS[model_name] = ModelConfig(f"Unknown-{model_name}", 32_000)

    def get_model_config(self) -> ModelConfig:
        if not self.current_model:
            return ModelConfig("Default", 32_000)
        return self.MODEL_CONFIGS.get(self.current_model, ModelConfig("Default", 32_000))

    def add_message(self, role: str, content: str, is_pinned: bool = False) -> ConversationMessage:
        # Strictly enforce formatting only for assistant outputs
        if role == "assistant":
            content = enforce_formatting(content, self.format)

        msg = ConversationMessage(
            role=role,
            content=content,
            timestamp=datetime.now(timezone.utc),
            token_count=self.token_counter.count_tokens(content, self.current_model),
            is_pinned=is_pinned,
        )
        self.messages.append(msg)
        self._manage_buffer_size()
        return msg

    def _calculate_total_tokens(self) -> int:
        return sum(m.token_count for m in self.messages)

    def _manage_buffer_size(self):
        cfg = self.get_model_config()
        max_tokens = cfg.max_tokens - cfg.reserve_tokens
        total = self._calculate_total_tokens()
        if total <= max_tokens:
            return
        threshold = int(max_tokens * cfg.summary_threshold)
        if total > threshold:
            self._truncate_with_summary(max_tokens)
        else:
            self._simple_truncate(max_tokens)

    def _simple_truncate(self, max_tokens: int):
        total = self._calculate_total_tokens()
        removable = [i for i, m in enumerate(self.messages) if not m.is_pinned and m.role != "system"]
        for i in sorted(removable):
            if total <= max_tokens:
                break
            removed = self.messages.pop(i)
            total -= removed.token_count
            removable = [idx - 1 if idx > i else idx for idx in removable if idx != i]

    def _truncate_with_summary(self, max_tokens: int):
        total = self._calculate_total_tokens()
        pinned = [m for m in self.messages if m.is_pinned or m.role == "system"]
        unpinned = [m for m in self.messages if not m.is_pinned and m.role != "system"]

        if len(unpinned) <= 2:
            self._simple_truncate(max_tokens)
            return

        pinned_tokens = sum(m.token_count for m in pinned)
        target_unpinned = max_tokens - pinned_tokens

        keep, summarize, current = [], [], 0
        for m in reversed(unpinned):
            if current + m.token_count <= target_unpinned:
                keep.insert(0, m)
                current += m.token_count
            else:
                summarize.insert(0, m)

        if summarize:
            summary_text = self.summarizer.create_summary(summarize)
            summary_msg = ConversationMessage(
                role="system",
                content=f"[CONVERSATION SUMMARY] {summary_text}",
                timestamp=datetime.now(timezone.utc),
                token_count=self.token_counter.count_tokens(summary_text),
                is_summary=True,
                is_pinned=True,
            )
            self.messages = pinned + [summary_msg] + keep

    def get_conversation_buffer(self) -> List[Dict[str, Any]]:
        buf = []
        for m in self.messages:
            entry: Dict[str, Any] = {"role": m.role, "content": m.content}
            if getattr(m, "is_summary", False):
                entry["metadata"] = {"is_summary": True}
            buf.append(entry)
        return buf

    def get_conversation_stats(self) -> Dict[str, Any]:
        cfg = self.get_model_config()
        total = self._calculate_total_tokens()
        return {
            "session_id": self.session_id,
            "current_model": self.current_model,
            "total_messages": len(self.messages),
            "total_tokens": total,
            "max_tokens": cfg.max_tokens,
            "utilization_percent": round((total / cfg.max_tokens) * 100, 2),
            "pinned_messages": sum(1 for m in self.messages if m.is_pinned),
            "summary_messages": sum(1 for m in self.messages if getattr(m, "is_summary", False)),
        }

    def clear_conversation(self, keep_system_prompt: bool = True):
        if keep_system_prompt:
            self.messages = [m for m in self.messages if m.is_pinned and m.role == "system"]
        else:
            self.messages = []
            self._add_system_prompt()

    def export_conversation(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "current_model": self.current_model,
            "messages": [asdict(m) for m in self.messages],
            "stats": self.get_conversation_stats(),
        }

    def import_conversation(self, data: Dict[str, Any]):
        self.session_id = data.get("session_id", "default")
        self.current_model = data.get("current_model")
        self.messages = []
        for m in data.get("messages", []):
            if isinstance(m.get("timestamp"), str):
                m["timestamp"] = datetime.fromisoformat(m["timestamp"].replace("Z", "+00:00"))
            self.messages.append(ConversationMessage(**m))


# Global memory managers
_memory_managers: Dict[str, ConversationMemoryManager] = {}


def get_memory_manager(session_id: str = "default", fmt: str = "markdown") -> ConversationMemoryManager:
    if session_id not in _memory_managers:
        _memory_managers[session_id] = ConversationMemoryManager(session_id, fmt=fmt)
    return _memory_managers[session_id]


def cleanup_old_sessions(max_sessions: int = 100):
    if len(_memory_managers) > max_sessions:
        oldest = sorted(_memory_managers.keys())[: len(_memory_managers) - max_sessions]
        for sid in oldest:
            del _memory_managers[sid]
