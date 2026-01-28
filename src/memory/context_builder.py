from typing import List, Dict, Any, Optional
from datetime import datetime
import tiktoken

from src.models.schemas import Message, CondensedSummary
from src.infrastructure.database import Database
from src.memory.condensation import CondensationManager


class ContextBuilder:
    def __init__(self, db: Database, config: Dict[str, Any]):
        self.db = db
        self.config = config
        self.condensation_config = config.get("condensation", {})
        self.hot_buffer_tokens = self.condensation_config.get("hot_buffer_tokens", 4000)
        self.summary_budget_tokens = self.condensation_config.get("summary_budget_tokens", 12000)
        self.max_context_tokens = config.get("memory", {}).get("max_context_tokens", 4000)

        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        self.condensation_manager = CondensationManager(db, config)

    def estimate_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def build_context(
        self,
        user_id: str,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        if max_tokens is None:
            max_tokens = self.max_context_tokens

        recent_messages = self._get_hot_buffer(user_id)
        recent_tokens = sum(self.estimate_tokens(msg.content) for msg in recent_messages)

        summary_budget = max_tokens - recent_tokens
        summaries = []

        if summary_budget > 0:
            summaries = self.condensation_manager.get_context_summaries(
                user_id,
                token_budget=summary_budget
            )

        return {
            "recent_messages": recent_messages,
            "condensed_summaries": summaries,
            "total_tokens": recent_tokens + sum(self.estimate_tokens(s.content) for s in summaries),
            "hot_buffer_tokens": recent_tokens,
            "summary_tokens": sum(self.estimate_tokens(s.content) for s in summaries)
        }

    def format_context_string(self, context: Dict[str, Any]) -> str:
        parts = []

        summaries: List[CondensedSummary] = context.get("condensed_summaries", [])
        if summaries:
            parts.append("## Historical Context (Condensed Summaries)")
            for summary in summaries:
                parts.append(
                    f"\n### Period: {summary.period_start.strftime('%Y-%m-%d')} to "
                    f"{summary.period_end.strftime('%Y-%m-%d')} "
                    f"(Level {summary.level}, {summary.source_message_count} messages)\n"
                    f"{summary.content}"
                )

        recent: List[Message] = context.get("recent_messages", [])
        if recent:
            parts.append("\n## Recent Conversation (Hot Buffer)")
            for msg in recent:
                parts.append(
                    f"\n[{msg.created_at.strftime('%Y-%m-%d %H:%M')}] "
                    f"{msg.role.upper()}: {msg.content}"
                )

        return "\n".join(parts)

    def _get_hot_buffer(self, user_id: str) -> List[Message]:
        all_messages = self.db.get_recent_messages(user_id, limit=100)

        hot_buffer = []
        total_tokens = 0

        for msg in reversed(all_messages):
            msg_tokens = self.estimate_tokens(msg.content)
            if total_tokens + msg_tokens <= self.hot_buffer_tokens:
                hot_buffer.insert(0, msg)
                total_tokens += msg_tokens
            else:
                break

        return hot_buffer

    def get_summary_statistics(self, user_id: str) -> Dict[str, Any]:
        summaries = self.db.get_condensed_summaries(user_id)

        if not summaries:
            return {
                "total_summaries": 0,
                "levels": {},
                "total_messages_condensed": 0,
                "total_words_condensed": 0,
                "earliest_period": None,
                "latest_period": None
            }

        levels = {}
        for s in summaries:
            if s.level not in levels:
                levels[s.level] = 0
            levels[s.level] += 1

        return {
            "total_summaries": len(summaries),
            "levels": levels,
            "total_messages_condensed": sum(s.source_message_count for s in summaries),
            "total_words_condensed": sum(s.source_word_count for s in summaries),
            "earliest_period": min(s.period_start for s in summaries),
            "latest_period": max(s.period_end for s in summaries)
        }
