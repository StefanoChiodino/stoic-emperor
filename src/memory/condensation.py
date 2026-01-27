from typing import List, Dict, Any, Optional
from datetime import datetime
import tiktoken

from src.models.schemas import Message, CondensedSummary
from src.infrastructure.database import Database
from src.core.aegean_consensus import AegeanConsensusProtocol
from src.utils.config import load_config


class CondensationManager:
    def __init__(self, db: Database, config: Dict[str, Any]):
        self.db = db
        self.config = config
        self.condensation_config = config.get("condensation", {})
        self.hot_buffer_tokens = self.condensation_config.get("hot_buffer_tokens", 4000)
        self.chunk_threshold_tokens = self.condensation_config.get("chunk_threshold_tokens", 8000)
        self.summary_budget_tokens = self.condensation_config.get("summary_budget_tokens", 12000)
        self.use_consensus = self.condensation_config.get("use_consensus", True)
        
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")

    def estimate_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def get_uncondensed_messages(self, user_id: str) -> List[Message]:
        recent = self.db.get_recent_messages(user_id, limit=100)
        
        if len(recent) <= 1:
            return []
        
        hot_buffer_count = 0
        hot_buffer_tokens = 0
        for msg in reversed(recent):
            tokens = self.estimate_tokens(msg.content)
            if hot_buffer_tokens + tokens <= self.hot_buffer_tokens:
                hot_buffer_count += 1
                hot_buffer_tokens += tokens
            else:
                break
        
        cutoff_date = recent[-hot_buffer_count].created_at if hot_buffer_count > 0 else datetime.now()
        
        summaries = self.db.get_condensed_summaries(user_id)
        latest_summary_end = None
        if summaries:
            latest_summary_end = max(s.period_end for s in summaries)
        
        start_date = latest_summary_end if latest_summary_end else None
        uncondensed = self.db.get_messages_in_range(
            user_id, 
            start_date=start_date, 
            end_date=cutoff_date
        )
        
        if start_date:
            uncondensed = [m for m in uncondensed if m.created_at > start_date]
        
        return uncondensed

    def should_condense(self, user_id: str) -> bool:
        uncondensed = self.get_uncondensed_messages(user_id)
        if not uncondensed:
            return False
        
        total_tokens = sum(self.estimate_tokens(msg.content) for msg in uncondensed)
        return total_tokens >= self.chunk_threshold_tokens

    def condense_chunk(self, user_id: str, messages: List[Message]) -> Optional[CondensedSummary]:
        if not messages:
            return None
        
        period_start = messages[0].created_at
        period_end = messages[-1].created_at
        message_count = len(messages)
        word_count = sum(len(msg.content.split()) for msg in messages)
        
        messages_text = "\n\n".join([
            f"[{msg.created_at.strftime('%Y-%m-%d %H:%M')}] {msg.role.upper()}: {msg.content}"
            for msg in messages
        ])
        
        summaries = self.db.get_condensed_summaries(user_id)
        previous_context = ""
        if summaries:
            recent_summaries = sorted(summaries, key=lambda s: s.period_end, reverse=True)[:3]
            previous_context = "\n\n".join([
                f"Previous period ({s.period_start.strftime('%Y-%m-%d')} to {s.period_end.strftime('%Y-%m-%d')}): {s.content[:500]}..."
                for s in recent_summaries
            ])
        
        prompts = self.config.get("prompts", {})
        condensation_prompt = prompts.get("condensation", "")
        
        if not condensation_prompt:
            raise ValueError("Condensation prompt not found in config")
        
        prompt_text = condensation_prompt.format(
            period_start=period_start.strftime('%Y-%m-%d'),
            period_end=period_end.strftime('%Y-%m-%d'),
            message_count=message_count,
            word_count=word_count,
            previous_context=previous_context if previous_context else "None (this is the first summary)",
            messages=messages_text
        )
        
        if self.use_consensus:
            consensus = AegeanConsensusProtocol(
                model_a=self.config["models"]["main"],
                model_b=self.config["models"]["reviewer"],
                prompts={"condensation": condensation_prompt},
                beta_threshold=2,
                verbose=False
            )
            
            result = consensus.reach_consensus(
                prompt_name="condensation",
                variables={
                    "period_start": period_start.strftime('%Y-%m-%d'),
                    "period_end": period_end.strftime('%Y-%m-%d'),
                    "message_count": message_count,
                    "word_count": word_count,
                    "previous_context": previous_context if previous_context else "None",
                    "messages": messages_text,
                    "source_data": messages_text
                },
                temperature=0.7
            )
            
            content = result.final_output
            consensus_log = result.to_dict()
        else:
            from openai import OpenAI
            import os
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=self.config["models"]["main"],
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.7,
                max_tokens=2000
            )
            content = response.choices[0].message.content.strip()
            consensus_log = None
        
        summary = CondensedSummary(
            user_id=user_id,
            level=1,
            content=content,
            period_start=period_start,
            period_end=period_end,
            source_message_count=message_count,
            source_word_count=word_count,
            source_summary_ids=[],
            consensus_log=consensus_log
        )
        
        return self.db.save_condensed_summary(summary)

    def should_recurse(self, user_id: str, level: int = 1) -> bool:
        summaries = self.db.get_condensed_summaries(user_id, level=level)
        if len(summaries) <= 1:
            return False
        
        total_tokens = sum(self.estimate_tokens(s.content) for s in summaries)
        return total_tokens > self.summary_budget_tokens

    def condense_summaries(self, user_id: str, level: int) -> Optional[CondensedSummary]:
        summaries = self.db.get_condensed_summaries(user_id, level=level)
        if len(summaries) <= 1:
            return None
        
        summaries.sort(key=lambda s: s.period_start)
        
        batch_size = max(2, len(summaries) // 2)
        batch = summaries[:batch_size]
        
        period_start = batch[0].period_start
        period_end = batch[-1].period_end
        total_message_count = sum(s.source_message_count for s in batch)
        total_word_count = sum(s.source_word_count for s in batch)
        
        summaries_text = "\n\n".join([
            f"[Period {s.period_start.strftime('%Y-%m-%d')} to {s.period_end.strftime('%Y-%m-%d')}, Level {s.level}]:\n{s.content}"
            for s in batch
        ])
        
        prompts = self.config.get("prompts", {})
        condensation_prompt = prompts.get("condensation", "")
        
        prompt_text = condensation_prompt.format(
            period_start=period_start.strftime('%Y-%m-%d'),
            period_end=period_end.strftime('%Y-%m-%d'),
            message_count=total_message_count,
            word_count=total_word_count,
            previous_context=f"Condensing {len(batch)} Level {level} summaries",
            messages=summaries_text
        )
        
        if self.use_consensus:
            consensus = AegeanConsensusProtocol(
                model_a=self.config["models"]["main"],
                model_b=self.config["models"]["reviewer"],
                prompts={"condensation": condensation_prompt},
                beta_threshold=2,
                verbose=False
            )
            
            result = consensus.reach_consensus(
                prompt_name="condensation",
                variables={
                    "period_start": period_start.strftime('%Y-%m-%d'),
                    "period_end": period_end.strftime('%Y-%m-%d'),
                    "message_count": total_message_count,
                    "word_count": total_word_count,
                    "previous_context": f"Condensing Level {level} summaries",
                    "messages": summaries_text,
                    "source_data": summaries_text
                },
                temperature=0.7
            )
            
            content = result.final_output
            consensus_log = result.to_dict()
        else:
            from openai import OpenAI
            import os
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=self.config["models"]["main"],
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.7,
                max_tokens=2000
            )
            content = response.choices[0].message.content.strip()
            consensus_log = None
        
        new_summary = CondensedSummary(
            user_id=user_id,
            level=level + 1,
            content=content,
            period_start=period_start,
            period_end=period_end,
            source_message_count=total_message_count,
            source_word_count=total_word_count,
            source_summary_ids=[s.id for s in batch],
            consensus_log=consensus_log
        )
        
        return self.db.save_condensed_summary(new_summary)

    def get_context_summaries(self, user_id: str, token_budget: int) -> List[CondensedSummary]:
        summaries = self.db.get_condensed_summaries(user_id)
        if not summaries:
            return []
        
        summaries_by_level = {}
        for s in summaries:
            if s.level not in summaries_by_level:
                summaries_by_level[s.level] = []
            summaries_by_level[s.level].append(s)
        
        max_level = max(summaries_by_level.keys())
        
        selected = []
        current_tokens = 0
        
        for level in range(max_level, 0, -1):
            level_summaries = sorted(summaries_by_level.get(level, []), key=lambda s: s.period_start)
            
            for summary in level_summaries:
                covered_by_higher = any(
                    s.period_start <= summary.period_start and s.period_end >= summary.period_end
                    for s in selected
                )
                
                if covered_by_higher:
                    continue
                
                tokens = self.estimate_tokens(summary.content)
                if current_tokens + tokens <= token_budget:
                    selected.append(summary)
                    current_tokens += tokens
        
        return sorted(selected, key=lambda s: s.period_start)

    def maybe_condense(self, user_id: str, verbose: bool = False) -> bool:
        if self.should_condense(user_id):
            if verbose:
                print("Condensing messages into Level 1 summary...")
            
            uncondensed = self.get_uncondensed_messages(user_id)
            if uncondensed:
                self.condense_chunk(user_id, uncondensed)
                
                level = 1
                max_levels = 10
                while level < max_levels and self.should_recurse(user_id, level):
                    if verbose:
                        print(f"Recursively condensing Level {level} summaries...")
                    self.condense_summaries(user_id, level)
                    level += 1
                
                return True
        
        return False
