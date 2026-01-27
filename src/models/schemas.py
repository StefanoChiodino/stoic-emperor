from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
import uuid

class PsychUpdate(BaseModel):
    """Hidden psychological analysis layer"""
    detected_patterns: List[str] = Field(description="List of detected cognitive patterns or distortions")
    emotional_state: str = Field(description="Current emotional state of the user")
    stoic_principle_applied: str = Field(description="The Stoic principle relevant to this situation")
    suggested_next_direction: str = Field(description="Internal strategy note for the therapist")
    confidence: float = Field(description="Confidence score 0.0-1.0")

class EmperorResponse(BaseModel):
    """Full response from the Emperor"""
    response_text: str = Field(description="The visible response to the user")
    psych_update: PsychUpdate = Field(description="The hidden analysis")

class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    role: str  # 'user' or 'emperor'
    content: str
    psych_update: Optional[PsychUpdate] = None
    created_at: datetime = Field(default_factory=datetime.now)
    semantic_processed_at: Optional[datetime] = None


class SemanticInsight(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    source_message_id: str
    assertion: str
    confidence: float
    created_at: datetime = Field(default_factory=datetime.now)

class Session(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class User(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: Optional[str] = None
    password_hash: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

class CondensedSummary(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    level: int
    content: str
    period_start: datetime
    period_end: datetime
    source_message_count: int
    source_word_count: int
    source_summary_ids: List[str] = Field(default_factory=list)
    consensus_log: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.now)
